import tushare as ts
import pandas as pd
import torch
import numpy as np
import plotly.graph_objects as go
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas_market_calendars as mcal
import streamlit as st
import datetime

# 设置Tushare的Token（请使用你的个人Token）
pro = ts.pro_api()
cn_market = mcal.get_calendar('SSE')

from pandas.api.types import is_numeric_dtype

class StockData(Dataset):
    def __init__(self, data, step=30, predict_value='close'):
        """
        data: 已经包含目标股票和辅助股票特征的 DataFrame
        step: 时间序列长度
        """
        self.step = step
        # 仅提取数值型特征列，并排除trade_date
        numeric_cols = [col for col in data.columns if col != 'trade_date' and is_numeric_dtype(data[col])]
        self.features = numeric_cols
        
        data_arr = data[self.features].values
        self.len = len(data_arr)

        # 确保predict_value在self.features中
        if predict_value not in self.features:
            raise ValueError(f"预测列 {predict_value} 不存在于特征列中，请检查数据。")

        self.close_col_idx = self.features.index(predict_value)
        self.close_max = data_arr[:, self.close_col_idx].max()
        self.close_min = data_arr[:, self.close_col_idx].min()

        data_norm = self.normalize(data_arr)
        self.X = torch.tensor(data_norm.astype(np.float32))
        self.y = torch.tensor(data_norm[:, self.close_col_idx].astype(np.float32))  # predict_value对应的目标列

    def __getitem__(self, index):
        return self.X[index:index + self.step], self.y[index + self.step]

    def __len__(self):
        return self.len - 1 - self.step

    def normalize(self, data):
        data = data.T
        for i in range(len(data)):
            data_min = data[i].min()
            data_max = data[i].max()
            data[i] = (data[i] - data_min) / (data_max - data_min + 1e-9)
        return data.T


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        batch_size, seq_len, hidden_dim = lstm_outputs.size()
        attn_scores = self.attention_weights(lstm_outputs.reshape(-1, hidden_dim))
        attn_scores = attn_scores.reshape(batch_size, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * lstm_outputs, dim=1)
        return context_vector

class Net(nn.Module):
    def __init__(self, input_size=12, hidden_size=30, output_size=1, num_layers=3, dropout=0.3):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X, _ = self.lstm(X)
        X = self.dropout(X[:, -1, :])  
        X = self.linear(X)
        return X

class NetWithAttention(nn.Module):
    def __init__(self, input_size=12, hidden_size=30, output_size=1, num_layers=3, dropout=0.3):
        super(NetWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.attention = Attention(hidden_size * 2)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        lstm_out, _ = self.lstm(X)
        context_vector = self.attention(lstm_out)
        context_vector = self.dropout(context_vector)
        output = self.linear(context_vector)
        return output

def load_single_stock_data(name, start_time):
    start_time = start_time.replace("-", "")
    df = pro.daily(ts_code=name, start_date=start_time)
    if df.empty:
        raise ValueError("股票数据为空，请检查股票代码和起始日期。")
    # 提取所需的特征列(可根据实际需求调整)
    df = df[['trade_date','open','high','close','low','vol','pct_chg','amount','change']]
    df['trade_date'] = df['trade_date'].astype(str)
    return df

def load_data(main_stock, start_time, data_file='filled_merged_data.csv'):
    """
    加载目标股票数据 + 宏观数据并合并
    """
    # 主股票数据
    df_main = load_single_stock_data(main_stock, start_time)
    
    # 宏观数据
    df_macro = pd.read_csv(data_file)
    df_macro['trade_date'] = df_macro['trade_date'].astype(str)

    # 将主股票与宏观数据合并
    df_merged = pd.merge(df_macro, df_main, on='trade_date')

    df_merged = df_merged.sort_values(by='trade_date', ascending=True)
    # 增加目标股票的mean列 (可根据需求增删)
    df_merged['mean'] = (df_merged['open'] + df_merged['close']) / 2
    df_merged = df_merged.reset_index(drop=True)

    # 对缺失数据进行适当填充（如果有）
    df_merged = df_merged.fillna(method='ffill').fillna(method='bfill')

    return df_merged

def train_model(model, dataloader, epochs=10, step=30, lr=0.001, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.9)
    model.to(device)

    # 使用dataloader大小的80%作为训练集
    # len(dataloader)是batch数，若batch为1，则无法划分
    # 建议在DataLoader中设置较小batch_size从而有多个batch
    train_len = int(len(dataloader) * 0.8)
    if train_len < 1:
        train_len = len(dataloader)  # 如果只有一个batch，那就全用于训练

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i, (X, y) in enumerate(dataloader):
            if i == train_len:
                break
            X, y = X.to(device), y.to(device)
            X = X.view(-1, step, X.size(-1))
            y = y.unsqueeze(1)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if train_len > 0:
            st.write(f'Epoch: {epoch + 1}, Loss: {total_loss/train_len:.6f}')

    return model

def evaluate_model(model, dataloader, df, step=30, device='cpu',predict_value='close'):
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.view(-1, step, X.size(-1))
            pred = model(X).data.squeeze().cpu().numpy()
            preds.append(pred)
            actuals.append(y.cpu().numpy())

    preds = np.concatenate([p.reshape(-1) for p in preds])
    actuals = np.concatenate(actuals)

    close_max = df[predict_value].max()
    close_min = df[predict_value].min()
    preds = preds * (close_max - close_min) + close_min
    actuals = actuals * (close_max - close_min) + close_min

    return preds, actuals

def predict_future(model, dataset, future_days=5, device='cpu',predict_value='close'):
    model.eval()
    last_input = dataset.X[-dataset.step:, :].unsqueeze(0).to(device)
    future_preds = []
    close_index = dataset.features.index(predict_value)

    with torch.no_grad():
        current_input = last_input
        for _ in range(future_days):
            pred = model(current_input)
            future_preds.append(pred.item())
            new_step = current_input[:, -1, :].clone().squeeze(0)
            new_step[close_index] = pred
            new_step = new_step.unsqueeze(0)
            current_input = torch.cat([current_input, new_step.unsqueeze(0)], dim=1)[:, 1:, :]

    close_max = dataset.close_max
    close_min = dataset.close_min
    future_preds = np.array(future_preds) * (close_max - close_min) + close_min
    return future_preds

def validate_and_tune_model(model, dataset, validation_steps=10, step=30, device='cpu', predict_value='close'):
    model.eval()
    validation_results = []
    # 若数据量不够，这里可能会有越界问题，请根据数据长度微调validation_steps
    available_steps = len(dataset) // dataset.step
    validation_steps = min(validation_steps, available_steps - 1)

    for i in range(validation_steps):
        start_idx = -dataset.step * (i + 2)
        end_idx = -dataset.step * (i + 1)
        X = dataset.X[start_idx:end_idx, :].unsqueeze(0).to(device)
        y_actual = dataset.y[end_idx - 1].item()

        with torch.no_grad():
            y_pred = model(X).item()

        y_pred = y_pred * (dataset.close_max - dataset.close_min) + dataset.close_min
        y_actual = y_actual * (dataset.close_max - dataset.close_min) + dataset.close_min

        validation_results.append((y_actual, y_pred))

    errors = [abs(actual - pred) for actual, pred in validation_results]
    if len(errors) > 0:
        avg_error = sum(errors) / len(errors)
        st.write(f'Validation Average Error: {avg_error:.6f}')

    return model

def plot_results(df, name, preds, actuals, future_preds=None, predict_value='close'):
    # step这里用于对齐预测值和真实值，但请确保preds与actuals长度一致
    step = len(df) - len(preds)
    if step < 0:
        step = 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['trade_date'].iloc[step:], y=preds, name='Prediction'))
    fig.add_trace(go.Scatter(x=df['trade_date'].iloc[step:], y=actuals, name='Actual'))
    if 'change' in df.columns:
        fig.add_trace(go.Scatter(x=df['trade_date'].iloc[step:], y=df['change'].iloc[step:], name='Change'))

    if future_preds is not None:
        last_date = pd.to_datetime(df['trade_date'].iloc[-1])
        # 使用工作日频率生成未来日期
        future_dates = pd.date_range(start=last_date, periods=len(future_preds)+1, freq='B')[1:]
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_preds,
            name='Future Predictions'
        ))

    fig.update_layout(title=f'{name} Prediction ({predict_value})',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      template='ggplot2',
                      width=1000,
                      height=600)
    return fig

def get_related_stocks(name, trade_date='20211012', top_n=50):
    df_bak = pro.bak_basic(trade_date=trade_date, fields='trade_date,ts_code,name,industry,pe')
    if df_bak.empty:
        raise ValueError("基础股票数据为空，请检查日期和请求参数。")

    main_info = df_bak[df_bak['ts_code'] == name]
    if main_info.empty:
        raise ValueError(f"未在bak_basic中找到主股票 {name} 的信息。")

    main_industry = main_info.iloc[0]['industry']
    same_industry_stocks = df_bak[df_bak['industry'] == main_industry]
    same_industry_stocks = same_industry_stocks[same_industry_stocks['ts_code'] != name]
    same_industry_stocks = same_industry_stocks.sort_values(by='pe', ascending=True)
    related_stocks = same_industry_stocks['ts_code'].head(top_n).tolist()

    return related_stocks

def stock_prediction(api,name, start_time, future_days_len=5, step=30, epoch=10, predict_value='close', model_type='Net', related_stocks='Yes',related_stocks_number=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载主股票数据（包含宏观数据）
    ts.set_token(api)
    pro = ts.pro_api()
    df_main = load_data(name, start_time)
    df_merged = df_main
    if related_stocks=='Yes':
        related_stocks_code=get_related_stocks(name=stock_code,top_n=related_stocks_number)
        for stk in  related_stocks_code:
            df_stk = load_data(stk, start_time)
            # 重命名相关股票的列
            feature_cols = [c for c in df_stk.columns if c not in ['trade_date']]
            rename_dict = {col: f"{col}_{stk}" for col in feature_cols}
            df_stk = df_stk.rename(columns=rename_dict)
            # 基于trade_date将相关股票的数据合并到df_merged中(横向合并)
            df_merged = pd.merge(df_merged, df_stk, on='trade_date', how='left')

        # 对缺失数据进行填充
        df_merged = df_merged.fillna(method='ffill').fillna(method='bfill')

    # 用合并后的df构建数据集
    df = df_merged
    dataset = StockData(data=df, step=step, predict_value=predict_value)
    input_size = dataset.X.size(-1)

    # 使用固定batch_size，确保有多个batch（如果数据太少，请根据情况调整）
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    if model_type == 'Net':
        net = Net(input_size=input_size)
    else:
        net = NetWithAttention(input_size=input_size)

    trained_model = train_model(net, data_loader, epochs=epoch, step=step, device=device)
    trained_model = validate_and_tune_model(trained_model, dataset, validation_steps=10, step=step, device=device, predict_value=predict_value)
    preds, actuals = evaluate_model(trained_model, data_loader, df=df, step=step, device=device, predict_value=predict_value)
    future_preds = predict_future(trained_model, dataset, future_days=future_days_len, device=device, predict_value=predict_value)

    fig = plot_results(df, name, preds, actuals, future_preds, predict_value=predict_value)
    return preds, actuals, future_preds, fig

# ------------------- Streamlit部分 -------------------
st.title("价格预测模型 by dayu")
api_type=st.text_input("Tushare api:", value='1a3e0639da9b72985daf214412e5f964db261dd6b036fdf29bf23c05')
stock_code = st.text_input("请输入股票代码(形如'000001.SZ'):", value="000001.SZ")
start_date = st.date_input("选择起始日期:", value=datetime.date(2023,1,1))
future_days = st.number_input("预测未来天数:", min_value=1, value=5)
step = st.number_input("输入时间窗口长度step:", min_value=1, value=5)
epoch = st.number_input("训练轮数epoch:", min_value=1, value=10)
predict_value = st.selectbox("预测值选择:", ["close", "open", "high", "low"])
model_type = st.selectbox("选择模型类型:", ["Net", "NetWithAttention"])
related_stocks_type=st.selectbox("是否选择相关板块的股票:", ["Yes", "NO"])
related_stocks_number_type=st.number_input("相关板块的股票数量:", min_value=10, value=20)
# 示例相关股票列表（可根据您的需求获取，也可留空）


if st.button("开始预测"):
    with st.spinner("模型训练中，请稍候..."):
        try:
            preds, actuals, future_preds, fig = stock_prediction(api=api_type,
                                                                 name=stock_code,
                                                                start_time=start_date.strftime("%Y%m%d"),
                                                                future_days_len=future_days,
                                                                step=step,
                                                                epoch=epoch,
                                                                predict_value=predict_value,
                                                                model_type=model_type,
                                                                related_stocks=related_stocks_type,
                                                                related_stocks_number=related_stocks_number_type
                                                                ) # 如果需要相关股票，在此传入列表
            st.success("预测完成！")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"出现错误: {e}")
