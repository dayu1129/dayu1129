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
ts.set_token('1a3e0639da9b72985daf214412e5f964db261dd6b036fdf29bf23c05')
pro = ts.pro_api()

# 创建中国股市的交易日历（上证所）
cn_market = mcal.get_calendar('SSE')

class StockData(Dataset):
    def __init__(self, data, step=30,predict_value='close'):
        """
        data: 包含特征和目标值的数据集 DataFrame
        step: 时间序列长度
        """
        self.step = step
        # 特征列（请根据你的数据列实际情况进行调整）
        self.features = ['open', 'high', 'close', 'low', 'vol', 'pct_chg', 'amount', 
                         'mean', 'change', 'RIFSPFF_N.WW', 'RIFSPBLP_N.WW', 'RIFSRP_F02_N.WW']
        data_arr = data[self.features].values
        self.len = len(data_arr)

        # 记录close列最大最小值用于反归一化
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
        """
        简单归一化到[0,1]
        """
        data = data.T
        for i in range(len(data)):
            data_min = data[i].min()
            data_max = data[i].max()
            data[i] = (data[i] - data_min) / (data_max - data_min + 1e-9)
        return data.T

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

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = lstm_outputs.size()
        # 计算注意力分数
        attn_scores = self.attention_weights(lstm_outputs.reshape(-1, hidden_dim))
        attn_scores = attn_scores.reshape(batch_size, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * lstm_outputs, dim=1)
        return context_vector


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
        # 使用注意力层获取上下文向量
        context_vector = self.attention(lstm_out)
        context_vector = self.dropout(context_vector)
        output = self.linear(context_vector)
        return output


def load_data(name, start_time, data_file='filled_merged_data.csv'):
    """
    从Tushare获取指定股票数据并与本地宏观数据合并
    """
    start_time = start_time.replace("-", "")
    df = pro.daily(ts_code=name, start_date=start_time)
    df_macro = pd.read_csv(data_file)
    if df.empty:
        raise ValueError("股票数据为空，请检查股票代码和起始日期。")

    df_macro['trade_date'] = df_macro['trade_date'].astype(str)
    df['trade_date'] = df['trade_date'].astype(str)
    df_merged = pd.merge(df_macro, df, on='trade_date')

    df_merged = df_merged[['trade_date', 'open', 'high', 'close', 'low', 'vol', 'pct_chg',
                           'amount', 'change', 'RIFSPFF_N.WW', 'RIFSPBLP_N.WW', 'RIFSRP_F02_N.WW']]
    df_merged = df_merged.sort_values(by='trade_date', ascending=True)
    df_merged['mean'] = (df_merged['open'] + df_merged['close']) / 2
    df_merged = df_merged.reset_index(drop=True)

    return df_merged

def train_model(model, dataloader, epochs=10, step=30, lr=0.001, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.9)
    model.to(device)

    # 使用80%数据作为训练，其余为验证
    train_len = int(len(dataloader) * 0.8)

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

    # 反归一化未来预测
    close_max = dataset.close_max
    close_min = dataset.close_min
    future_preds = np.array(future_preds) * (close_max - close_min) + close_min
    return future_preds

def validate_and_tune_model(model, dataset, validation_steps=10, step=30, device='cpu', predict_value='close'):
    model.eval()
    validation_results = []
    for i in range(validation_steps):
        start_idx = -dataset.step * (i + 2)
        end_idx = -dataset.step * (i + 1)
        X = dataset.X[start_idx:end_idx, :].unsqueeze(0).to(device)
        y_actual = dataset.y[end_idx - 1].item()

        with torch.no_grad():
            y_pred = model(X).item()

        # 反归一化
        y_pred = y_pred * (dataset.close_max - dataset.close_min) + dataset.close_min
        y_actual = y_actual * (dataset.close_max - dataset.close_min) + dataset.close_min

        validation_results.append((y_actual, y_pred))

    # 根据验证结果计算误差并优化
    errors = [abs(actual - pred) for actual, pred in validation_results]
    avg_error = sum(errors) / len(errors)
    st.write(f'Validation Average Error: {avg_error:.6f}')

    return model

def plot_results(df,name, preds, actuals, future_preds=None, predict_value='close'):
    step = len(df) - len(preds)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['trade_date'].iloc[step:], y=preds, name='Prediction'))
    fig.add_trace(go.Scatter(x=df['trade_date'].iloc[step:], y=actuals, name='Actual'))
    fig.add_trace(go.Scatter(x=df['trade_date'].iloc[step:], y=df['change'], name='Change'))

    # 预测未来的日期（使用工作日频率）
    if future_preds is not None:
        last_date = pd.to_datetime(df['trade_date'].iloc[-1])
        future_dates = pd.bdate_range(start=last_date, periods=len(future_preds)+1)[1:]
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

def stock_prediction(name, start_time, future_days_len=5, step=30, epoch=10, predict_value='close', model_type='Net'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = load_data(name, start_time)
    dataset = StockData(data=df, step=step, predict_value=predict_value)
    input_size = dataset.X.size(-1)  # 动态获取输入维度

    # DataLoader
    data_loader = DataLoader(dataset=dataset, batch_size=len(df.T)-1, shuffle=False)

    # 根据选择的模型类型初始化模型
    if model_type == 'Net':
        net = Net(input_size=input_size)
    else:
        net = NetWithAttention(input_size=input_size)

    # 训练模型
    trained_model = train_model(net, data_loader, epochs=epoch, step=step, device=device)

    # 验证并调整模型
    trained_model = validate_and_tune_model(trained_model, dataset, validation_steps=10, step=step, device=device, predict_value=predict_value)

    # 评估模型
    preds, actuals = evaluate_model(trained_model, data_loader, df=df, step=step, device=device, predict_value=predict_value)

    # 预测未来
    future_preds = predict_future(trained_model, dataset, future_days=future_days_len, device=device, predict_value=predict_value)

    # 绘图结果
    fig = plot_results(df,name, preds, actuals, future_preds, predict_value=predict_value)

    return preds, actuals, future_preds, fig

# ------------------- Streamlit部分 -------------------
st.title("价格模型by dayu")

# 用户输入参数
stock_code = st.text_input("请输入股票代码(形如'000001.SZ'):", value="000001.SZ")
start_date = st.date_input("选择起始日期:", value=datetime.date(2022,8,25))
future_days = st.number_input("预测未来天数:", min_value=1, value=5)
step = st.number_input("输入时间窗口长度step:", min_value=1, value=5)
epoch = st.number_input("训练轮数epoch:", min_value=1, value=10)
predict_value = st.selectbox("预测值选择:", ["close", "open", "high", "low"])
model_type = st.selectbox("选择模型类型:", ["Net", "NetWithAttention"])

# 按钮提交
if st.button("开始预测"):
    with st.spinner("模型训练中，请稍候..."):
        try:
            preds, actuals, future_preds, fig = stock_prediction(name=stock_code,
                                                                start_time=start_date.strftime("%Y%m%d"),
                                                                future_days_len=future_days,
                                                                step=step,
                                                                epoch=epoch,
                                                                predict_value=predict_value,
                                                                model_type=model_type)
            st.success("预测完成！")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"出现错误: {e}")
