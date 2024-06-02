import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 下载比特币历史数据
data = yf.download('BTC-USD', start='2015-01-01', end='2024-06-03')

# 计算移动平均线 (SMA) 作为趋势指标
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_60'] = data['Close'].rolling(window=60).mean()
data['SMA_120'] = data['Close'].rolling(window=120).mean()

# 初始化持仓
data['Position'] = 0

# 将前一天的价格加入作为特征
data['Previous_Close'] = data['Close'].shift(1)

# 确定交易信号
data['Buy_Signal'] = np.where(
    (data['Close'] > data['SMA_120']) & (data['Close'] > data['Previous_Close'] * 1.005),
    1, 0
)
data['Sell_Signal'] = np.where(
    (data['Close'] < data['SMA_120']) & (data['Close'] < data['SMA_5']) & (data['Close'] < data['SMA_20']),
    1, 0
)

# 模拟交易
for i in range(1, len(data)):
    if data['Buy_Signal'].iloc[i] == 1:
        data['Position'].iloc[i] = 1
    elif data['Sell_Signal'].iloc[i] == 1:
        data['Position'].iloc[i] = 0
    else:
        data['Position'].iloc[i] = data['Position'].iloc[i-1]

# 计算策略收益率
data['Strategy_Return'] = data['Position'].shift(1) * data['Close'].pct_change()

# 准备特征和目标变量
X = data[['SMA_5', 'SMA_20', 'SMA_60', 'SMA_120']].values
y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 重塑数据以符合LSTM输入要求
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# 构建LSTM模型
model = Sequential([
    LSTM(50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))

# 使用模型进行预测
predictions = model.predict(X_test_reshaped)
predictions_binary = (predictions > 0.5).astype(int)

# 将预测结果添加到数据框中
data['Predicted_Signal'] = np.nan
data.iloc[-len(predictions_binary):, -1] = predictions_binary.flatten()

# 计算策略收益率
data['Strategy_Return'] = data['Predicted_Signal'] * data['Close'].pct_change()

# 累积收益计算
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易点位
buy_signals = data[data['Predicted_Signal'] == 1].index
sell_signals = data[data['Predicted_Signal'] == 0].index

# 生成交互式图表
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Candlestick'),
                      go.Scatter(x=buy_signals, y=data.loc[buy_signals]['Low'], mode='markers', name='Buy Signal',
                                 marker=dict(color='green', size=10, symbol='triangle-up')),
                      go.Scatter(x=sell_signals, y=data.loc[sell_signals]['High'], mode='markers', name='Sell Signal',
                                 marker=dict(color='red', size=10, symbol='triangle-down'))])

fig.update_layout(title='BTC-USD Trading Strategy (LSTM)', xaxis_title='Date', yaxis_title='Price', showlegend=True)

# 生成HTML内容
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>交易结果</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>交易结果</h1>
    <h2>累积收益</h2>
    <p>{final_cumulative_return:.2f}</p>
    <h2>交易点位</h2>
    <ul>
        <li>买入点位: {buy_signals[:3].to_list()}</li>
        <li>卖出点位: {sell_signals[:3].to_list()}</li>
    </ul>
    <h2>交易图表</h2>
    <div id="plotly-chart"></div>
    <script>
        var figure = {fig.to_json()};
        Plotly.newPlot('plotly-chart', figure.data, figure.layout);
    </script>
</body>
</html>
"""

# 写入HTML文件
with open("trading_LSTMresult.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打开浏览器
webbrowser.open("trading_LSTMresult.html")
