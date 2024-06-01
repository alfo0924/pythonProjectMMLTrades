import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# 下载比特币历史数据
data = yf.download('BTC-USD', start='2015-01-01', end='2024-06-01')

# 计算移动平均线 (SMA) 作为趋势指标
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_60'] = data['Close'].rolling(window=60).mean()
data['SMA_120'] = data['Close'].rolling(window=120).mean()

# 初始化持仓
data['Position'] = 0

# 将前一天的价格加入作为特征
data['Previous_Close'] = data['Close'].shift(1)

# 准备训练数据
X = data[['Close', 'SMA_5', 'SMA_20', 'SMA_60', 'SMA_120', 'Previous_Close']].dropna()
y = np.where(data['Close'].shift(-1).reindex(X.index) > X['Close'], 1, 0)  # 修改标签，不再使用-1，1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 由于循环神经网络需要 3D 的输入 (samples, time steps, features)
# 我们需要重塑数据
X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

# 建立循环神经网络模型
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# 在测试数据上进行预测
pred_proba = model.predict(X_test)
pred = (pred_proba > 0.5).astype(int).reshape(-1)

# 将预测转换为 DataFrame
pred_df = pd.DataFrame(pred_proba, index=X.index[-len(pred_proba):], columns=['Position'])

# 将预测值分配到 'Position' 列
data['Position'] = 0  # 重新初始化
data.loc[pred_df.index, 'Position'] = pred_df['Position']

data['Strategy_Return'] = data['Position'].shift(1) * data['Close'].pct_change()

# 计算累积收益
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易点位
buy_signals = data[data['Position'] == 1].index
sell_signals = data[data['Position'] == 0].index

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

fig.update_layout(title='Gold Trading Strategy (RNN)', xaxis_title='Date', yaxis_title='Price', showlegend=True)

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
with open("trading_RNN_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打开浏览器
webbrowser.open("trading_RNN_result.html")
