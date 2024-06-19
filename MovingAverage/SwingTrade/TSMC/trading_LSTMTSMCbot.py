import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 下載比特幣歷史數據
data = yf.download('2330.TW', start='2015-01-01', end='2025-06-03')

# 計算移動平均線 (SMA) 作為趨勢指標
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_60'] = data['Close'].rolling(window=60).mean()
data['SMA_120'] = data['Close'].rolling(window=120).mean()

# 每周一次計算移動平均線
data_weekly = data.resample('W-FRI').last()
data_weekly['SMA_5'] = data_weekly['Close'].rolling(window=5).mean()
data_weekly['SMA_20'] = data_weekly['Close'].rolling(window=20).mean()
data_weekly['SMA_60'] = data_weekly['Close'].rolling(window=60).mean()
data_weekly['SMA_120'] = data_weekly['Close'].rolling(window=120).mean()

# 初始化持倉
data_weekly['Position'] = 0

# 將前一周的收盤價加入作為特徵
data_weekly['Previous_Close'] = data_weekly['Close'].shift(1)

# 確定交易信號
data_weekly['Buy_Signal'] = np.where(
    (data_weekly['Close'] > data_weekly['SMA_120']) & (data_weekly['Close'] > data_weekly['Previous_Close'] * 1.005),
    1, 0
)
data_weekly['Sell_Signal'] = np.where(
    (data_weekly['Close'] < data_weekly['SMA_120']) & (data_weekly['Close'] < data_weekly['SMA_5']) & (data_weekly['Close'] < data_weekly['SMA_20']),
    1, 0
)

# 模擬交易
for i in range(1, len(data_weekly)):
    if data_weekly['Buy_Signal'].iloc[i] == 1:
        data_weekly['Position'].iloc[i] = 1
    elif data_weekly['Sell_Signal'].iloc[i] == 1:
        data_weekly['Position'].iloc[i] = 0
    else:
        data_weekly['Position'].iloc[i] = data_weekly['Position'].iloc[i-1]

# 計算策略收益率
data_weekly['Strategy_Return'] = data_weekly['Position'].shift(1) * data_weekly['Close'].pct_change()

# 准備特徵和目標變量
X = data_weekly[['SMA_5', 'SMA_20', 'SMA_60', 'SMA_120']].values
y = np.where(data_weekly['Close'].shift(-1) > data_weekly['Close'], 1, 0)

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 數據標準化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 重塑數據以符合LSTM輸入要求
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# 構建LSTM模型
model = Sequential([
    LSTM(50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(1, activation='sigmoid')
])

# 編譯模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=0)

# 使用模型進行預測
predictions = model.predict(X_test_reshaped)
predictions_binary = (predictions > 0.5).astype(int)

# 將預測結果添加到數據框中
data_weekly['Predicted_Signal'] = np.nan
data_weekly.iloc[-len(predictions_binary):, -1] = predictions_binary.flatten()

# 計算策略收益率
data_weekly['Strategy_Return'] = data_weekly['Predicted_Signal'] * data_weekly['Close'].pct_change()

# 累積收益計算
cumulative_return = (data_weekly['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = data_weekly[data_weekly['Predicted_Signal'] == 1].index
sell_signals = data_weekly[data_weekly['Predicted_Signal'] == 0].index

# 生成互動式圖表
fig = go.Figure(data=[go.Candlestick(x=data_weekly.index,
                                     open=data_weekly['Open'],
                                     high=data_weekly['High'],
                                     low=data_weekly['Low'],
                                     close=data_weekly['Close'],
                                     name='Candlestick'),
                      go.Scatter(x=buy_signals, y=data_weekly.loc[buy_signals]['Low'], mode='markers', name='買入信號',
                                 marker=dict(color='green', size=10, symbol='triangle-up')),
                      go.Scatter(x=sell_signals, y=data_weekly.loc[sell_signals]['High'], mode='markers', name='賣出信號',
                                 marker=dict(color='red', size=10, symbol='triangle-down'))])

fig.update_layout(title='台積電 2330 交易策略 (長短期記憶網路 LSTM  + 每周一次波段移動平均線策略)', xaxis_title='日期', yaxis_title='價格', showlegend=True)

# 生成HTML內容
html_content = f"""
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>交易結果</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>交易結果</h1>
    <h2>累積收益</h2>
    <p>{final_cumulative_return:.2f}</p>
    <h2>交易點位</h2>
    <ul>
        <li>買入點位: {buy_signals[:3].to_list()}</li>
        <li>賣出點位: {sell_signals[:3].to_list()}</li>
    </ul>
    <h2>交易圖表</h2>
    <div id="plotly-chart"></div>
    <script>
        var figure = {fig.to_json()};
        Plotly.newPlot('plotly-chart', figure.data, figure.layout);
    </script>
</body>
</html>
"""

# 寫入HTML文件
with open("trading_LSTM2330result_weekly.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_LSTM2330result_weekly.html")
