import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 下載比特幣歷史數據
data = yf.download('BTC-USD', start='2015-01-01', end='2025-06-03')

# 移除NaN值
data.dropna(inplace=True)

# 準備特徵和目標變量
X = data[['Close']].values
y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 構建DNN模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 編譯模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=0)

# 使用模型進行預測
predictions = model.predict(X_test_scaled)
predictions_binary = (predictions > 0.5).astype(int)

# 將預測結果添加到數據框中
data['Predicted_Signal'] = np.nan
data.iloc[-len(predictions_binary):, -1] = predictions_binary.flatten()

# 計算策略收益率
data['Strategy_Return'] = data['Predicted_Signal'] * data['Close'].pct_change()

# 累積收益計算
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = data[data['Predicted_Signal'] == 1].index
sell_signals = data[data['Predicted_Signal'] == 0].index

# 生成互動式圖表
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Candlestick'),
                      go.Scatter(x=buy_signals, y=data.loc[buy_signals]['Low'], mode='markers', name='買入信號',
                                 marker=dict(color='green', size=10, symbol='triangle-up')),
                      go.Scatter(x=sell_signals, y=data.loc[sell_signals]['High'], mode='markers', name='賣出信號',
                                 marker=dict(color='red', size=10, symbol='triangle-down'))])

fig.update_layout(title='BTC-USD 交易策略 (深度神經網絡 DNN 自主學習 無任何自定義交易策略框架  交易頻率:一天多次)', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
with open("trading_DNN_autonomous_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_DNN_autonomous_result.html")
