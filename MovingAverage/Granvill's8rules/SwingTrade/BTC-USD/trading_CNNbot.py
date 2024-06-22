import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# 下載比特幣歷史數據
data = yf.download('BTC-USD', start='2015-01-01', end='2025-06-03')

# 將數據轉換為DatetimeIndex
data.index = pd.to_datetime(data.index)

# 將數據按每週重採樣，選擇每週最後一天的價格作為代表
weekly_data = data.resample('W').last()

# 計算移動平均線
weekly_data['SMA_5'] = weekly_data['Close'].rolling(window=5).mean()
weekly_data['SMA_20'] = weekly_data['Close'].rolling(window=20).mean()
weekly_data['SMA_60'] = weekly_data['Close'].rolling(window=60).mean()
weekly_data['SMA_120'] = weekly_data['Close'].rolling(window=120).mean()

# 移除NaN值
weekly_data.dropna(inplace=True)

# 準備特徵和目標變量
X = weekly_data[['SMA_5', 'SMA_20', 'SMA_60', 'SMA_120']].values
y = np.where(weekly_data['Close'].shift(-1) > weekly_data['Close'], 1, 0)

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 構建CNN模型
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 編譯模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 將數據調整為CNN模型的輸入形狀
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# 訓練模型
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=0)

# 使用模型進行預測
predictions = model.predict(X_test_scaled)
predictions_binary = (predictions > 0.5).astype(int)

# 將預測結果添加到數據框中
weekly_data['Predicted_Signal'] = np.nan
weekly_data.iloc[-len(predictions_binary):, -1] = predictions_binary.flatten()

# 定義交易信號函數（修改後避免使用SMA_200）
def generate_signals(data):
    signals = np.zeros(len(data))

    for i in range(1, len(data)):
        # 使用SMA_120替代SMA_200
        if (data['Close'].iloc[i] > data['SMA_120'].iloc[i]) and (data['Close'].iloc[i-1] <= data['SMA_120'].iloc[i-1]):
            signals[i] = 1  # 突破買入訊號
        elif (data['Close'].iloc[i] < data['SMA_120'].iloc[i]) and (data['Close'].iloc[i-1] > data['SMA_120'].iloc[i-1]):
            signals[i] = -1  # 跌破賣出訊號

    return signals

# 生成交易信號
weekly_data['Signal'] = generate_signals(weekly_data)

# 計算策略收益率
weekly_data['Strategy_Return'] = weekly_data['Signal'] * weekly_data['Close'].pct_change()

# 累積收益計算
cumulative_return = (weekly_data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = weekly_data[weekly_data['Signal'] == 1].index
sell_signals = weekly_data[weekly_data['Signal'] == -1].index

# 生成互動式圖表
fig = go.Figure(data=[go.Candlestick(x=weekly_data.index,
                                     open=weekly_data['Open'],
                                     high=weekly_data['High'],
                                     low=weekly_data['Low'],
                                     close=weekly_data['Close'],
                                     name='Candlestick'),
                      go.Scatter(x=buy_signals, y=weekly_data.loc[buy_signals]['Low'], mode='markers', name='買入信號',
                                 marker=dict(color='green', size=10, symbol='triangle-up')),
                      go.Scatter(x=sell_signals, y=weekly_data.loc[sell_signals]['High'], mode='markers', name='賣出信號',
                                 marker=dict(color='red', size=10, symbol='triangle-down'))])

fig.update_layout(title='BTC-USD 交易策略 (卷積神經網絡 CNN + 格蘭碧8大法則 均線:120均 交易頻率:一周一次)', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
with open("trading_Granvills8rules_BTCUSD_CNN_result_weekly.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_Granvills8rules_BTCUSD_CNN_result_weekly.html")
