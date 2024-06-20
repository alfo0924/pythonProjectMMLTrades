import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 下載比特幣歷史數據
data = yf.download('BTC-USD', start='2015-01-01', end='2025-06-03')

# 將數據按每週重採樣，選擇每週最後一天的價格作為代表
weekly_data = data.resample('W').last()

# 計算移動平均線 (SMA) 作為趨勢指標
weekly_data['SMA_5'] = weekly_data['Close'].rolling(window=5).mean()
weekly_data['SMA_20'] = weekly_data['Close'].rolling(window=20).mean()
weekly_data['SMA_60'] = weekly_data['Close'].rolling(window=60).mean()
weekly_data['SMA_120'] = weekly_data['Close'].rolling(window=120).mean()

# 將前一週的收盤價加入作為特徵
weekly_data['Previous_Close'] = weekly_data['Close'].shift(1)

# 刪除包含NaN值的列
weekly_data.dropna(inplace=True)

# 準備特徵和目標變量
X = weekly_data[['Close', 'SMA_5', 'SMA_20', 'SMA_60', 'SMA_120', 'Previous_Close']]
y = np.where(weekly_data['Close'].shift(-1) > weekly_data['Close'], 1, 0)

# 划分訓練集和測試集
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 對特徵進行標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 由於循環神經網絡需要 3D 的輸入 (samples, time steps, features)
# 我們需要重塑數據
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# 建立循環神經網絡模型
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=0)

# 在測試數據上進行預測
pred_proba = model.predict(X_test_reshaped)
pred = (pred_proba > 0.5).astype(int).reshape(-1)

# 將預測轉換為 DataFrame
pred_df = pd.DataFrame(pred_proba, index=X.index[-len(pred_proba):], columns=['Position'])

# 將預測值分配到 'Position' 列
weekly_data['Position'] = 0  # 重新初始化
weekly_data.loc[pred_df.index, 'Position'] = pred_df['Position']

# 計算策略收益率
weekly_data['Strategy_Return'] = weekly_data['Position'].shift(1) * weekly_data['Close'].pct_change()

# 計算累積收益
cumulative_return = (weekly_data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = weekly_data[weekly_data['Position'] == 1].index
sell_signals = weekly_data[weekly_data['Position'] == 0].index

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

fig.update_layout(title='BTC-USD 交易策略 (遞歸神經網絡 RNN 自主學習 無任何自定義交易策略框架 交易頻率: 每周交易一次)', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
with open("trading_RNN_autonomous_weekly_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_RNN_autonomous_weekly_result.html")
