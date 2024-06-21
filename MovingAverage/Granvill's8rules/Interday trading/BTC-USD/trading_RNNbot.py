import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 下載比特幣歷史數據
data = yf.download('BTC-USD', start='2015-01-01', end='2025-06-03')

# 計算移動平均線 (SMA) 作為趨勢指標
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# 初始化持倉
data['Position'] = 0

# 確定買入與賣出訊號
# 買入訊號
buy_signal_1 = (data['Close'] > data['SMA_200']) & (data['Close'].shift(1) <= data['SMA_200'].shift(1))
buy_signal_2 = (data['Close'] < data['SMA_200']) & (data['Close'].shift(1) >= data['SMA_200'].shift(1))
buy_signal_3 = (data['Close'] > data['SMA_200']) & (data['Close'].shift(1) < data['SMA_200'].shift(1))
buy_signal_4 = (data['Close'] < data['SMA_200']) & (data['Close'].shift(1) > data['SMA_200'].shift(1))

# 賣出訊號
sell_signal_1 = (data['Close'] < data['SMA_200']) & (data['Close'].shift(1) >= data['SMA_200'].shift(1))
sell_signal_2 = (data['Close'] > data['SMA_200']) & (data['Close'].shift(1) <= data['SMA_200'].shift(1))
sell_signal_3 = (data['Close'] < data['SMA_200']) & (data['Close'] < data['SMA_200'].shift(1))
sell_signal_4 = (data['Close'] > data['SMA_200']) & (data['Close'] > data['SMA_200'].shift(1))

# 將訊號條件應用到持倉中
data.loc[buy_signal_1 | buy_signal_2 | buy_signal_3 | buy_signal_4, 'Position'] = 1
data.loc[sell_signal_1 | sell_signal_2 | sell_signal_3 | sell_signal_4, 'Position'] = 0

# 準備訓練數據
X = data[['Close', 'SMA_200']].dropna()
y = np.where(data['Close'].shift(-1).reindex(X.index) > X['Close'], 1, 0)  # 修改標籤，不再使用-1，1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 由於循環神經網絡需要 3D 的輸入 (samples, time steps, features)
# 我們需要重塑數據
X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

# 建立循環神經網絡模型
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

# 訓練模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# 在測試數據上進行預測
pred_proba = model.predict(X_test)
pred = (pred_proba > 0.5).astype(int).reshape(-1)

# 將預測轉換為 DataFrame
pred_df = pd.DataFrame(pred_proba, index=X.index[-len(pred_proba):], columns=['Position'])

# 將預測值分配到 'Position' 列
data['Position'] = 0  # 重新初始化
data.loc[pred_df.index, 'Position'] = pred_df['Position']

data['Strategy_Return'] = data['Position'].shift(1) * data['Close'].pct_change()

# 計算累積收益
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = data[data['Position'] == 1].index
sell_signals = data[data['Position'] == 0].index

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

fig.update_layout(title='BTC-USD 交易策略 (遞歸神經網絡 RNN + 格蘭碧8大法則 均線:200均 交易頻率:一天多次 )', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
with open("trading_Granvills8rules_RNN_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_Granvills8rules_RNN_result.html")
