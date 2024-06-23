import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

# 下載比特幣歷史數據
data = yf.download('GOLD', start='2015-01-01', end='2025-06-03')

# 將前一天的價格加入作為特徵
data['Previous_Close'] = data['Close'].shift(1)

# 準備訓練數據
X = data[['Close', 'Previous_Close']].dropna()
y = np.where(data['Close'].shift(-1).reindex(X.index) > X['Close'], 1, 0)

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

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
data['Position'] = 0  # 重新初始化
data.loc[pred_df.index, 'Position'] = pred_df['Position']

# 每周一次交易的設置
data['Trade_Signal'] = 0
weekly_buy_signal = False
for i in range(len(data)):
    if i % 7 == 0:  # 每周一次
        if data['Position'].iloc[i] == 1:
            weekly_buy_signal = True
    if weekly_buy_signal:
        data['Trade_Signal'].iloc[i] = 1
        weekly_buy_signal = False

# 確認Trade_Signal列中沒有NaN值
data['Trade_Signal'].fillna(0, inplace=True)

data['Strategy_Return'] = data['Trade_Signal'].shift(1) * data['Close'].pct_change()

# 計算累積收益
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = data[data['Trade_Signal'] == 1].index
sell_signals = data[data['Trade_Signal'] == 0].index

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

fig.update_layout(title='黃金 GOLD (遞歸神經網絡 RNN 自主學習 無任何自定義交易策略框架  交易頻率:每周一次)', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
with open("trading_GOLD_RNN_autonomous_result_weekly.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_GOLD_RNN_autonomous_result_weekly.html")
