import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 下載比特幣歷史數據
data = yf.download('2330.TW', start='2015-01-01', end='2025-06-03')

# 計算移動平均線 (SMA) 作為趨勢指標
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# 初始化持倉
data['Position'] = 0

# 將前一天的價格加入作為特徵
data['Previous_Close'] = data['Close'].shift(1)

# 根據策略生成交易信號
# 買進訊號條件
# 1. 突破
buy_signal_1 = (data['Close'] > data['SMA_200']) & (data['Close'] > data['Previous_Close'])
# 2. 假跌破
buy_signal_2 = (data['Close'] < data['SMA_200']) & (data['Close'] > data['SMA_200'].shift(1))
# 3. 支撐
buy_signal_3 = (data['Close'] > data['SMA_200']) & (data['Close'].shift(1) < data['SMA_200'])
# 4. 抄底
buy_signal_4 = (data['Close'] < data['SMA_200']) & (data['Close'].shift(1) < data['SMA_200'])

data['Buy_Signal'] = (buy_signal_1 | buy_signal_2 | buy_signal_3 | buy_signal_4).astype(int)

# 賣出訊號條件
# 5. 跌破
sell_signal_1 = (data['Close'] < data['SMA_200']) & (data['Close'] < data['Previous_Close'])
# 6. 假突破
sell_signal_2 = (data['Close'] > data['SMA_200']) & (data['Close'] < data['SMA_200'].shift(1))
# 7. 反壓
sell_signal_3 = (data['Close'] < data['SMA_200']) & (data['Close'] < data['SMA_200'].shift(1))
# 8. 反轉
sell_signal_4 = (data['Close'] > data['SMA_200']) & (data['Close'] < data['SMA_200'].shift(1))

data['Sell_Signal'] = (sell_signal_1 | sell_signal_2 | sell_signal_3 | sell_signal_4).astype(int)

# 初始化持倉狀態
position = 0

# 生成交易信號
for i in range(len(data)):
    if data['Buy_Signal'].iloc[i] == 1:
        position = 1
    elif data['Sell_Signal'].iloc[i] == 1:
        position = 0
    data['Position'].iloc[i] = position

# 特徵和目標變量
X = data[['Close', 'SMA_200', 'Previous_Close']].dropna()
y = np.where(data['Close'].shift(-1).reindex(X.index) > X['Close'], 1, -1)

# 初始化随機森林模型
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))

# 訓練模型
model.fit(X, y)

# 預測交易信號
pred = model.predict(X)
data['Position'] = pd.Series(pred, index=X.index)

# 計算策略收益率
data['Strategy_Return'] = data['Position'].shift(1) * data['Close'].pct_change()

# 計算累積收益
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = data[data['Position'] == 1].index
sell_signals = data[data['Position'] == -1].index

# 生成互動式圖表
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Candlestick'),
                      go.Scatter(x=buy_signals, y=data.loc[buy_signals]['Low'], mode='markers', name='買進信號',
                                 marker=dict(color='green', size=10, symbol='triangle-up')),
                      go.Scatter(x=sell_signals, y=data.loc[sell_signals]['High'], mode='markers', name='賣出信號',
                                 marker=dict(color='red', size=10, symbol='triangle-down'))])

fig.update_layout(title='台積電 2330 交易策略 (隨機森林 RF + 格蘭碧8大法則 均線:200均 交易頻率:一天多次 )', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
        <li>買進點位: {buy_signals[:3].to_list()}</li>
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
with open("trading_Granvills8rules_2330_RF_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_Granvills8rules_2330_RF_result.html")
