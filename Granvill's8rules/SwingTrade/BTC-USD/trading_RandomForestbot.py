import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 下載比特幣歷史數據
data = yf.download('BTC-USD', start='2015-01-01', end='2025-06-03')

# 計算移動平均線 (SMA) 作為趨勢指標
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# 初始化持倉
data['Position'] = 0

# 將前一天的價格加入作為特徵
data['Previous_Close'] = data['Close'].shift(1)

# 根據策略生成交易信號
# 買進訊號條件
buy_signal_condition = (
        (data['Close'] > data['SMA_200']) &
        (data['SMA_200'].diff().shift(-1) > 0) |
        ((data['Close'] < data['SMA_200']) & (data['Close'] > data['SMA_200'].shift(1))) |
        ((data['Close'] > data['SMA_200']) & (data['Close'] > data['SMA_200']) & (data['SMA_200'].diff().shift(-1) < 0))
).astype(int)

# 賣出訊號條件
sell_signal_condition = (
        (data['Close'] < data['SMA_200']) &
        (data['SMA_200'].diff().shift(-1) < 0) |
        ((data['Close'] > data['SMA_200']) & (data['Close'] < data['SMA_200'].shift(1))) |
        ((data['Close'] < data['SMA_200']) & (data['Close'] > data['SMA_200']) & (data['SMA_200'].diff().shift(-1) > 0))
).astype(int)

data['Buy_Signal'] = buy_signal_condition
data['Sell_Signal'] = sell_signal_condition

# 將日期設置為索引，方便後續每周操作
data.set_index(pd.to_datetime(data.index), inplace=True)

# 每週最後一天的收盤價作為模型輸入
weekly_data = data.resample('W').last().dropna()

# 初始化随機森林模型
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))

# 特徵和目標變量
X = weekly_data[['Close', 'SMA_200', 'Previous_Close']]
y = np.where(weekly_data['Close'].shift(-1) > weekly_data['Close'], 1, -1)

# 訓練模型
model.fit(X, y)

# 預測每周的交易信號
pred = model.predict(X)
weekly_data['Position'] = pd.Series(pred, index=X.index)

# 計算策略收益率
weekly_data['Strategy_Return'] = weekly_data['Position'].shift(1) * weekly_data['Close'].pct_change()

# 計算累積收益
cumulative_return = (weekly_data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = weekly_data[weekly_data['Position'] == 1].index
sell_signals = weekly_data[weekly_data['Position'] == -1].index

# 生成互動式圖表
fig = go.Figure(data=[go.Candlestick(x=weekly_data.index,
                                     open=weekly_data['Open'],
                                     high=weekly_data['High'],
                                     low=weekly_data['Low'],
                                     close=weekly_data['Close'],
                                     name='Candlestick'),
                      go.Scatter(x=buy_signals, y=weekly_data.loc[buy_signals]['Low'], mode='markers', name='買入訊號',
                                 marker=dict(color='green', size=10, symbol='triangle-up')),
                      go.Scatter(x=sell_signals, y=weekly_data.loc[sell_signals]['High'], mode='markers', name='賣出訊號',
                                 marker=dict(color='red', size=10, symbol='triangle-down'))])

fig.update_layout(title='比特幣/美元 BTC-USD 交易策略 (隨機森林 RF + 格蘭碧8大法則 均線:200均 交易頻率:一周一次)', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
with open("trading_Granvills8rules_BTCUSD_RF_result_weekly.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_Granvills8rules_BTCUSD_RF_result_weekly.html")
