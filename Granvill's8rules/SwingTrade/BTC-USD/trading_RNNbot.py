import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go

# 下載比特幣歷史數據
data = yf.download('BTC-USD', start='2015-01-01', end='2025-06-03')

# 計算移動平均線 (SMA) 作為趨勢指標
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# 移除其他移動平均線
data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_200']]

# 將前一天的價格加入作為特徵
data['Previous_Close'] = data['Close'].shift(1)

# 每週最後一天的數據來生成交易信號
weekly_data = data.resample('W').last().dropna()

# 買進訊號條件
buy_signal_1 = (weekly_data['Close'] > weekly_data['SMA_200']) & (weekly_data['Close'] > weekly_data['Previous_Close'])
buy_signal_2 = (weekly_data['Close'] < weekly_data['SMA_200']) & (weekly_data['Close'] > weekly_data['SMA_200'].shift(1))
buy_signal_3 = (weekly_data['Close'] > weekly_data['SMA_200']) & (weekly_data['Close'].shift(1) < weekly_data['SMA_200'])
buy_signal_4 = (weekly_data['Close'] < weekly_data['SMA_200']) & (weekly_data['Close'].shift(1) < weekly_data['SMA_200'])

weekly_data['Buy_Signal'] = (buy_signal_1 | buy_signal_2 | buy_signal_3 | buy_signal_4).astype(int)

# 賣出訊號條件
sell_signal_1 = (weekly_data['Close'] < weekly_data['SMA_200']) & (weekly_data['Close'] < weekly_data['Previous_Close'])
sell_signal_2 = (weekly_data['Close'] > weekly_data['SMA_200']) & (weekly_data['Close'] < weekly_data['SMA_200'].shift(1))
sell_signal_3 = (weekly_data['Close'] < weekly_data['SMA_200']) & (weekly_data['Close'] < weekly_data['SMA_200'].shift(1))
sell_signal_4 = (weekly_data['Close'] > weekly_data['SMA_200']) & (weekly_data['Close'] < weekly_data['SMA_200'].shift(1))

weekly_data['Sell_Signal'] = (sell_signal_1 | sell_signal_2 | sell_signal_3 | sell_signal_4).astype(int)

# 初始化持倉狀態
weekly_data['Position'] = 0

# 生成交易信號
for i in range(len(weekly_data)):
    if weekly_data['Buy_Signal'].iloc[i] == 1:
        weekly_data['Position'].iloc[i] = 1
    elif weekly_data['Sell_Signal'].iloc[i] == 1:
        weekly_data['Position'].iloc[i] = 0

# 將每週的交易信號擴展到每日數據
data['Position'] = weekly_data['Position'].reindex(data.index, method='ffill')

# 計算策略收益
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
                      go.Scatter(x=buy_signals, y=data.loc[buy_signals]['Low'], mode='markers', name='買入訊號',
                                 marker=dict(color='green', size=10, symbol='triangle-up')),
                      go.Scatter(x=sell_signals, y=data.loc[sell_signals]['High'], mode='markers', name='賣出訊號',
                                 marker=dict(color='red', size=10, symbol='triangle-down'))])

fig.update_layout(title='比特幣/美元 BTC-USD 交易策略 (遞歸神經網絡 RNN + 格蘭碧8大法則 均線:200均 交易頻率:一周一次)', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
with open("trading_Granvills8rules_BTCUSD_RNN_result_weekly.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_Granvills8rules_BTCUSD_RNN_result_weekly.html")
