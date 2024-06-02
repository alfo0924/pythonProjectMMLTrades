import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go

# 下載比特幣歷史數據
data = yf.download('BTC-USD', start='2015-01-01', end='2024-06-03')

# 計算移動平均線
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_60'] = data['Close'].rolling(window=60).mean()
data['SMA_120'] = data['Close'].rolling(window=120).mean()

# 初始化持倉
data['Position'] = 0

# 确定交易信号
data['Previous_Close'] = data['Close'].shift(1)

# 生成交易信号
data['Buy_Signal'] = np.where(
    (data['Close'] > data['SMA_120']) & (data['Close'] > data['Previous_Close'] * 1.005),
    1, 0
)
data['Sell_Signal'] = np.where(
    (data['Close'] < data['SMA_120']) & (data['Close'] < data['SMA_5']) & (data['Close'] < data['SMA_20']),
    1, 0
)

# 模擬交易
for i in range(1, len(data)):
    if data['Buy_Signal'].iloc[i] == 1:
        data['Position'].iloc[i] = 1
    elif data['Sell_Signal'].iloc[i] == 1:
        data['Position'].iloc[i] = 0
    else:
        data['Position'].iloc[i] = data['Position'].iloc[i-1]

# 計算策略收益
data['Strategy_Return'] = data['Position'].shift(1) * data['Close'].pct_change()

# 累積收益計算
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = data[data['Buy_Signal'] == 1].index
sell_signals = data[data['Sell_Signal'] == 1].index

# 生成交互式圖表
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

fig.update_layout(title='BTC-USD Trading Strategy (CNN)', xaxis_title='Date', yaxis_title='Price', showlegend=True)

# 生成HTML內容
html_content = f"""
<!DOCTYPE html>
<html lang="en">
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
with open("trading_CNNresult.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_CNNresult.html")
