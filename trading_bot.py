import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go

# 下載黃金歷史數據
data = yf.download('GC=F', start='2019-01-01', end='2024-05-30')

# 計算均線
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()

# 計算壓力位（SMA200）
data['Resistance'] = data['SMA200']

# 計算區間
data['Range'] = data['High'] - data['Low']

# 日內波交易策略
# 生成交易信號：只要趨勢往上就做多，前提是在壓力之上
data['Signal'] = np.where((data['Close'] > data['Resistance']) &
                          (data['Close'].pct_change().rolling(window=5).mean() > 0), 1, 0)

# 處理缺失值
data.dropna(inplace=True)

# 計算策略收益率
data['Return'] = data['Close'].pct_change()
data['Strategy_Return'] = data['Signal'].shift(1) * data['Return']
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位，只顯示3筆
buy_signals = data[data['Signal'] == 1].index[:3]
sell_signals = data[data['Signal'] == 0].index[:3]

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

fig.update_layout(title='Gold Trading Strategy (Intraday Wave Trading)', xaxis_title='Date', yaxis_title='Price', showlegend=True)

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
        <li>買進點位: {buy_signals.to_list()}</li>
        <li>賣出點位: {sell_signals.to_list()}</li>
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
with open("trading_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_result.html")
