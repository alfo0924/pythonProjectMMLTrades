import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go

# 下載黃金歷史數據
data = yf.download('GC=F', start='2014-01-01', end='2024-05-30', interval='1d')

# 計算策略收益率
data['Strategy_Return'] = data['Close'].pct_change()

# 處理缺失值
data.dropna(inplace=True)

# 計算RSI
def compute_RSI(data, time_window):
    diff = data.diff(1)
    up_chg = diff.where(diff > 0, 0)
    down_chg = -diff.where(diff < 0, 0)

    up_chg_avg = up_chg.rolling(time_window).mean()
    down_chg_avg = down_chg.rolling(time_window).mean()

    rs = up_chg_avg / down_chg_avg
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_RSI(data['Close'], 14)

# 設定壓力水平（假設為前期高點作為壓力水平）
data['Resistance'] = data['Close'].rolling(window=50).max()

# 生成交互式圖表
fig = go.Figure(data=[
    go.Candlestick(x=data.index,
                   open=data['Open'],
                   high=data['High'],
                   low=data['Low'],
                   close=data['Close'],
                   name='Candlestick')
])
# 生成交易信号
data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 计算50日简单移动平均线
data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 计算200日简单移动平均线

# 当50日SMA上穿200日SMA时产生买入信号，当50日SMA下穿200日SMA时产生卖出信号
data['Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, -1)


# 加入交易信號
buy_signals = data[data['Signal'] == 1].index
sell_signals = data[data['Signal'] == -1].index

fig.add_trace(go.Scatter(x=buy_signals, y=data.loc[buy_signals]['Low'], mode='markers', name='Buy Signal',
                         marker=dict(color='green', size=10, symbol='triangle-up')))
fig.add_trace(go.Scatter(x=sell_signals, y=data.loc[sell_signals]['High'], mode='markers', name='Sell Signal',
                         marker=dict(color='red', size=10, symbol='triangle-down')))

fig.update_layout(title='Gold Trading Strategy', xaxis_title='Date', yaxis_title='Price', showlegend=True)

# Calculate cumulative return
final_cumulative_return = (1 + data['Strategy_Return']).cumprod()[-1] - 1

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
with open("trading_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_result.html")
