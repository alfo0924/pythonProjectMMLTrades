import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go

# 下載黃金歷史數據
data = yf.download('GC=F', start='2019-01-01', end='2024-05-30', interval='1h')

# 將數據重新採樣為每小時頻率
hourly_data = data.resample('H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

# 計算移動平均線 (SMA) 作為趨勢指標
hourly_data['SMA'] = hourly_data['Close'].rolling(window=20).mean()

# 計算RSI
def compute_RSI(data, time_window):
    diff = data.diff(1).dropna()
    up_chg = 0 * diff
    down_chg = 0 * diff

    up_chg[diff > 0] = diff[diff > 0]
    down_chg[diff < 0] = -diff[diff < 0]

    up_chg_avg = up_chg.rolling(time_window, min_periods=1).mean()
    down_chg_avg = down_chg.rolling(time_window, min_periods=1).mean()

    rs = up_chg_avg / down_chg_avg
    rsi = 100 - 100 / (1 + rs)
    return rsi

hourly_data['RSI'] = compute_RSI(hourly_data['Close'], 14)

# 設定壓力水平（假設為前期高點作為壓力水平）
hourly_data['Resistance'] = hourly_data['Close'].rolling(window=50).max()

# 設定起始資金為1000美金
initial_capital = 1000

# 生成交易信號和策略收益率
hourly_data['Buy_Signal'] = ((hourly_data['Close'] > hourly_data['SMA']) &
                             (hourly_data['RSI'] < 70) &
                             (hourly_data['Close'] < hourly_data['Resistance'])).astype(int)

hourly_data['Sell_Signal'] = ((hourly_data['Close'] < hourly_data['SMA']) &
                              (hourly_data['RSI'] > 30) &
                              (hourly_data['Close'] > hourly_data['Resistance'])).astype(int)

hourly_data['Signal'] = hourly_data['Buy_Signal'] - hourly_data['Sell_Signal']
hourly_data['Strategy_Return'] = hourly_data['Signal'].shift(1) * hourly_data['Close'].pct_change()

# 處理缺失值
hourly_data.dropna(inplace=True)

# 計算累積收益
hourly_data['Cumulative_Return'] = (hourly_data['Strategy_Return'] + 1).cumprod()

# 計算最後的盈餘
final_profit = initial_capital * hourly_data['Cumulative_Return'].iloc[-1]

# 生成交易點位
buy_signals = hourly_data[hourly_data['Signal'] == 1].index
sell_signals = hourly_data[hourly_data['Signal'] == -1].index

# 生成交互式圖表
fig = go.Figure(data=[go.Candlestick(x=hourly_data.index,
                                     open=hourly_data['Open'],
                                     high=hourly_data['High'],
                                     low=hourly_data['Low'],
                                     close=hourly_data['Close'],
                                     name='Candlestick'),
                      go.Scatter(x=buy_signals, y=hourly_data.loc[buy_signals]['Low'], mode='markers', name='Buy Signal',
                                 marker=dict(color='green', size=10, symbol='triangle-up')),
                      go.Scatter(x=sell_signals, y=hourly_data.loc[sell_signals]['High'], mode='markers', name='Sell Signal',
                                 marker=dict(color='red', size=10, symbol='triangle-down'))])

fig.update_layout(title='Gold Trading Strategy', xaxis_title='Date', yaxis_title='Price', showlegend=True)

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
    <h2>最後的盈餘</h2>
    <p>{final_profit:.2f} 美金</p>
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
