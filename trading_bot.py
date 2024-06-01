import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go

# 下載黃金歷史數據
data = yf.download('GC=F', start='2019-01-01', end='2024-05-30')

# 計算移動平均線 (SMA) 作為趨勢指標
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_60'] = data['Close'].rolling(window=60).mean()
data['SMA_120'] = data['Close'].rolling(window=120).mean()

# 計算KD指標
def compute_KD(data, window=9):
    high = data['High']
    low = data['Low']
    close = data['Close']

    low_min = low.rolling(window=window, min_periods=1).min()
    high_max = high.rolling(window=window, min_periods=1).max()

    rsv = (close - low_min) / (high_max - low_min) * 100

    k_series = rsv.ewm(com=2).mean()
    d_series = k_series.ewm(com=2).mean()

    return k_series, d_series

data['K'], data['D'] = compute_KD(data)

# 計算MACD指標
def compute_MACD(data, short_window=12, long_window=26):
    short_ema = data['Close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, min_periods=1, adjust=False).mean()

    macd = short_ema - long_ema
    signal_line = macd.ewm(span=9, min_periods=1, adjust=False).mean()

    return macd, signal_line

data['MACD'], data['Signal_Line'] = compute_MACD(data)

# 計算交易量
data['Volume_MA'] = data['Volume'].rolling(window=20).mean()

# 生成交易信號
data['Buy_Signal'] = ((data['Close'] < data['SMA_120']) &
                      (data['K'] < 20) &
                      (data['MACD'] < data['Signal_Line']) &
                      (data['Volume'] > data['Volume_MA'])).astype(int)

data['Sell_Signal'] = ((data['K'] > data['D']) &
                       (data['MACD'] > data['Signal_Line']) &
                       ((data['Close'] < data['SMA_20']) | (data['Close'] < data['SMA_60']))).astype(int)

# 初始化持倉
data['Position'] = 0

# 設定停利停損
take_profit_pct = 0.015  # 1.5%
stop_loss_pct = 0.01  # 1%

for i in range(1, len(data)):
    if data['Buy_Signal'].iloc[i] == 1 and data['Position'].iloc[i-1] == 0:
        data.at[data.index[i], 'Position'] = 1
    elif data['Sell_Signal'].iloc[i] == 1 and data['Position'].iloc[i-1] == 1:
        data.at[data.index[i], 'Position'] = 0
    else:
        data.at[data.index[i], 'Position'] = data['Position'].iloc[i-1]

# 計算策略收益率
data['Strategy_Return'] = data['Position'] * data['Close'].pct_change()

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
