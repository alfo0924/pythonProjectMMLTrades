import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go

# 下載比特幣歷史數據
data = yf.download('gc=f', start='2015-01-01', end='2025-06-03')

# 計算移動平均線 (SMA) 作為趨勢指標
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_60'] = data['Close'].rolling(window=60).mean()
data['SMA_120'] = data['Close'].rolling(window=120).mean()

# 初始化持倉
data['Position'] = 0

# 設定交易週期為一周一次
weekly_data = data.resample('W').last()  # 取每周最後一天的價格作為代表

for i in range(1, len(weekly_data)):
    if (weekly_data['Close'].iloc[i] > weekly_data['SMA_120'].iloc[i] and
            weekly_data['Close'].iloc[i] > weekly_data['Close'].iloc[i - 1] * 1.005):
        data.at[weekly_data.index[i], 'Position'] = 1
    elif (weekly_data['Close'].iloc[i] < weekly_data['SMA_120'].iloc[i] and
          weekly_data['Close'].iloc[i] < weekly_data['SMA_5'].iloc[i] and
          weekly_data['Close'].iloc[i] < weekly_data['SMA_20'].iloc[i]):
        if data['Position'].iloc[i - 1] == 1 and weekly_data['Close'].iloc[i] > weekly_data['Close'].iloc[i - 1] * 1.005:
            data.at[weekly_data.index[i], 'Position'] = 0
        else:
            data.at[weekly_data.index[i], 'Position'] = -1

# 計算策略收益率
data['Strategy_Return'] = data['Position'].shift(1) * data['Close'].pct_change()

# 移除NaN值
data.dropna(inplace=True)

# 累積收益計算
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = data[data['Position'] == 1].index
sell_signals = data[data['Position'] == -1].index

# 生成交互式圖表
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

fig.update_layout(title='黃金 GC=F 交易策略 (波段移動平均線策略 交易頻率:每周交易一次)', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
with open("trading_GOLDresult_weekly.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_GOLDresult_weekly.html")
