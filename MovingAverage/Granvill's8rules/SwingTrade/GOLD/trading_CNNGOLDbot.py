import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go

# 下載比特幣的歷史數據
data = yf.download('GOLD', start='2015-01-01', end='2025-06-03')

# 計算移動平均線 (SMA) 作為趨勢指標，只保留200均線
data['SMA_200'] = data['Close'].rolling(window=200).mean()
data.dropna(inplace=True)

# 計算移動平均線的斜率
data['Slope'] = np.gradient(data['SMA_200'])

# 計算當天價格相對於200均線的位置，1表示在均線上方，0表示在均線下方
data['Above_SMA_200'] = np.where(data['Close'] > data['SMA_200'], 1, 0)

# 計算前一天的價格作為特徵
data['Previous_Close'] = data['Close'].shift(1)

# 生成交易信號的DataFrame
signals = pd.DataFrame(index=data.index)

# 買進訊號條件
# 1. 突破
signals['Buy_Signal_1'] = np.where((data['Slope'] >= 0) & (data['Above_SMA_200'].shift(1) == 0) & (data['Above_SMA_200'] == 1), 1, 0)
# 2. 假跌破
signals['Buy_Signal_2'] = np.where((data['Close'] < data['SMA_200']) & (data['Close'].shift(1) > data['SMA_200']) & (data['Slope'] > 0), 1, 0)
# 3. 支撐
signals['Buy_Signal_3'] = np.where((data['Above_SMA_200'].shift(1) == 1) & (data['Above_SMA_200'] == 1) & (data['Close'] < data['SMA_200']), 1, 0)
# 4. 抄底
signals['Buy_Signal_4'] = np.where((data['Close'] < data['SMA_200']) & (data['Close'] < data['Low'].rolling(window=20).min().shift(1)), 1, 0)

# 賣出訊號條件
# 5. 跌破
signals['Sell_Signal_1'] = np.where((data['Slope'] < 0) & (data['Above_SMA_200'].shift(1) == 1) & (data['Above_SMA_200'] == 0), 1, 0)
# 6. 假突破
signals['Sell_Signal_2'] = np.where((data['Close'] > data['SMA_200']) & (data['Close'].shift(1) < data['SMA_200']) & (data['Slope'] < 0), 1, 0)
# 7. 反壓
signals['Sell_Signal_3'] = np.where((data['Above_SMA_200'].shift(1) == 0) & (data['Above_SMA_200'] == 0) & (data['Close'] > data['SMA_200']), 1, 0)
# 8. 反轉
signals['Sell_Signal_4'] = np.where((data['Close'] > data['SMA_200']) & (data['Close'] > data['High'].rolling(window=20).max().shift(1)), 1, 0)

# 將信號合併成一個買入訊號和一個賣出訊號
signals['Buy_Signal'] = signals[['Buy_Signal_1', 'Buy_Signal_2', 'Buy_Signal_3', 'Buy_Signal_4']].any(axis=1)
signals['Sell_Signal'] = signals[['Sell_Signal_1', 'Sell_Signal_2', 'Sell_Signal_3', 'Sell_Signal_4']].any(axis=1)

# 將交易信號加入到原始數據中
data['Position'] = 0  # 初始化持倉為0
data.loc[signals[signals['Buy_Signal']].index, 'Position'] = 1  # 根據買入訊號標記持倉為1
data.loc[signals[signals['Sell_Signal']].index, 'Position'] = 0  # 根據賣出訊號標記持倉為0

# 計算策略收益
data['Strategy_Return'] = data['Position'].shift(1) * data['Close'].pct_change()

# 計算累積收益率
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
                      go.Scatter(x=buy_signals, y=data.loc[buy_signals]['Low'], mode='markers', name='買入信號',
                                 marker=dict(color='green', size=10, symbol='triangle-up')),
                      go.Scatter(x=sell_signals, y=data.loc[sell_signals]['High'], mode='markers', name='賣出信號',
                                 marker=dict(color='red', size=10, symbol='triangle-down'))])

fig.update_layout(title='黃金 GOLD 交易策略 (卷積神經網絡 CNN + 均線:200均 交易頻率:一周一次)', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
with open("trading_Granvills8rules_GOLD_CNN_result_weekly.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器顯示結果
webbrowser.open("trading_Granvills8rules_GOLD_CNN_result_weekly.html")
