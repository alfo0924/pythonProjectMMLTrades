import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 下載比特幣歷史數據
data = yf.download('2454.TW', start='2015-01-01', end='2025-06-03')

# 計算移動平均線 (SMA) 作為趨勢指標
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_60'] = data['Close'].rolling(window=60).mean()
data['SMA_120'] = data['Close'].rolling(window=120).mean()

# 初始化持倉
data['Position'] = 0

# 將前一周的價格加入作為特徵 (因為週期是一周一次)
data['Previous_Close'] = data['Close'].shift(5)  # 5天 = 1周

# 移動平均線交易策略
data['Buy_Signal'] = ((data['Close'] > data['SMA_120']) &
                      (data['Close'].pct_change(periods=5) > 0.005) &
                      (data['Close'] > data['Previous_Close']))
data['Sell_Signal'] = ((data['Close'] < data['SMA_120']) &
                       (data['Close'] < data['SMA_5']) &
                       (data['Close'] < data['SMA_20']))

# 計算持倉
data.loc[data['Buy_Signal'], 'Position'] = 1
data.loc[data['Sell_Signal'], 'Position'] = -1

# 準備訓練數據
X = data[['SMA_5', 'SMA_20', 'SMA_60', 'SMA_120', 'Previous_Close']].dropna()
y = np.where(data['Close'].shift(-5).reindex(X.index) > X['SMA_120'], 1, -1)  # 預測未來一周的價格變化

# 初始化支持向量機模型
model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))

# 訓練模型
model.fit(X, y)

# 預測交易信號
pred = model.predict(X)
data['Position'] = pd.Series(pred, index=X.index)

# 計算策略收益率
data['Strategy_Return'] = data['Position'].shift(5) * data['Close'].pct_change(periods=5)

# 累積收益計算
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位 (因週期為一周一次，只需每周第一個交易日的點位)
buy_signals = data[data['Position'] == 1].resample('W').first().index
sell_signals = data[data['Position'] == -1].resample('W').first().index

# 確保交易信號在原始數據的索引中存在
buy_signals = buy_signals[buy_signals.isin(data.index)]
sell_signals = sell_signals[sell_signals.isin(data.index)]

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

fig.update_layout(title='聯發科 2454 交易策略 (支持向量機 SVM + 波段移動平均線策略 交易頻率:每周交易一次)', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
with open("trading_SVM2454result_weekly.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_SVM2454result_weekly.html")
