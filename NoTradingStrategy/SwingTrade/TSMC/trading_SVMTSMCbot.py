import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 下載台積電歷史數據
data = yf.download('2330.TW', start='2015-01-01', end='2025-06-03')

# 每週重採樣，選擇每週最後一天的價格作為代表
weekly_data = data.resample('W').last()

# 計算移動平均線 (SMA) 作為趨勢指標
weekly_data['SMA_5'] = weekly_data['Close'].rolling(window=5).mean()
weekly_data['SMA_20'] = weekly_data['Close'].rolling(window=20).mean()
weekly_data['SMA_60'] = weekly_data['Close'].rolling(window=60).mean()
weekly_data['SMA_120'] = weekly_data['Close'].rolling(window=120).mean()

# 將前一週的收盤價加入作為特徵
weekly_data['Previous_Close'] = weekly_data['Close'].shift(1)

# 刪除包含NaN值的列
weekly_data.dropna(inplace=True)

# 準備特徵和目標變量
X = weekly_data[['SMA_5', 'SMA_20', 'SMA_60', 'SMA_120', 'Previous_Close']]
y = np.zeros(len(weekly_data))  # 初始化目標變數為全0，表示持平

# 上漲為1，下跌為-1
y[weekly_data['Close'].shift(-1) > weekly_data['Close']] = 1
y[weekly_data['Close'].shift(-1) < weekly_data['Close']] = -1

# 划分訓練集和測試集
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 初始化支持向量機模型
model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))

# 訓練模型
model.fit(X_train, y_train)

# 在測試數據上進行預測
pred = model.predict(X_test)
weekly_data['Position'] = pd.Series(pred, index=X_test.index)

# 計算策略收益率
weekly_data['Strategy_Return'] = weekly_data['Position'].shift(1) * weekly_data['Close'].pct_change()

# 累積收益計算
cumulative_return = (weekly_data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = weekly_data[weekly_data['Position'] == 1].index
sell_signals = weekly_data[weekly_data['Position'] == -1].index

# 生成交互式圖表
fig = go.Figure(data=[go.Candlestick(x=weekly_data.index,
                                     open=weekly_data['Open'],
                                     high=weekly_data['High'],
                                     low=weekly_data['Low'],
                                     close=weekly_data['Close'],
                                     name='Candlestick'),
                      go.Scatter(x=buy_signals, y=weekly_data.loc[buy_signals]['Low'], mode='markers', name='買入信號',
                                 marker=dict(color='green', size=10, symbol='triangle-up')),
                      go.Scatter(x=sell_signals, y=weekly_data.loc[sell_signals]['High'], mode='markers', name='賣出信號',
                                 marker=dict(color='red', size=10, symbol='triangle-down'))])

fig.update_layout(title='台積電 2330 交易策略 (支持向量機 SVM 自主學習 無任何自定義交易策略框架 交易頻率: 每周交易一次)', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
with open("trading_2330_SVM_autonomous_weekly_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_2330_SVM_autonomous_weekly_result.html")