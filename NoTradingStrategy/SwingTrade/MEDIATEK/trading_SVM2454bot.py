import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# 下載比特幣歷史數據
data = yf.download('2454.TW', start='2015-01-01', end='2025-06-03')

# 將前一天的價格加入作為特徵
data['Previous_Close'] = data['Close'].shift(1)

# 移除NaN值
data.dropna(inplace=True)

# 准備訓練數據
X = data[['Close', 'Previous_Close']].values
y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)

# 划分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 初始化支持向量機模型
model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))

# 訓練模型
model.fit(X_train, y_train)

# 在測試集上進行預測
pred = model.predict(X_test)

# 將預測結果映射回原始數據的索引
pred_series = pd.Series(pred, index=data.index[X_train.shape[0]:])

# 每周一次交易的設置
data['Trade_Signal'] = 0
weekly_buy_signal = False
for i in range(len(data)):
    if data.index[i].dayofweek == 0:  # 每周的第一個交易日
        if pred_series.iloc[i] == 1:
            weekly_buy_signal = True
    if weekly_buy_signal:
        data['Trade_Signal'].iloc[i] = 1
        weekly_buy_signal = False

# 確認Trade_Signal列中沒有NaN值
data['Trade_Signal'].fillna(0, inplace=True)

# 計算策略收益率
data['Strategy_Return'] = data['Trade_Signal'].shift(1) * data['Close'].pct_change()

# 累積收益計算
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = data[data['Trade_Signal'] == 1].index
sell_signals = data[data['Trade_Signal'] == 0].index

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

fig.update_layout(title='聯發科 2454 交易策略 (支持向量機 SVM 自主學習 無任何自定義交易策略框架  交易頻率:每周一次)', xaxis_title='日期', yaxis_title='價格', showlegend=True)

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
with open("trading_2454_SVM_autonomous_result_weekly.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_2454_SVM_autonomous_result_weekly.html")
