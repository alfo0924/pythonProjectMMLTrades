import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# 下載黃金歷史數據
data = yf.download('GC=F', start='2019-01-01', end='2024-05-30')

# 將數據重新採樣為每週頻率
weekly_data = data.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

# 計算移動平均線 (SMA) 作為趨勢指標
weekly_data['SMA'] = weekly_data['Close'].rolling(window=20).mean()

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

weekly_data['RSI'] = compute_RSI(weekly_data['Close'], 14)

# 設定壓力水平（假設為前期高點作為壓力水平）
weekly_data['Resistance'] = weekly_data['Close'].rolling(window=50).max()

# 生成特徵：價格變化率
weekly_data['Price_Change'] = weekly_data['Close'].pct_change()

# 生成目標變量：未來一週的價格變化率
weekly_data['Target'] = weekly_data['Price_Change'].shift(-1)

# 填充缺失值
weekly_data.dropna(inplace=True)

# 分割特徵和目標變量
X = weekly_data[['SMA', 'RSI', 'Resistance', 'Price_Change']].values
y = weekly_data['Target'].values

# 對特徵進行標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分訓練集和測試集
split = int(0.8 * len(X))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# 創建並訓練決策樹模型
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# 生成預測
predictions = tree_model.predict(X_test)

# 將預測轉換為交易信號和策略收益率
weekly_data['Predicted_Price_Change'] = np.nan
weekly_data.iloc[split:, -1] = predictions
weekly_data['Buy_Signal'] = np.where(weekly_data['Predicted_Price_Change'] > 0, 1, 0)
weekly_data['Sell_Signal'] = np.where(weekly_data['Predicted_Price_Change'] < 0, 1, 0)
weekly_data['Signal'] = weekly_data['Buy_Signal'] - weekly_data['Sell_Signal']
weekly_data['Strategy_Return'] = weekly_data['Signal'].shift(1) * weekly_data['Close'].pct_change()

# 計算累積收益
cumulative_return = (weekly_data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = weekly_data[weekly_data['Signal'] == 1].index
sell_signals = weekly_data[weekly_data['Signal'] == -1].index

# 生成交互式圖表
fig = go.Figure(data=[go.Candlestick(x=weekly_data.index,
                                     open=weekly_data['Open'],
                                     high=weekly_data['High'],
                                     low=weekly_data['Low'],
                                     close=weekly_data['Close'],
                                     name='Candlestick'),
                      go.Scatter(x=buy_signals, y=weekly_data.loc[buy_signals]['Low'], mode='markers', name='Buy Signal',
                                 marker=dict(color='green', size=10, symbol='triangle-up')),
                      go.Scatter(x=sell_signals, y=weekly_data.loc[sell_signals]['High'], mode='markers', name='Sell Signal',
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
