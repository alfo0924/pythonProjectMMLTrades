import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import webbrowser
import matplotlib.pyplot as plt

# 下載黃金歷史數據
data = yf.download('GC=F', start='2019-01-01', end='2024-05-30')
data = data[['Close']]

# 生成特徵和標籤
data['Return'] = data['Close'].pct_change()
data['Direction'] = (data['Return'] > 0).astype(int)
data = data.dropna()

# 特徵和標籤
X = data[['Close']]
y = data['Direction']

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 評估模型
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# 生成交易信號
data['Signal'] = model.predict(X)
data['Strategy_Return'] = data['Signal'].shift(1) * data['Return']
data = data.dropna()

# 計算累積收益
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
final_cumulative_return = cumulative_return.iloc[-1]

# 生成交易點位
buy_signals = data[data['Signal'] == 1].index
sell_signals = data[data['Signal'] == 0].index

# 生成圖表
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(cumulative_return, label='Strategy Cumulative Return')
plt.scatter(buy_signals, data.loc[buy_signals]['Close'], marker='^', color='g', label='Buy Signal', alpha=1)
plt.scatter(sell_signals, data.loc[sell_signals]['Close'], marker='v', color='r', label='Sell Signal', alpha=1)
plt.title('Gold Trading Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.savefig('trading_strategy.png')

# 生成HTML內容
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>交易結果</title>
</head>
<body>
    <h1>交易結果</h1>
    <p>模型準確度: {accuracy:.2f}</p>
    <h2>累積收益</h2>
    <p>{final_cumulative_return:.2f}</p>
    <h2>交易點位</h2>
    <ul>
        <li>買進點位: {buy_signals[:3].to_list()}</li>
        <li>賣出點位: {sell_signals[:3].to_list()}</li>
    </ul>
    <h2>交易圖表</h2>
    <img src="trading_strategy.png" alt="Trading Strategy">
</body>
</html>
"""

# 寫入HTML文件
with open("trading_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_result.html")
