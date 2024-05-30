import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import webbrowser

# 下載黃金歷史數據
data = yf.download('GC=F', start='2024-01-01', end='2024-05-01')
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

# 生成交易點位
buy_signals = data[data['Signal'] == 1].index
sell_signals = data[data['Signal'] == 0].index

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
    <p>{cumulative_return.to_list()}</p>
    <h2>買進點位</h2>
    <ul>
        {''.join([f'<li>{date}</li>' for date in buy_signals])}
    </ul>
    <h2>賣出點位</h2>
    <ul>
        {''.join([f'<li>{date}</li>' for date in sell_signals])}
    </ul>
</body>
</html>
"""

# 寫入HTML文件
with open("trading_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_result.html")
