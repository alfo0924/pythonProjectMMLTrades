import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 下載歷史數據
data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
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
print(f'Accuracy: {accuracy_score(y_test, predictions):.2f}')

# 生成交易信號
data['Signal'] = model.predict(X)
data['Strategy_Return'] = data['Signal'].shift(1) * data['Return']
data = data.dropna()

# 計算累積收益
cumulative_return = (data['Strategy_Return'] + 1).cumprod()
print(cumulative_return)
