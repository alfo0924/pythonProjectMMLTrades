

import yfinance as yf
import pandas as pd
import numpy as np
import webbrowser
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import alpaca_trade_api as tradeapi

# Alpaca API設置
API_KEY = 'PKMDUVCIDRSCG1EJMYJM'
API_SECRET = '2tHAOcRI3tjXw95cPbheq5sHfEMkfAgKE9s1uZ0Z'
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, api_version='v2')

# 下載黃金歷史數據
data = yf.download('BTC-USD', start='2015-01-01', end='2024-06-01')

# 計算移動平均線 (SMA) 作為趨勢指標
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_60'] = data['Close'].rolling(window=60).mean()
data['SMA_120'] = data['Close'].rolling(window=120).mean()

# 初始化持倉
data['Position'] = 0

# 將前一天的價格加入作為特徵
data['Previous_Close'] = data['Close'].shift(1)

# 訓練Random Forest模型
X = data[['Close', 'SMA_5', 'SMA_20', 'SMA_60', 'SMA_120', 'Previous_Close']].dropna()
y = np.where(data['Close'].shift(-1).reindex(X.index) > X['Close'], 1, -1)
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X, y)

# Define the symbol you're interested in trading
symbol = 'BTCUSD'

# Get account information
account = api.get_account()
buying_power = float(account.buying_power)

# Get the current price of the asset
asset_price = data['Close'].iloc[-1]

# Calculate cumulative return based on trading positions
data['Returns'] = data['Close'].pct_change().shift(-1) * data['Position']
cumulative_return = (data['Returns'] + 1).cumprod() - 1
final_cumulative_return = cumulative_return.iloc[-1]


# Store buy and sell signals
data['Buy_Signal'] = np.where(data['Position'] == 1, data['Close'], np.nan)
data['Sell_Signal'] = np.where(data['Position'] == -1, data['Close'], np.nan)

# Get the indices of buy and sell signals
buy_signals = data[data['Position'] == 1].index.tolist()[:3]
sell_signals = data[data['Position'] == -1].index.tolist()[:3]


# Check buying power before placing buy order
if buying_power >= asset_price:
    # Predict trading signals
    pred = model.predict(X)
    data['Position'] = pd.Series(pred, index=X.index)

    # Initialize variables to track order status
    last_position = 0

    # Iterate through each trading signal
    for index, position in data['Position'].iteritems():
        # Check if the position has changed
        if position != last_position:
            if position == 1:  # Buy signal
                api.submit_order(
                    symbol=symbol,
                    qty=1,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Bought 1 unit of {symbol} at market price.")
            elif position == -1:  # Sell signal
                api.submit_order(
                    symbol=symbol,
                    qty=1,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Sold 1 unit of {symbol} at market price.")
            last_position = position



# Generate the Plotly figure
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_5'], mode='lines', name='SMA_5'))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA_20'))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_60'], mode='lines', name='SMA_60'))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_120'], mode='lines', name='SMA_120'))

# Update layout
fig.update_layout(title='Trading Signals', xaxis_title='Date', yaxis_title='Price')

# Generate HTML content with the Plotly chart
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
        <li>買進點位: {buy_signals[:3]}</li>
        <li>賣出點位: {sell_signals[:3]}</li>
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
with open("trading_RF_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器並導航到Alpaca模擬交易結果的官網
webbrowser.open("https://app.alpaca.markets/paper/dashboard/overview")
