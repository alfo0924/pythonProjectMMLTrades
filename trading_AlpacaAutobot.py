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

# 預測交易信號
pred = model.predict(X)
data['Position'] = pd.Series(pred, index=X.index)

# Alpaca下單函數
def submit_order(position, symbol):
    try:
        if position == 1:
            account = api.get_account()
            buying_power = float(account.buying_power)
            asset_price = float(data['Close'].iloc[-1])  # Assuming you're using the latest closing price for the asset
            if buying_power >= asset_price:
                api.submit_order(
                    symbol=symbol,
                    qty=1,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
            else:
                print("Insufficient buying power to place buy order.")
        elif position == -1:
            positions = api.list_positions()
            existing_positions = {pos.symbol: pos for pos in positions}
            if symbol in existing_positions:
                asset_qty = float(existing_positions[symbol].qty)
                if asset_qty >= 1:
                    api.submit_order(
                        symbol=symbol,
                        qty=1,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                else:
                    print("Insufficient asset quantity to place sell order.")
            else:
                print(f"No existing position for {symbol}.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# 執行交易
for i in range(len(data)):
    submit_order(data['Position'].iloc[i], 'BTCUSD')

# 計算策略收益率
data['Strategy_Return'] = data['Position'].shift(1) * data['Close'].pct_change()

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
                      go.Scatter(x=buy_signals, y=data.loc[buy_signals]['Low'], mode='markers', name='Buy Signal',
                                 marker=dict(color='green', size=10, symbol='triangle-up')),
                      go.Scatter(x=sell_signals, y=data.loc[sell_signals]['High'], mode='markers', name='Sell Signal',
                                 marker=dict(color='red', size=10, symbol='triangle-down'))])

fig.update_layout(title='BTC-USD Trading Strategy (Random Forest)', xaxis_title='Date', yaxis_title='Price', showlegend=True)

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
with open("trading_RF_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器並導航到Alpaca模擬交易結果的官網
webbrowser.open("https://app.alpaca.markets/paper/dashboard/overview")
