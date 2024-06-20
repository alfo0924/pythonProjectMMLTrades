import yfinance as yf
import pandas as pd
import webbrowser
import plotly.graph_objects as go

# 下載比特幣歷史數據
data = yf.download('GOLD', start='2015-01-01', end='2025-06-03')

# 將數據按每週重採樣，選擇每週最後一天的價格作為代表
weekly_data = data.resample('W').last()

# 生成交互式圖表
fig = go.Figure(data=[go.Candlestick(x=weekly_data.index,
                                     open=weekly_data['Open'],
                                     high=weekly_data['High'],
                                     low=weekly_data['Low'],
                                     close=weekly_data['Close'],
                                     name='Candlestick')])

fig.update_layout(title='黃金 GOLD 交易策略(無任何自定義交易策略框架 交易頻率: 每周交易一次) ', xaxis_title='日期', yaxis_title='價格', showlegend=True)

# 生成HTML內容
html_content = f"""
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>黃金價格走勢</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>黃金價格走勢</h1>
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
with open("trading_GOLD_NoTradingStrategy_weekly_result.html", "w", encoding="utf-8") as file:
    file.write(html_content)

# 打開瀏覽器
webbrowser.open("trading_GOLD_NoTradingStrategy_weekly_result.html")
