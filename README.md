# Stock Price Forecasting App

A basic web application for interactive stock price forecasting using [Prophet](https://facebook.github.io/prophet/) and [yfinance](https://github.com/ranaroussi/yfinance).  
The app allows users to fetch historical stock data, visualize it, and generate future forecasts for any US stock ticker.

---

## Features

- **Fetches historical stock data** using yfinance for any US ticker.
- **Displays company summary** (name, sector, industry, business summary).
- **Interactive sidebar** for user input:
  - Stock ticker
  - Date range
  - Price column (Open, High, Low, Close)
  - Forecast period (days)
- **Visualizes historical prices** with line charts.
- **Forecasts future prices** using Prophet.
- **Displays forecasted price chart and table** (last 10 days).
- **Shows model performance metrics** (MAE, RMSE) on historical data.

## Requirements
- See `requirements.txt` for package dependencies
