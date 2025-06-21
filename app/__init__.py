import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.title('Stock Price Forecasting App')

#st.set_page_config(page_title="Stock Price Forecasting", page_icon=":chart_with_upwards_trend:", layout="wide")

st.sidebar.header('User Input')
ticker = st.sidebar.text_input('Stock Ticker', value='AAPL')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2022-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2023-01-01'))
forecast_period = st.sidebar.number_input('Forecast Days', min_value=1, max_value=365, value=30)
price_column = st.sidebar.selectbox('Price Column to Forecast', ['Open', 'High', 'Low', 'Close'], index=3)


# Fetch stock info for summary
try:
    stock_info = yf.Ticker(ticker).info
    company_name = stock_info.get('longName', 'N/A')
    sector = stock_info.get('sector', 'N/A')
    industry = stock_info.get('industry', 'N/A')
    summary = stock_info.get('longBusinessSummary', 'No summary available.')
    with st.expander("Company Summary"):
        st.write(summary)
except Exception as e:
    st.warning("Could not fetch stock summary information.")


# Fetch data
data = yf.download(ticker, start=start_date, end=end_date)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

if data.empty:
    st.error("No data found for the selected ticker and date range. Please try a different combination.")
else:
    df_raw = data.reset_index()
    date_col = 'Date'

    if price_column not in df_raw.columns:
        st.error(f"Column '{price_column}' not found in data after reset_index. Available columns: {df_raw.columns.tolist()}")
    else:
        # Show historical price chart
        st.subheader(f'Historical {price_column} Prices')
        st.line_chart(df_raw.set_index(date_col)[price_column])

        # Prepare data for Prophet
        df = df_raw[[date_col, price_column]].rename(columns={date_col: 'ds', price_column: 'y'})
        df = df.dropna(subset=['y'])
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['y'])

        if df.empty:
            st.error("No valid data available for forecasting after cleaning. Please try a different ticker, date range, or price column.")
        else:
            try:
                # Fit Prophet model
                model = Prophet()
                model.fit(df)
                future = model.make_future_dataframe(periods=forecast_period)
                forecast = model.predict(future)

                # Show forecast chart
                st.subheader('Forecasted Prices')
                fig1 = model.plot(forecast)
                st.pyplot(fig1)

                # Show forecast table
                st.subheader('Forecast Table (Last 10 Days)')
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

                # Model performance metrics (on historical data)
                # Compare Prophet's in-sample prediction to actuals
                forecast_hist = forecast.set_index('ds').loc[df['ds']]
                mae = mean_absolute_error(df['y'], forecast_hist['yhat'])
                rmse = np.sqrt(mean_squared_error(df['y'], forecast_hist['yhat']))
                st.markdown(f"**Model Performance on Historical Data:**")
                st.markdown(f"- MAE: {mae:.2f}")
                st.markdown(f"- RMSE: {rmse:.2f}")
            except Exception as e:
                st.error(f"An error occurred during forecasting: {e}")