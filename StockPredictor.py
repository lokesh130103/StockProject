import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

# Select stock
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Prediction period
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Load data from Yahoo Finance
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Show raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for Prophet
df_train = data[['Date', 'Close']].dropna()  # Drop NaN values
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Debug info
st.subheader("Training data preview")
st.write(df_train.head())
st.write("Shape:", df_train.shape)
st.write("Missing values:", df_train.isna().sum())

# Check if we have data
if df_train.empty:
    st.error("No data available for this stock. Try another ticker.")
else:
    # Forecast
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
