import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Stock Prediction App")

selected_stock = st.text_input("Select dataset for prediction", "AAPL")

# Check if the selected_stock is empty
if not selected_stock:
    st.warning("Please enter a valid stock symbol.")
else:
    n_years = st.slider("Years of prediction:", 1, 4)
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        try:
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            return None

    data_load_state = st.text("Load data...")

    data = load_data(selected_stock)
    data_load_state.text("Loading data...done!")

    # Check if we got valid data
    if data is None or data.empty or data['Close'].dropna().shape[0] < 2:
        st.error("No sufficient data found for this stock symbol. Please try another (e.g., AAPL, MSFT, TSLA).")
        st.stop()

    # Show raw data
    st.subheader('Raw Data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'],
                                 name='stock_open', line=dict(color='green', width=1)))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'],
                                 name='stock_close', line=dict(color='red', width=1)))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Forecasting
    df_train = data[['Date', 'Close']].dropna()
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    if len(df_train) < 2:
        st.error("Not enough valid rows to train Prophet. Try another stock.")
        st.stop()

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Forecast Data')
    st.write(forecast.tail())

    st.write('Forecast Plot')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write('Forecast Components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)
