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
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text("Load data...")

    # Only load data if selected_stock is not empty
    if selected_stock:
        data = load_data(selected_stock)
        data_load_state.text("Loading data...done!")

        st.subheader('Raw Data')
        st.write(data.tail())

        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open', line=dict(color='green', width=1)))  # Adjust width here
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close', line=dict(color='red', width=1)))  # Adjust width here
            fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)




        plot_raw_data()

        # Forecasting
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader('Forecast Data')
        st.write(forecast.tail())

        st.write('forecast data')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write('forecast components')
        fig2 = m.plot_components(forecast)
        st.write(fig2)
