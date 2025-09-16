import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Title
st.title("Stock Prediction App")

# Stock input
selected_stock = st.text_input("Enter stock symbol (e.g., AAPL, MSFT, TSLA)", "AAPL")

# Check if stock symbol is entered
if not selected_stock:
    st.warning("Please enter a valid stock symbol.")
else:
    # Slider for number of years
    n_years = st.slider("Years of prediction:", 1, 4)
    period = n_years * 365

    # Data loading function
    @st.cache_data
    def load_data(ticker):
        try:
            data = yf.download(ticker, START, TODAY)
            if data is None or data.empty:
                return None
            data.reset_index(inplace=True)
            return data
        except Exception:
            return None

    data_load_state = st.text("Loading data...")

    # Load data
    data = load_data(selected_stock.upper())  # Ensure ticker is uppercase

    if data is None or data.empty:
        st.error(
            f"Could not fetch data for '{selected_stock}'. "
            "Check if the symbol is correct or try another (e.g., AAPL, MSFT, TSLA, AMZN)."
        )
    else:
        data_load_state.text("Loading data...done!")

        # Show raw data
        st.subheader('Raw Data')
        st.write(data.tail())

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Open'],
                name='stock_open',
                line=dict(color='green', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Close'],
                name='stock_close',
                line=dict(color='red', width=1)
            ))
            fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()

        # Forecasting
        df_train = data[['Date', 'Close']].dropna()
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        if df_train.empty or len(df_train) < 30:
            st.error(
                f"Not enough data available for '{selected_stock}' to train Prophet. "
                "Please try another symbol (e.g., AAPL, MSFT, TSLA, AMZN)."
            )
        else:
            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            # Forecast results
            st.subheader('Forecast Data')
            st.write(forecast.tail())

            st.write('Forecast Plot')
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

            st.write('Forecast Components')
            fig2 = m.plot_components(forecast)
            st.write(fig2)
