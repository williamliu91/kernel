import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import statsmodels.api as sm
import plotly.graph_objects as go

# List of 50 popular US shares
popular_shares = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "UNH", "V",
    "JPM", "MA", "HD", "DIS", "PYPL", "NFLX", "INTC", "CSCO", "PEP", "ADBE",
    "KO", "CMCSA", "T", "XOM", "MRK", "NKE", "PFE", "ABT", "CVX", "MCD",
    "WMT", "IBM", "BA", "CAT", "MDT", "AMGN", "ORCL", "GILD", "AMT", "AVGO",
    "TXN", "COST", "UPS", "LLY", "BMY", "TMO", "SBUX", "GE", "CVS", "QCOM"
]

# MACD Calculation
def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    df['EMA_short'] = df['Close'].ewm(span=short_period, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long_period, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    return df

def kernel_regression(df, bandwidth=5, future_days=30):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values
    
    # Ensure the bandwidth is passed as a list
    kernel_model = sm.nonparametric.KernelReg(endog=y, exog=X, var_type='c', bw=[bandwidth])
    
    y_hat, _ = kernel_model.fit()
    
    # Future Predictions
    future_X = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
    future_predictions, _ = kernel_model.fit(future_X)
    
    future_dates = pd.date_range(df.index[-1], periods=future_days+1, freq='D')[1:]
    
    return y_hat, future_dates, future_predictions

# Plot function
def plot_analysis(df, y_hat, future_dates, future_predictions):
    fig = go.Figure()

    # Historical Close Prices
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))

    # MACD Line and Signal Line
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='orange')))

    # Future Predictions
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Prediction', line=dict(dash='dot')))

    # Determine trend direction
    trend_changes = np.diff(y_hat)
    trend_segments = []
    start_idx = 0

    for i in range(1, len(trend_changes)):
        if (trend_changes[i-1] > 0 and trend_changes[i] <= 0) or (trend_changes[i-1] < 0 and trend_changes[i] >= 0):
            # Store the segment
            trend_segments.append((start_idx, i))
            start_idx = i

    # Add the last segment
    trend_segments.append((start_idx, len(y_hat)))

    # Initialize signal columns
    df['Buy_Signal'] = np.nan
    df['Sell_Signal'] = np.nan

    # Add trend segments to the plot and generate signals
    for start_idx, end_idx in trend_segments:
        if end_idx - start_idx > 1:
            segment_dates = df.index[start_idx:end_idx]
            segment_values = y_hat[start_idx:end_idx]
            color = 'green' if segment_values[-1] > segment_values[0] else 'red'
            
            # Add trend segment to the plot
            fig.add_trace(go.Scatter(x=segment_dates, y=segment_values, mode='lines', name='Trend', line=dict(color=color)))
            
            # Generate signals based on trend direction
            if color == 'green':
                df['Buy_Signal'].iloc[start_idx:end_idx] = np.where(
                    (df['MACD'].iloc[start_idx:end_idx] > df['Signal_Line'].iloc[start_idx:end_idx]) &
                    (df['MACD'].shift(1).iloc[start_idx:end_idx] <= df['Signal_Line'].shift(1).iloc[start_idx:end_idx]), 
                    df['Close'].iloc[start_idx:end_idx], np.nan)
            else:
                df['Sell_Signal'].iloc[start_idx:end_idx] = np.where(
                    (df['MACD'].iloc[start_idx:end_idx] < df['Signal_Line'].iloc[start_idx:end_idx]) &
                    (df['MACD'].shift(1).iloc[start_idx:end_idx] >= df['Signal_Line'].shift(1).iloc[start_idx:end_idx]), 
                    df['Close'].iloc[start_idx:end_idx], np.nan)

    # Add buy/sell signals to the plot
    fig.add_trace(go.Scatter(x=df.index, y=df['Buy_Signal'], mode='markers', name='Buy Signal', marker=dict(color='green', size=10)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Sell_Signal'], mode='markers', name='Sell Signal', marker=dict(color='red', size=10)))

    fig.update_layout(title='MACD with Kernel Regression Prediction and Trading Signals', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')

    st.plotly_chart(fig)

# Main Streamlit App
def main():
    st.title('Stock Analysis with MACD and Kernel Regression')

    # Dropdown menu for selecting stock
    ticker = st.sidebar.selectbox('Select Stock', popular_shares)

    # User inputs for MACD periods and bandwidth
    short_period = st.sidebar.slider('Short EMA Period (MACD)', 5, 50, 12)
    long_period = st.sidebar.slider('Long EMA Period (MACD)', 10, 100, 26)
    signal_period = st.sidebar.slider('Signal Line Period', 5, 50, 9)
    bandwidth = st.sidebar.slider('Kernel Regression Bandwidth', 1, 20, 5)
    future_days = st.sidebar.slider('Days to Predict', 1, 60, 30)

    # Fetch data for selected stock
    start_date = st.sidebar.date_input('Start Date', datetime(2024, 1, 1))
    end_date = st.sidebar.date_input('End Date', datetime.now() - timedelta(days=1))
    df = yf.download(ticker, start=start_date, end=end_date)

    if not df.empty:
        df = calculate_macd(df, short_period, long_period, signal_period)
        df.dropna(inplace=True)  # Drop rows with NaN values after MACD calculation

        # Apply Kernel Regression for prediction
        y_hat, future_dates, future_predictions = kernel_regression(df, bandwidth, future_days)

        # Plot results
        plot_analysis(df, y_hat, future_dates, future_predictions)

        # Display Model Metrics
        r2_kr = r2_score(df['Close'], y_hat)
        mae_kr = mean_absolute_error(df['Close'], y_hat)
        rmse_kr = np.sqrt(mean_squared_error(df['Close'], y_hat))
        
        # Calculate MAPE
        mape_kr = np.mean(np.abs((df['Close'] - y_hat) / df['Close'])) * 100
        
        st.write(f"**R-squared:** {r2_kr:.4f}")
        st.write(f"**Mean Absolute Error:** ${mae_kr:.2f}")
        st.write(f"**Root Mean Squared Error:** ${rmse_kr:.2f}")
        st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape_kr:.2f}%")

    else:
        st.write("No data available for the selected date range.")

# Run the app
if __name__ == "__main__":
    main()
