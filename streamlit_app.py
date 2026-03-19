# ==============================
# LIVE MONTE CARLO CRYPTO FORECAST STREAMLIT APP
# ==============================
import pandas as pd
import numpy as np
import requests
import datetime
import time
from binance.client import Client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import streamlit as st

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(
    page_title="🚀 Live Crypto AI Forecast",
    layout="wide"
)

st.title("🚀 Live Monte Carlo Crypto Forecast Dashboard")

# ------------------------------
# Binance Client (Public)
# ------------------------------
api_key = ""   # Optional for public data
api_secret = ""
client = Client(api_key, api_secret)

# ------------------------------
# Parameters (Sidebar)
# ------------------------------
coins = st.sidebar.multiselect("Select Coins", ["BTCUSDT", "ETHUSDT", "BNBUSDT"], default=["BTCUSDT","ETHUSDT","BNBUSDT"])
days = st.sidebar.number_input("Days of Historical Data", min_value=100, max_value=2000, value=1500)
simulations = st.sidebar.number_input("Monte Carlo Simulations", min_value=50, max_value=1000, value=300)
forecast_steps = st.sidebar.number_input("Forecast Steps (Hours)", min_value=1, max_value=72, value=24)
refresh_interval = st.sidebar.number_input("Refresh Interval (Seconds)", min_value=60, max_value=600, value=180)

st.sidebar.markdown("---")
st.sidebar.info("This app fetches live crypto data, performs Monte Carlo simulations, computes sentiment, and gives an AI trading signal.")

# ------------------------------
# Functions
# ------------------------------
@st.cache_data(ttl=300)
def fetch_hourly_data(symbol, days=1500):
    start_str = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).strftime("%d %b %Y %H:%M:%S")
    klines = client.get_historical_klines(symbol, "1h", start_str)
    df = pd.DataFrame(klines, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df = df[["timestamp","open","high","low","close","volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col])
    return df

def get_live_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        return float(requests.get(url).json()['price'])
    except:
        return None

def monte_carlo(df, current_price, simulations, steps):
    returns = df["close"].pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    results = []
    for _ in range(simulations):
        prices = [current_price]
        for _ in range(steps):
            shock = np.random.normal(mu, sigma)
            prices.append(prices[-1]*(1+shock))
        results.append(prices)
    return np.array(results)

@st.cache_data(ttl=300)
def get_sentiment():
    url="https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    data = requests.get(url).json()
    news_list = []
    if "Data" in data and isinstance(data["Data"], list):
        news_list = [a["title"] for a in data["Data"][:10]]
    else:
        news_list = ["No news available"]
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(n)["compound"] for n in news_list]
    positive = sum(1 for s in scores if s>0.2)
    negative = sum(1 for s in scores if s<-0.2)
    neutral = sum(1 for s in scores if -0.2<=s<=0.2)
    return positive, neutral, negative, news_list

def final_signal(mc, current_price):
    mc_avg = np.mean(mc[:,-1])
    prob_up = np.sum(mc[:,-1]>current_price)/len(mc)
    signal = "BULLISH" if prob_up>0.6 else "BEARISH" if prob_up<0.4 else "NEUTRAL"
    confidence = prob_up*100 if signal=="BULLISH" else (1-prob_up)*100 if signal=="BEARISH" else 50
    return signal, confidence, mc_avg

# ------------------------------
# MAIN LOOP
# ------------------------------
st.markdown("### Monte Carlo Simulation & AI Signal")
placeholder = st.empty()

while True:
    with placeholder.container():
        for coin in coins:
            st.subheader(f"{coin} Forecast")
            
            df = fetch_hourly_data(coin, days)
            current_price = get_live_price(coin)
            if current_price is None:
                current_price = df["close"].iloc[-1]

            mc = monte_carlo(df, current_price, simulations, forecast_steps)
            volatility = df["close"].pct_change().std() * np.sqrt(24)
            risk = "HIGH" if volatility>0.05 else "MEDIUM" if volatility>0.03 else "LOW"
            sentiment = get_sentiment()
            signal, confidence, mc_avg = final_signal(mc, current_price)

            # Monte Carlo plot (50 paths)
            fig_mc = go.Figure()
            for i in range(min(50, simulations)):
                fig_mc.add_trace(go.Scatter(
                    y=mc[i],
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.3,
                    name=f"{coin} Path" if i==0 else None,
                    showlegend=(i==0)
                ))
            fig_mc.update_layout(
                title=f"{coin} Monte Carlo Forecast",
                xaxis_title="Forecast Steps (Hours)",
                yaxis_title="Price",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            # Metrics
            st.markdown(f"""
            **Current Price:** ${round(current_price,2)}  
            **Volatility:** {round(volatility,4)}  
            **Crash Risk:** {risk}  
            **Monte Carlo Avg Price:** ${round(mc_avg,2)}  
            **AI Signal:** {signal}  
            **Confidence:** {round(confidence,2)}%  
            **Expected Value for $1000 Investment:** ${round(1000/current_price * mc_avg,2)}  
            **Expected Profit:** ${round((1000/current_price * mc_avg)-1000,2)}  
            """)
            st.markdown(f"**Sentiment:** Positive: {sentiment[0]} | Neutral: {sentiment[1]} | Negative: {sentiment[2]}")
            st.markdown("**Latest News Headlines:**")
            for n in sentiment[3]:
                st.markdown(f"• {n}")
        
        st.info(f"Next update in {refresh_interval} seconds...")
        time.sleep(refresh_interval)
