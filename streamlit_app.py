# ============================
# IMPORTS
# ============================
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from binance.client import Client

# ============================
# STREAMLIT PAGE CONFIG
# ============================
st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")
st.title("🚀 Binance AI — Crypto Intelligence Dashboard")

# ============================
# BINANCE CLIENT SETUP
# ============================
api_key = ""  # optional, leave empty
api_secret = ""
client = Client(api_key, api_secret)

# ============================
# FUNCTIONS
# ============================

# Fetch historical OHLC hourly data
@st.cache_data(ttl=3600)
def fetch_hourly_data(symbol="BTCUSDT", interval="1h", days=1500):
    start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%d %b %Y %H:%M:%S")
    klines = client.get_historical_klines(symbol, interval, start_str)
    df = pd.DataFrame(klines, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df = df[["timestamp","open","high","low","close","volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col])
    return df

# Monte Carlo simulation
def monte_carlo(df, current_price, simulations=300, steps=24, sentiment_factor=0):
    log_returns = np.log(df["close"]/df["close"].shift(1)).dropna()
    mu = log_returns.mean() + sentiment_factor
    sigma = log_returns.std()
    
    mc_results = []
    for _ in range(simulations):
        prices = [current_price]
        for _ in range(steps):
            shock = np.random.normal(mu, sigma)
            prices.append(prices[-1]*np.exp(shock))
        mc_results.append(prices)
    return np.array(mc_results)

# Fetch news sentiment
@st.cache_data(ttl=600)
def get_sentiment():
    try:
        data_news = requests.get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN").json()
        news_list = [a["title"] for a in data_news.get("Data", [])[:10]]
    except:
        news_list = ["No news available"]
    
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(n)["compound"] for n in news_list]
    pos = sum(1 for s in scores if s>0.2)
    neg = sum(1 for s in scores if s<-0.2)
    neu = sum(1 for s in scores if -0.2<=s<=0.2)
    return news_list, pos, neu, neg

# ============================
# SIDEBAR CONTROLS
# ============================
st.sidebar.header("AI Controls")
coin_choice = st.sidebar.selectbox("Select Coin", ["BTCUSDT","ETHUSDT","BNBUSDT"])
simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 1000, 300)
forecast_steps = st.sidebar.slider("Forecast Hours", 12, 72, 24)

# ============================
# FETCH DATA
# ============================
st.sidebar.info("Fetching historical data...")
df = fetch_hourly_data(symbol=coin_choice, interval="1h", days=1500)
current_price = df["close"].iloc[-1]

# ============================
# PRICE CHART
# ============================
st.subheader(f"{coin_choice} Price Chart")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], mode="lines", name="Close Price"))
fig_price.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig_price, use_container_width=True)

# ============================
# VOLATILITY & RISK
# ============================
df["returns"] = df["close"].pct_change()
volatility = df["returns"].std() * np.sqrt(24)
risk = "HIGH" if volatility>0.05 else "MEDIUM" if volatility>0.03 else "LOW"

# ============================
# NEWS SENTIMENT
# ============================
st.subheader("Crypto News Sentiment")
news_list, positive, neutral, negative = get_sentiment()
col1,col2,col3 = st.columns(3)
col1.metric("Positive News", positive)
col2.metric("Neutral News", neutral)
col3.metric("Negative News", negative)

fig_news = px.pie(
    names=["Positive","Neutral","Negative"],
    values=[positive,neutral,negative],
    color_discrete_map={"Positive":"green","Neutral":"gray","Negative":"red"}
)
fig_news.update_layout(template="plotly_dark")
st.plotly_chart(fig_news)

st.subheader("Latest Crypto News")
for n in news_list:
    st.write("•", n)

# ============================
# SENTIMENT FACTOR
# ============================
sentiment_factor = 0.001 if positive > negative else -0.001 if negative > positive else 0

# ============================
# MONTE CARLO SIMULATION
# ============================
mc = monte_carlo(df, current_price, simulations=simulations, steps=forecast_steps, sentiment_factor=sentiment_factor)

mc_df = pd.DataFrame(mc.T)
percentiles = mc_df.quantile([0.25,0.5,0.75], axis=1).T

fig_mc = go.Figure()
fig_mc.add_trace(go.Scatter(y=percentiles[0.5], name="Median", line=dict(color="yellow")))
fig_mc.add_trace(go.Scatter(y=percentiles[0.25], name="25th percentile", line=dict(dash="dash", color="red")))
fig_mc.add_trace(go.Scatter(y=percentiles[0.75], name="75th percentile", line=dict(dash="dash", color="green")))
fig_mc.update_layout(template="plotly_dark", title="Monte Carlo Forecast")
st.subheader("Monte Carlo Forecast")
st.plotly_chart(fig_mc, use_container_width=True)

# ============================
# AI PREDICTIONS & SIGNAL
# ============================
final_prices = mc[:,-1]
expected_price = np.mean(final_prices)
bullish_price = np.percentile(final_prices, 75)
bearish_price = np.percentile(final_prices, 25)
prob_up = np.sum(final_prices > current_price)/len(final_prices)
signal = "BULLISH" if prob_up>0.6 else "BEARISH" if prob_up<0.4 else "NEUTRAL"

st.subheader("AI Metrics")
col4,col5,col6 = st.columns(3)
col4.metric("Volatility", round(volatility,4))
col5.metric("Crash Risk", risk)
col6.metric("AI Signal", signal)

col7,col8,col9 = st.columns(3)
col7.metric("Current Price", round(current_price,2))
col8.metric("Expected Price", round(expected_price,2))
col9.metric("Bullish Target", round(bullish_price,2))

# ============================
# PORTFOLIO SIMULATOR
# ============================
st.subheader("Portfolio Simulator")
initial_investment = st.number_input("Initial Investment (USD)", 100, 100000, 1000)
coins_owned = initial_investment/current_price
future_value = coins_owned * expected_price
profit = future_value - initial_investment

st.write("Future Value:", round(future_value,2))
st.write("Expected Profit:", round(profit,2))

# ============================
# MARKET HEATMAP
# ============================
st.subheader("Market Heatmap")
coins_list = ["BTC","ETH","BNB","XRP","ADA"]
changes = np.random.uniform(-5,5,len(coins_list))
heat_df = pd.DataFrame({"Coin":coins_list, "Change":changes})

fig_heat = px.bar(
    heat_df,
    x="Coin",
    y="Change",
    color="Change",
    color_continuous_scale="RdYlGn"
)
fig_heat.update_layout(template="plotly_dark")
st.plotly_chart(fig_heat, use_container_width=True)
