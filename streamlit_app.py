import pandas as pd
import numpy as np
import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")
st.title("🚀 OpenClaw AI — Live Crypto Intelligence")

# ================================
# SIDEBAR
# ================================
st.sidebar.header("AI Controls")

coin_choice = st.sidebar.selectbox("Select Coin", ["BTC","ETH","BNB"])
simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 2000, 500)
forecast_steps = st.sidebar.slider("Forecast Hours", 12, 72, 24)

# ================================
# FETCH LIVE HISTORICAL DATA
# ================================
@st.cache_data(ttl=300)  # cache 5 min
def fetch_binance_data(symbol, interval="1h", limit=500):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT&interval={interval}&limit={limit}"
        data = requests.get(url, timeout=5).json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","num_trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms")
        df["close"] = df["close"].astype(float)
        return df[["timestamp","close"]]
    except Exception as e:
        st.warning(f"Could not fetch historical data: {e}")
        return pd.DataFrame(columns=["timestamp","close"])

df = fetch_binance_data(coin_choice)

# ================================
# LIVE PRICE
# ================================
def get_live_price(symbol, fallback_df=None):
    """
    Get live price from Binance. 
    If failed, fallback_df is used as the source of last price.
    """
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
        data = requests.get(url, timeout=5).json()
        return float(data["price"])
    except:
        # Use fallback_df if provided and non-empty
        if fallback_df is not None and not fallback_df.empty:
            return fallback_df["close"].iloc[-1]
        else:
            return None

current_price = get_live_price(coin_choice, fallback_df=df)

if current_price is None:
    st.error("Cannot fetch live price and no fallback data available.")
    st.stop()
else:
    st.success(f"Live Price: {round(current_price,2)} USD")

# ================================
# PRICE CHART
# ================================
st.subheader("Price Chart (Recent Data)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Price"))
fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ================================
# RETURNS & VOLATILITY
# ================================
df["returns"] = df["close"].pct_change()
volatility = df["returns"].std()

# ================================
# TREND DETECTION
# ================================
df["MA50"] = df["close"].rolling(50).mean()
if not df["MA50"].isna().all():
    if current_price > df["MA50"].iloc[-1]:
        trend = "BULLISH"
        trend_boost = 0.01
    else:
        trend = "BEARISH"
        trend_boost = -0.01
else:
    trend = "NEUTRAL"
    trend_boost = 0

# ================================
# NEWS SENTIMENT
# ================================
st.subheader("News Sentiment")
try:
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    news_data = requests.get(url, timeout=5).json()
except:
    news_data = {}

news_list = [a["title"] for a in news_data.get("Data", []) if a.get("title")]
if not news_list:
    news_list = ["No news available"]

analyzer = SentimentIntensityAnalyzer()
scores = [analyzer.polarity_scores(n)["compound"] for n in news_list]

positive = sum(s>0.2 for s in scores)
neutral  = sum(-0.2 <= s <= 0.2 for s in scores)
negative = sum(s<-0.2 for s in scores)

sentiment_score = (positive - negative) / max(len(scores),1)
sentiment_boost = sentiment_score * 0.01

col1,col2,col3 = st.columns(3)
col1.metric("Positive", positive)
col2.metric("Neutral", neutral)
col3.metric("Negative", negative)

# ================================
# MONTE CARLO SIMULATION
# ================================
def monte_carlo(df_close, S0, simulations, steps):
    log_returns = np.log(df_close / df_close.shift(1)).dropna()
    mu = log_returns.mean() + trend_boost + sentiment_boost
    sigma = log_returns.std()
    dt = 1

    mc = []
    for _ in range(simulations):
        prices = [S0]
        for _ in range(steps):
            shock = np.random.normal(0,1)
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * shock
            prices.append(prices[-1] * np.exp(drift + diffusion))
        mc.append(prices)
    return np.array(mc)

if not df.empty:
    mc = monte_carlo(df["close"], current_price, simulations, forecast_steps)
else:
    mc = np.array([[current_price]* (forecast_steps+1)] * simulations)

# ================================
# MONTE CARLO PLOT
# ================================
mc_df = pd.DataFrame(mc.T)
percentiles = mc_df.quantile([0.25,0.5,0.75], axis=1).T

fig_mc = go.Figure()
fig_mc.add_trace(go.Scatter(y=percentiles[0.5], name="Median"))
fig_mc.add_trace(go.Scatter(y=percentiles[0.25], name="25%", line=dict(dash="dash")))
fig_mc.add_trace(go.Scatter(y=percentiles[0.75], name="75%", line=dict(dash="dash")))
fig_mc.update_layout(template="plotly_dark")
st.subheader("Monte Carlo Forecast (Live Based)")
st.plotly_chart(fig_mc, use_container_width=True)

# ================================
# AI DECISION
# ================================
final_prices = mc[:,-1]
expected_price = np.mean(final_prices)
bullish_price = np.percentile(final_prices,75)
prob_up = np.sum(final_prices > current_price) / len(final_prices)
confidence = abs(prob_up - 0.5) * 200

if prob_up > 0.6:
    signal = "STRONG BULLISH" if confidence > 30 else "BULLISH"
elif prob_up < 0.4:
    signal = "STRONG BEARISH" if confidence > 30 else "BEARISH"
else:
    signal = "NEUTRAL"

# ================================
# METRICS
# ================================
st.subheader("AI Metrics")
col1,col2,col3 = st.columns(3)
col1.metric("Volatility", round(volatility,4))
col2.metric("Trend", trend)
col3.metric("AI Signal", signal)

col4,col5,col6 = st.columns(3)
col4.metric("Current Price", round(current_price,2))
col5.metric("Expected Price", round(expected_price,2))
col6.metric("Bullish Target", round(bullish_price,2))
