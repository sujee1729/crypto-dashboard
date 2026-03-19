
import pandas as pd
import numpy as np
import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")
st.title("🚀 OpenClaw AI — Real-Time Hybrid Crypto Intelligence")

# ================================
# SIDEBAR
# ================================
st.sidebar.header("AI Controls")

coin_choice = st.sidebar.selectbox("Select Coin", ["BTC","ETH","BNB"])
simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 1000, 300)
forecast_steps = st.sidebar.slider("Forecast Hours", 12, 72, 24)

# ================================
# LOAD DATA (FEB DATASET)
# ================================
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

btc = load_csv("BTC.csv")
eth = load_csv("ETH.csv")
bnb = load_csv("BNB.csv")

df = {"BTC": btc, "ETH": eth, "BNB": bnb}[coin_choice].copy()

# ================================
# LIVE PRICE (KEY FIX)
# ================================
def get_live_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
        data = requests.get(url).json()
        return float(data["price"])
    except:
        return None

live_price = get_live_price(coin_choice)
historical_price = df["close"].iloc[-1]

if live_price:
    current_price = live_price
    st.success(f"Live Price: {round(current_price,2)}")
else:
    current_price = historical_price
    st.warning("Using historical price")

# ================================
# PRICE CHART
# ================================
st.subheader("Price Chart (February Data)")

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

if current_price > df["MA50"].iloc[-1]:
    trend = "BULLISH"
    trend_boost = 0.01
else:
    trend = "BEARISH"
    trend_boost = -0.01

# ================================
# NEWS SENTIMENT
# ================================
st.subheader("News Sentiment")

try:
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    data = requests.get(url).json()
except:
    data = {}

news_list = []
for article in data.get("Data", []):
    title = article.get("title")
    if title:
        news_list.append(title)
    if len(news_list) >= 15:
        break

if not news_list:
    news_list = ["No news available"]

analyzer = SentimentIntensityAnalyzer()

scores = [analyzer.polarity_scores(n)["compound"] for n in news_list]

positive = sum(1 for s in scores if s > 0.2)
neutral = sum(1 for s in scores if -0.2 <= s <= 0.2)
negative = sum(1 for s in scores if s < -0.2)

sentiment_score = (positive - negative) / max(len(scores),1)
sentiment_boost = sentiment_score * 0.01

col1,col2,col3 = st.columns(3)
col1.metric("Positive", positive)
col2.metric("Neutral", neutral)
col3.metric("Negative", negative)

# ================================
# MONTE CARLO (LIVE START)
# ================================
def monte_carlo(df_close, live_price, simulations, steps):

    log_returns = np.log(df_close / df_close.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()

    # Combine intelligence
    mu = mu + trend_boost + sentiment_boost

    S0 = live_price  # 🔥 KEY FIX

    dt = 1
    mc = []

    for _ in range(simulations):
        prices = [S0]

        for _ in range(steps):
            shock = np.random.normal(0,1)
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * shock

            price = prices[-1] * np.exp(drift + diffusion)
            prices.append(price)

        mc.append(prices)

    return np.array(mc)

mc = monte_carlo(df["close"], current_price, simulations, forecast_steps)

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

# ================================
# AI EXPLANATION
# ================================
st.subheader("AI Decision Explanation")

st.write(f"Confidence: {round(confidence,1)}%")
st.write(f"Trend: {trend}")
st.write(f"Sentiment Score: {round(sentiment_score,2)}")

reasons = []

if trend == "BULLISH":
    reasons.append("Price above MA50")
else:
    reasons.append("Price below MA50")

if sentiment_score > 0:
    reasons.append("Positive news sentiment")
elif sentiment_score < 0:
    reasons.append("Negative news sentiment")

if prob_up > 0.5:
    reasons.append("Monte Carlo favors upside")
else:
    reasons.append("Monte Carlo favors downside")

for r in reasons:
    st.write("•", r)

# ================================
# PORTFOLIO
# ================================
st.subheader("Portfolio Simulator")

initial = st.number_input("Investment", 100, 100000, 1000)

coins_owned = initial / current_price
future_value = coins_owned * expected_price
profit = future_value - initial

st.write("Future Value:", round(future_value,2))
st.write("Expected Profit:", round(profit,2))


