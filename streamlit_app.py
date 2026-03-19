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
st.title("🚀 OpenClaw AI — Hybrid Crypto Intelligence")

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
# LIVE + HISTORICAL PRICE
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

st.subheader("Price Comparison")

col1, col2 = st.columns(2)
col1.metric("Feb Closing Price", round(historical_price,2))

if live_price:
    col2.metric("Live Market Price", round(live_price,2))
else:
    col2.metric("Live Market Price", "Unavailable")

# ================================
# PRICE CHART
# ================================
st.subheader("Price Chart (February Dataset)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Price"))
fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ================================
# VOLATILITY
# ================================
df["returns"] = df["close"].pct_change()
volatility = df["returns"].std()

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
    if len(news_list) >= 10:
        break

if not news_list:
    news_list = ["No news available"]

analyzer = SentimentIntensityAnalyzer()

scores = [analyzer.polarity_scores(n)["compound"] for n in news_list]

positive = sum(1 for s in scores if s > 0.2)
neutral = sum(1 for s in scores if -0.2 <= s <= 0.2)
negative = sum(1 for s in scores if s < -0.2)

colA, colB, colC = st.columns(3)
colA.metric("Positive", positive)
colB.metric("Neutral", neutral)
colC.metric("Negative", negative)

# ================================
# TREND (REAL-TIME)
# ================================
df["MA50"] = df["close"].rolling(50).mean()

if live_price:
    if live_price > df["MA50"].iloc[-1]:
        real_trend = "BULLISH"
    else:
        real_trend = "BEARISH"
else:
    real_trend = "UNKNOWN"

# ================================
# MONTE CARLO (FEB MODEL)
# ================================
def monte_carlo(df_close, simulations, steps):

    S0 = df_close.iloc[-1]

    log_returns = np.log(df_close / df_close.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()

    mc = []

    for _ in range(simulations):
        prices = [S0]

        for _ in range(steps):
            shock = np.random.normal(0,1)
            drift = (mu - 0.5 * sigma**2)
            diffusion = sigma * shock

            price = prices[-1] * np.exp(drift + diffusion)
            prices.append(price)

        mc.append(prices)

    return np.array(mc)

mc = monte_carlo(df["close"], simulations, forecast_steps)

# ================================
# HISTORICAL AI SIGNAL
# ================================
final_prices = mc[:,-1]

expected_price = np.mean(final_prices)

prob_up_hist = np.sum(final_prices > historical_price) / len(final_prices)

if prob_up_hist > 0.6:
    hist_signal = "BULLISH"
elif prob_up_hist < 0.4:
    hist_signal = "BEARISH"
else:
    hist_signal = "NEUTRAL"

# ================================
# REAL-TIME SIGNAL
# ================================
if live_price:
    if live_price > historical_price:
        realtime_signal = "BULLISH"
    else:
        realtime_signal = "BEARISH"
else:
    realtime_signal = "UNKNOWN"

# ================================
# FINAL AI DECISION
# ================================
if hist_signal == "BEARISH" and realtime_signal == "BULLISH":
    final_signal = "HOLD / REVERSAL"
elif hist_signal == "BULLISH" and realtime_signal == "BEARISH":
    final_signal = "UNCERTAIN"
else:
    final_signal = hist_signal

# ================================
# AI METRICS
# ================================
st.subheader("AI Decision Engine")

col1, col2, col3 = st.columns(3)
col1.metric("Historical AI (Feb)", hist_signal)
col2.metric("Real-Time Market", realtime_signal)
col3.metric("Final Decision", final_signal)

st.write("Expected Price (Feb Model):", round(expected_price,2))
st.write("Real-Time Trend:", real_trend)

# ================================
# EXPLANATION
# ================================
st.subheader("AI Explanation")

reasons = []

if hist_signal == "BEARISH":
    reasons.append("February dataset shows bearish structure")

if hist_signal == "BULLISH":
    reasons.append("February dataset shows bullish structure")

if realtime_signal == "BULLISH":
    reasons.append("Current market is above February closing price")

if realtime_signal == "BEARISH":
    reasons.append("Current market below February closing price")

if final_signal == "HOLD / REVERSAL":
    reasons.append("Market shows possible reversal from historical trend")

for r in reasons:
    st.write("•", r)

# ================================
# PORTFOLIO
# ================================
st.subheader("Portfolio Simulation")

initial = st.number_input("Investment", 100, 100000, 1000)

coins_owned = initial / historical_price
future_value = coins_owned * expected_price
profit = future_value - initial

st.write("Future Value:", round(future_value,2))
st.write("Expected Profit:", round(profit,2))
