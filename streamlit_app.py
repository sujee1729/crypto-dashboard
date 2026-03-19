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
st.title("🚀 OpenClaw AI — Crypto Intelligence Dashboard")

# ================================
# SIDEBAR
# ================================
st.sidebar.header("AI Controls")

coin_choice = st.sidebar.selectbox("Select Coin", ["BTC","ETH","BNB"])
simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 1000, 300)
forecast_steps = st.sidebar.slider("Forecast Hours", 12, 72, 24)
indicator_choice = st.sidebar.selectbox("Technical Indicator", ["None","Moving Average","RSI"])

# ================================
# LOAD DATA
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
# LIVE PRICE (CRITICAL FIX)
# ================================
def get_live_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
        data = requests.get(url).json()
        return float(data["price"])
    except:
        return None

live_price = get_live_price(coin_choice)

if live_price:
    current_price = live_price
    st.success(f"Live Price: {round(current_price,2)}")
else:
    current_price = df["close"].iloc[-1]
    st.warning("Using historical price")

# ================================
# PRICE CHART
# ================================
st.subheader(f"{coin_choice} Price Chart")

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Price"))

if indicator_choice == "Moving Average":
    df["MA50"] = df["close"].rolling(50).mean()
    fig_price.add_trace(go.Scatter(x=df["timestamp"], y=df["MA50"], name="MA50"))

elif indicator_choice == "RSI":
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100/(1+rs))

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df["timestamp"], y=rsi, name="RSI"))

fig_price.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig_price, use_container_width=True)

# ================================
# VOLATILITY
# ================================
df["returns"] = df["close"].pct_change()
volatility = df["returns"].std() * np.sqrt(24)

if volatility > 0.05:
    risk = "HIGH"
elif volatility > 0.03:
    risk = "MEDIUM"
else:
    risk = "LOW"

# ================================
# NEWS SENTIMENT
# ================================
st.subheader("Crypto News Sentiment")

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
    if len(news_list) >= 20:
        break

if not news_list:
    news_list = ["No news available"]

analyzer = SentimentIntensityAnalyzer()

scores = [analyzer.polarity_scores(n)["compound"] for n in news_list]

positive = sum(1 for s in scores if s > 0.2)
neutral = sum(1 for s in scores if -0.2 <= s <= 0.2)
negative = sum(1 for s in scores if s < -0.2)

total_news = max(len(scores), 1)
sentiment_score = (positive - negative) / total_news

# DISPLAY
col1, col2, col3 = st.columns(3)
col1.metric("Positive", positive)
col2.metric("Neutral", neutral)
col3.metric("Negative", negative)

fig_news = px.pie(
    names=["Positive","Neutral","Negative"],
    values=[positive, neutral, negative]
)
fig_news.update_layout(template="plotly_dark")
st.plotly_chart(fig_news)

st.write("### Top News")
for n in news_list[:5]:
    st.write("•", n)

# ================================
# TREND DETECTION
# ================================
df["MA50"] = df["close"].rolling(50).mean()

if current_price > df["MA50"].iloc[-1]:
    trend = "BULLISH"
    trend_score = 0.02
else:
    trend = "BEARISH"
    trend_score = -0.02

# ================================
# MONTE CARLO (GBM)
# ================================
def monte_carlo(df_close, simulations, steps):

    S0 = current_price

    log_returns = np.log(df_close / df_close.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()

    # Combine intelligence
    mu = mu + (sentiment_score * 0.01) + trend_score

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

mc = monte_carlo(df["close"], simulations, forecast_steps)

# ================================
# MC PLOT
# ================================
mc_df = pd.DataFrame(mc.T)
percentiles = mc_df.quantile([0.25,0.5,0.75], axis=1).T

fig_mc = go.Figure()
fig_mc.add_trace(go.Scatter(y=percentiles[0.5], name="Median"))
fig_mc.add_trace(go.Scatter(y=percentiles[0.25], name="25%", line=dict(dash="dash")))
fig_mc.add_trace(go.Scatter(y=percentiles[0.75], name="75%", line=dict(dash="dash")))

fig_mc.update_layout(template="plotly_dark", title="Monte Carlo Forecast")
st.plotly_chart(fig_mc, use_container_width=True)

# ================================
# AI DECISION
# ================================
final_prices = mc[:,-1]

expected_price = np.mean(final_prices)
bullish_price = np.percentile(final_prices, 75)

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
col2.metric("Crash Risk", risk)
col3.metric("AI Signal", signal)

col4,col5,col6 = st.columns(3)
col4.metric("Current Price", round(current_price,2))
col5.metric("Expected Price", round(expected_price,2))
col6.metric("Bullish Target", round(bullish_price,2))

# ================================
# AI EXPLANATION
# ================================
st.subheader("AI Decision Engine")

st.write(f"Signal: {signal}")
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
st.write("Profit:", round(profit,2))
