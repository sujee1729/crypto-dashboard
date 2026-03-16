import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")

st.title("🚀 Crypto AI Trading Dashboard")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("AI Controls")

coin_choice = st.sidebar.selectbox(
    "Select Coin",
    ["BTC", "ETH", "BNB"]
)

simulations = st.sidebar.slider(
    "Monte Carlo Simulations",
    100,
    1000,
    300
)

steps = st.sidebar.slider(
    "Forecast Hours",
    12,
    72,
    24
)

# ---------------------------
# LOAD DATA
# ---------------------------

@st.cache_data
def load_data(coin):
    df = pd.read_csv(f"{coin}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data(coin_choice)

# ---------------------------
# PRICE CHART
# ---------------------------

st.subheader(f"{coin_choice} Price Chart")

st.line_chart(
    df.set_index("timestamp")["close"],
    use_container_width=True
)

# ---------------------------
# RETURNS & VOLATILITY
# ---------------------------

df = df.copy()
df["returns"] = df["close"].pct_change()

vol = df["returns"].std() * np.sqrt(24)

# ---------------------------
# CRASH RISK
# ---------------------------

def crash_risk(vol):

    if vol > 0.05:
        return "HIGH"

    elif vol > 0.03:
        return "MEDIUM"

    else:
        return "LOW"

risk = crash_risk(vol)

# ---------------------------
# MONTE CARLO SIMULATION
# ---------------------------

def monte_carlo(price_series, simulations, steps):

    last_price = price_series.iloc[-1]

    returns = price_series.pct_change().dropna()

    mu = returns.mean()
    sigma = returns.std()

    results = []

    for i in range(simulations):

        prices = [last_price]

        for j in range(steps):

            shock = np.random.normal(mu, sigma)

            price = prices[-1] * (1 + shock)

            prices.append(price)

        results.append(prices)

    return np.array(results)

mc = monte_carlo(df["close"], simulations, steps)

# ---------------------------
# MONTE CARLO CHART
# ---------------------------

st.subheader("📊 Monte Carlo Forecast")

fig, ax = plt.subplots()

for i in range(min(100, simulations)):
    ax.plot(mc[i], alpha=0.2)

ax.set_title("Price Simulation Paths")
ax.set_xlabel("Hours")
ax.set_ylabel("Price")

st.pyplot(fig)

# ---------------------------
# AI PREDICTION
# ---------------------------

final_prices = mc[:, -1]

expected_price = np.mean(final_prices)
bullish_price = np.percentile(final_prices, 75)
bearish_price = np.percentile(final_prices, 25)

current_price = df["close"].iloc[-1]

prob_up = np.sum(final_prices > current_price) / len(final_prices)

if prob_up > 0.6:
    signal = "BULLISH 📈"

elif prob_up < 0.4:
    signal = "BEARISH 📉"

else:
    signal = "NEUTRAL ⚖️"

# ---------------------------
# AI METRICS PANEL
# ---------------------------

st.subheader("🤖 AI Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Volatility", round(vol, 4))
col2.metric("Crash Risk", risk)
col3.metric("AI Signal", signal)

col4, col5, col6 = st.columns(3)

col4.metric("Current Price", round(current_price, 2))
col5.metric("Expected Price", round(expected_price, 2))
col6.metric("Bullish Target", round(bullish_price, 2))

# ---------------------------
# NEWS SENTIMENT
# ---------------------------

st.subheader("📰 Crypto News Sentiment")

url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

response = requests.get(url)

data = response.json()

news_list = []

if "Data" in data:

    articles = data["Data"]

    news_list = [article["title"] for article in articles[:10]]

else:

    news_list = ["No news available"]

analyzer = SentimentIntensityAnalyzer()

scores = [analyzer.polarity_scores(news)["compound"] for news in news_list]

positive = sum(1 for s in scores if s > 0.2)
negative = sum(1 for s in scores if s < -0.2)
neutral = sum(1 for s in scores if -0.2 <= s <= 0.2)

col7, col8, col9 = st.columns(3)

col7.metric("Positive News", positive)
col8.metric("Neutral News", neutral)
col9.metric("Negative News", negative)

# ---------------------------
# LATEST NEWS
# ---------------------------

st.subheader("Latest Crypto News")

for news in news_list:
    st.write("•", news)
