import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Page setup
st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")

st.title("🚀 Crypto AI Dashboard")

# Sidebar
st.sidebar.title("AI Controls")
coin_choice = st.sidebar.selectbox("Select Coin", ["BTC", "ETH", "BNB"])
simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 1000, 300)

# Load data
btc = pd.read_csv("BTC.csv")
eth = pd.read_csv("ETH.csv")
bnb = pd.read_csv("BNB.csv")

btc["timestamp"] = pd.to_datetime(btc["timestamp"])
eth["timestamp"] = pd.to_datetime(eth["timestamp"])
bnb["timestamp"] = pd.to_datetime(bnb["timestamp"])

# Choose coin
if coin_choice == "BTC":
    df = btc
elif coin_choice == "ETH":
    df = eth
else:
    df = bnb

# Price Chart
st.header(f"{coin_choice} Price Chart")
st.line_chart(df.set_index("timestamp")["close"])

# Returns
df["returns"] = df["close"].pct_change()

# Volatility
vol = df["returns"].std() * np.sqrt(24)

# Crash risk function
def crash_risk(vol):
    if vol > 0.05:
        return "HIGH"
    elif vol > 0.03:
        return "MEDIUM"
    else:
        return "LOW"

risk = crash_risk(vol)

# Monte Carlo Simulation
def monte_carlo(price_series, simulations=300, steps=24):

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

mc = monte_carlo(df["close"], simulations)

# Monte Carlo Chart
st.header("Monte Carlo Forecast")

fig, ax = plt.subplots()

for i in range(50):
    ax.plot(mc[i])

st.pyplot(fig)

# Prediction
final_prices = mc[:, -1]

expected_price = np.mean(final_prices)
bullish_price = np.percentile(final_prices, 75)
bearish_price = np.percentile(final_prices, 25)

current_price = df["close"].iloc[-1]

prob_up = np.sum(final_prices > current_price) / len(final_prices)

if prob_up > 0.6:
    signal = "BULLISH"
elif prob_up < 0.4:
    signal = "BEARISH"
else:
    signal = "NEUTRAL"

# Metrics dashboard
st.header("AI Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Volatility", round(vol, 4))
col2.metric("Crash Risk", risk)
col3.metric("AI Signal", signal)

col4, col5, col6 = st.columns(3)

col4.metric("Expected Price", round(expected_price, 2))
col5.metric("Bullish Price", round(bullish_price, 2))
col6.metric("Bearish Price", round(bearish_price, 2))

# News sentiment
st.header("Crypto News Sentiment")

url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

response = requests.get(url)
data = response.json()

news_list = []

if "Data" in data and isinstance(data["Data"], list):
    articles = data["Data"]
    news_list = [article["title"] for article in articles[:10]]
else:
    news_list = ["No news available"]

analyzer = SentimentIntensityAnalyzer()

scores = [analyzer.polarity_scores(news)["compound"] for news in news_list]

positive = sum(1 for s in scores if s > 0.2)
negative = sum(1 for s in scores if s < -0.2)
neutral = sum(1 for s in scores if -0.2 <= s <= 0.2)

st.subheader("Market Sentiment")

col7, col8, col9 = st.columns(3)

col7.metric("Positive News", positive)
col8.metric("Neutral News", neutral)
col9.metric("Negative News", negative)

st.subheader("Latest Crypto News")

for n in news_list:
    st.write("•", n)

