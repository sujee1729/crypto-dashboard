import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title("🚀 Crypto AI Dashboard")

# Load data
btc = pd.read_csv("BTC.csv")
eth = pd.read_csv("ETH.csv")
bnb = pd.read_csv("BNB.csv")

btc["timestamp"] = pd.to_datetime(btc["timestamp"])
eth["timestamp"] = pd.to_datetime(eth["timestamp"])
bnb["timestamp"] = pd.to_datetime(bnb["timestamp"])

# Price charts
st.header("Crypto Prices")

st.line_chart(btc.set_index("timestamp")["close"])
st.line_chart(eth.set_index("timestamp")["close"])
st.line_chart(bnb.set_index("timestamp")["close"])

# Returns
btc["returns"] = btc["close"].pct_change()
eth["returns"] = eth["close"].pct_change()
bnb["returns"] = bnb["close"].pct_change()

# Volatility
btc_vol = btc["returns"].std() * np.sqrt(24)
eth_vol = eth["returns"].std() * np.sqrt(24)
bnb_vol = bnb["returns"].std() * np.sqrt(24)

st.header("Volatility")

st.write("BTC Volatility:", btc_vol)
st.write("ETH Volatility:", eth_vol)
st.write("BNB Volatility:", bnb_vol)

# Crash risk
def crash_risk(vol):

    if vol > 0.05:
        return "HIGH"

    elif vol > 0.03:
        return "MEDIUM"

    else:
        return "LOW"

st.header("Crash Risk")

st.write("BTC Crash Risk:", crash_risk(btc_vol))
st.write("ETH Crash Risk:", crash_risk(eth_vol))
st.write("BNB Crash Risk:", crash_risk(bnb_vol))

# Monte Carlo
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

btc_mc = monte_carlo(btc["close"])

st.header("BTC Monte Carlo Forecast")

fig, ax = plt.subplots()

for i in range(50):
    ax.plot(btc_mc[i])

st.pyplot(fig)

# Prediction
final_prices = btc_mc[:, -1]

expected_price = np.mean(final_prices)
bullish_price = np.percentile(final_prices, 75)
bearish_price = np.percentile(final_prices, 25)

st.write("Expected Price:", expected_price)
st.write("Bullish Price:", bullish_price)
st.write("Bearish Price:", bearish_price)

current_price = btc["close"].iloc[-1]

prob_up = np.sum(final_prices > current_price) / len(final_prices)

if prob_up > 0.6:
    signal = "BULLISH"

elif prob_up < 0.4:
    signal = "BEARISH"

else:
    signal = "NEUTRAL"

st.header("AI Trading Signal")

st.write(signal)

# News API
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

st.write("Positive:", positive)
st.write("Neutral:", neutral)
st.write("Negative:", negative)

for n in news_list:
    st.write("•", n)


