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
# SIDEBAR CONTROLS
# ================================
st.sidebar.header("AI Controls")
coin_choice = st.sidebar.selectbox("Select Coin", ["BTC","ETH","BNB"])
simulations = st.sidebar.slider("Monte Carlo Simulations", 100, 1000, 300)
forecast_steps = st.sidebar.slider("Forecast Hours", 12, 72, 24)
indicator_choice = st.sidebar.selectbox("Technical Indicator", ["None","Moving Average","RSI"])

# ================================
# LOAD CSV DATA
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
# LIVE PRICE FETCH
# ================================
@st.cache_data(ttl=60)
def fetch_live_price(symbol):
    try:
        url = f"https://min-api.cryptocompare.com/data/price?fsym={symbol}&tsyms=USD"
        data = requests.get(url).json()
        return data.get("USD", df["close"].iloc[-1])
    except:
        return df["close"].iloc[-1]

current_price = fetch_live_price(coin_choice)

# ================================
# PRICE CHART
# ================================
st.subheader(f"{coin_choice} Price Chart")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], mode="lines", name="Price"))

# Technical Indicators
if indicator_choice == "Moving Average":
    df["MA20"] = df["close"].rolling(20).mean()
    fig_price.add_trace(go.Scatter(x=df["timestamp"], y=df["MA20"], name="MA20"))
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

fig_price.update_layout(height=400, template="plotly_dark")
st.plotly_chart(fig_price, use_container_width=True)

# ================================
# RETURNS & VOLATILITY
# ================================
df["returns"] = df["close"].pct_change()
volatility = df["returns"].std() * np.sqrt(24)
risk = "HIGH" if volatility>0.05 else "MEDIUM" if volatility>0.03 else "LOW"

# ================================
# MONTE CARLO SIMULATION
# ================================
@st.cache_data
def monte_carlo(price_series, simulations, steps):
    last_price = price_series.iloc[-1]
    returns = price_series.pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    results = []
    for _ in range(simulations):
        prices = [last_price]
        for _ in range(steps):
            shock = np.random.normal(mu, sigma)
            prices.append(prices[-1]*(1+shock))
        results.append(prices)
    return np.array(results)

mc = monte_carlo(df["close"], simulations, forecast_steps)

# Monte Carlo percentile plot
mc_df = pd.DataFrame(mc.T)
percentiles = mc_df.quantile([0.25,0.5,0.75], axis=1).T
fig_mc = go.Figure()
fig_mc.add_trace(go.Scatter(y=percentiles[0.5], name="Median", line=dict(color="yellow")))
fig_mc.add_trace(go.Scatter(y=percentiles[0.25], name="25th percentile", line=dict(dash="dash", color="red")))
fig_mc.add_trace(go.Scatter(y=percentiles[0.75], name="75th percentile", line=dict(dash="dash", color="green")))
fig_mc.update_layout(template="plotly_dark", title="Monte Carlo Forecast")
st.subheader("Monte Carlo Forecast")
st.plotly_chart(fig_mc, use_container_width=True)

# ================================
# PREDICTIONS & AI SIGNAL
# ================================
final_prices = mc[:,-1]
expected_price = np.mean(final_prices)
bullish_price = np.percentile(final_prices,75)
bearish_price = np.percentile(final_prices,25)
prob_up = np.sum(final_prices>current_price)/len(final_prices)
signal = "BULLISH" if prob_up>0.6 else "BEARISH" if prob_up<0.4 else "NEUTRAL"

# ================================
# AI METRICS
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
# PORTFOLIO SIMULATOR
# ================================
st.subheader("Portfolio Simulator")
initial = st.number_input("Initial Investment", 100, 100000, 1000)
coins_owned = initial/current_price
future_value = coins_owned*expected_price
profit = future_value-initial
st.write("Future Value:", round(future_value,2))
st.write("Expected Profit:", round(profit,2))

# ================================
# NEWS SENTIMENT
# ================================
st.subheader("Crypto News Sentiment")
try:
    response = requests.get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN")
    data = response.json()
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
positive = sum(1 for s in scores if s>0.2)
neutral = sum(1 for s in scores if -0.2<=s<=0.2)
negative = sum(1 for s in scores if s<-0.2)

col7,col8,col9 = st.columns(3)
col7.metric("Positive News", positive)
col8.metric("Neutral News", neutral)
col9.metric("Negative News", negative)

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

# ================================
# MARKET HEATMAP
# ================================
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
