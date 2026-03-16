import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")

st.title("🚀 Advanced Crypto AI Trading Dashboard")

# ================================
# SIDEBAR CONTROLS
# ================================

st.sidebar.title("AI Controls")

coin_choice = st.sidebar.selectbox(
    "Select Coin",
    ["BTC","ETH","BNB"]
)

simulations = st.sidebar.slider(
    "Monte Carlo Simulations",
    100,
    1000,
    300
)

forecast_steps = st.sidebar.slider(
    "Forecast Hours",
    12,
    72,
    24
)

indicator_choice = st.sidebar.selectbox(
    "Technical Indicator",
    ["None","Moving Average","RSI","MACD"]
)

# ================================
# LOAD DATA
# ================================

btc = pd.read_csv("BTC.csv")
eth = pd.read_csv("ETH.csv")
bnb = pd.read_csv("BNB.csv")

btc["timestamp"] = pd.to_datetime(btc["timestamp"])
eth["timestamp"] = pd.to_datetime(eth["timestamp"])
bnb["timestamp"] = pd.to_datetime(bnb["timestamp"])

if coin_choice == "BTC":
    df = btc.copy()
elif coin_choice == "ETH":
    df = eth.copy()
else:
    df = bnb.copy()

# ================================
# PRICE CHART
# ================================

st.subheader(f"{coin_choice} Price Chart")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["close"],
        mode="lines",
        name="Price"
    )
)

fig.update_layout(
    height=400,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ================================
# RETURNS & VOLATILITY
# ================================

df["returns"] = df["close"].pct_change()

volatility = df["returns"].std() * np.sqrt(24)

# ================================
# TECHNICAL INDICATORS
# ================================

if indicator_choice == "Moving Average":

    df["MA20"] = df["close"].rolling(20).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Price"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MA20"], name="MA20"))

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

if indicator_choice == "RSI":

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100/(1+rs))

    fig = px.line(rsi)

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig)

# ================================
# CRASH RISK
# ================================

def crash_risk(vol):

    if vol > 0.05:
        return "HIGH"

    elif vol > 0.03:
        return "MEDIUM"

    else:
        return "LOW"

risk = crash_risk(volatility)

# ================================
# MONTE CARLO SIMULATION
# ================================

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

            price = prices[-1]*(1+shock)

            prices.append(price)

        results.append(prices)

    return np.array(results)

mc = monte_carlo(df["close"], simulations, forecast_steps)

# ================================
# MONTE CARLO PLOT
# ================================

st.subheader("Monte Carlo Forecast")

fig, ax = plt.subplots()

for i in range(min(100, simulations)):
    ax.plot(mc[i], alpha=0.3)

st.pyplot(fig)

# ================================
# PREDICTIONS
# ================================

final_prices = mc[:,-1]

expected_price = np.mean(final_prices)
bullish_price = np.percentile(final_prices,75)
bearish_price = np.percentile(final_prices,25)

current_price = df["close"].iloc[-1]

prob_up = np.sum(final_prices>current_price)/len(final_prices)

if prob_up>0.6:
    signal="BULLISH"

elif prob_up<0.4:
    signal="BEARISH"

else:
    signal="NEUTRAL"

# ================================
# AI METRICS
# ================================

st.subheader("AI Metrics")

col1,col2,col3 = st.columns(3)

col1.metric("Volatility",round(volatility,4))
col2.metric("Crash Risk",risk)
col3.metric("AI Signal",signal)

col4,col5,col6 = st.columns(3)

col4.metric("Current Price",round(current_price,2))
col5.metric("Expected Price",round(expected_price,2))
col6.metric("Bullish Target",round(bullish_price,2))

# ================================
# PORTFOLIO SIMULATOR
# ================================

st.subheader("Portfolio Simulator")

initial = st.number_input("Initial Investment",100,100000,1000)

coins = initial/current_price

future_value = coins*expected_price

profit = future_value-initial

st.write("Future Value:",round(future_value,2))
st.write("Expected Profit:",round(profit,2))

# ================================
# NEWS SENTIMENT
# ================================

st.subheader("Crypto News Sentiment")

url="https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

response=requests.get(url)

data=response.json()

news_list=[]

if "Data" in data:

    articles=data["Data"]

    news_list=[a["title"] for a in articles[:10]]

analyzer=SentimentIntensityAnalyzer()

scores=[analyzer.polarity_scores(n)["compound"] for n in news_list]

positive=sum(1 for s in scores if s>0.2)
negative=sum(1 for s in scores if s<-0.2)
neutral=sum(1 for s in scores if -0.2<=s<=0.2)

col7,col8,col9=st.columns(3)

col7.metric("Positive News",positive)
col8.metric("Neutral News",neutral)
col9.metric("Negative News",negative)

st.subheader("Latest Crypto News")

for n in news_list:

    st.write("•",n)

# ================================
# MARKET HEATMAP
# ================================

st.subheader("Market Heatmap")

coins=["BTC","ETH","BNB","XRP","ADA"]

changes=np.random.uniform(-5,5,len(coins))

heat_df=pd.DataFrame({
    "Coin":coins,
    "Change":changes
})

fig=px.bar(
    heat_df,
    x="Coin",
    y="Change",
    color="Change",
    color_continuous_scale="RdYlGn"
)

fig.update_layout(template="plotly_dark")

st.plotly_chart(fig)
