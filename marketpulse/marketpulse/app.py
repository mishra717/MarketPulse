"""MarketPulse Streamlit app. Run with:  streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from marketpulse.data import load_news, load_prices
from marketpulse.features import add_indicators, latest_features, make_dataset
from marketpulse.model import backtest, predict_tomorrow, walk_forward_eval
from marketpulse.sentiment import average_sentiment, label, score_headlines

st.set_page_config(page_title="MarketPulse", page_icon="📈", layout="wide")
st.title("📈 MarketPulse")
st.caption("Stock trend analysis and next-day direction prediction. Educational use only, not financial advice.")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    start = st.date_input("History from", pd.Timestamp("2015-01-01")).isoformat()
    st.caption("Indian stocks: add .NS (e.g. RELIANCE.NS, TCS.NS)")
    run = st.button("Analyze", type="primary")

if not run:
    st.info("Enter a ticker and press Analyze.")
    st.stop()

try:
    with st.spinner("Loading prices..."):
        prices = load_prices(ticker, start=start)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load data: {e}")
    st.stop()

ind = add_indicators(prices)
data = make_dataset(prices)

tab_price, tab_model, tab_news, tab_fc = st.tabs(
    ["Price and indicators", "Model evaluation", "News sentiment", "Next-day forecast"])

with tab_price:
    last = ind.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"{last['Close']:.2f}", f"{last['ret_1d'] * 100:.2f}%")
    c2.metric("RSI (14)", f"{last['rsi_14']:.1f}")
    c3.metric("Volatility (ann.)", f"{last['vol_20d'] * 100:.1f}%")
    c4.metric("vs 50-day avg", f"{last['price_vs_sma50'] * 100:.1f}%")

    view = ind.tail(250)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"],
                                 low=view["Low"], close=view["Close"], name="Price"), row=1, col=1)
    for col, name in [("sma_10", "SMA 10"), ("sma_50", "SMA 50"),
                      ("bb_upper", "BB upper"), ("bb_lower", "BB lower")]:
        fig.add_trace(go.Scatter(x=view.index, y=view[col], name=name, line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["rsi_14"], name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view["macd"] - view["macd_signal"], name="MACD hist"), row=3, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

with tab_model:
    st.write(f"Walk-forward evaluation on **{len(data):,}** trading days. "
             "Models are trained only on the past and tested on the future.")
    if len(data) < 500:
        st.warning("Not enough history for a reliable evaluation. Use an earlier start date.")
    else:
        with st.spinner("Training and evaluating..."):
            res = walk_forward_eval(data)
        st.dataframe(res.table.style.format({"Accuracy": "{:.3f}", "F1": "{:.3f}",
                                             "Return RMSE": "{:.5f}"}, na_rep="-"),
                     use_container_width=True)
        st.caption(f"Test rows: {res.n_test_rows:,}. A model is only useful if it beats its baseline.")
        col = st.selectbox("Backtest model", [c for c in res.predictions if c.startswith("pred_") and c != "pred_ret"])
        curve = backtest(res.predictions, col)
        st.line_chart(curve)
        st.caption("Long when the model predicts up, cash otherwise, 5 bps per trade. Past results do not predict the future.")

with tab_news:
    try:
        heads = load_news(ticker)
        scored = score_headlines(heads)
        if scored:
            avg = average_sentiment(scored)
            st.metric("Average headline sentiment", f"{avg:+.2f}", label(avg))
            st.dataframe(pd.DataFrame(scored, columns=["Headline", "Score"]), use_container_width=True)
            st.caption("News covers recent items only and is not used to train the price model.")
        else:
            st.info("No recent headlines found.")
    except Exception as e:  # noqa: BLE001
        st.warning(f"News unavailable: {e}")

with tab_fc:
    probs = predict_tomorrow(data, latest_features(prices))
    for name, p in probs.items():
        st.metric(f"{name}: probability next close is higher", f"{p * 100:.1f}%")
    st.caption("Daily direction is close to a coin flip for most stocks. Treat this as a demo, not a signal.")
