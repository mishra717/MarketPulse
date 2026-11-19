# MarketPulse

Stock trend analysis and next-day direction prediction with Python, Pandas, scikit-learn, yFinance and Plotly (Streamlit UI).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest tests        # optional
```
Indian stocks: use `.NS` tickers (e.g. `RELIANCE.NS`).

## What it does
- **Price and indicators:** candlesticks, SMA, Bollinger Bands, RSI, MACD, annualised volatility.
- **Model evaluation:** walk-forward (train on past, test on future) for Logistic Regression and Random Forest, always compared with naive baselines (majority class, 0% return).
- **Backtest:** long-or-cash strategy vs buy and hold with per-trade costs.
- **News sentiment:** VADER scores on recent headlines (display only, not a training feature).
- **Forecast:** probability that the next close is higher.

## Getting real numbers for your resume
Run the Model evaluation tab on several tickers and note the rows tested, accuracy vs baseline, and RMSE vs the 0% baseline. Report exactly what you see, including when the model does not beat the baseline.

## Design notes
- Features at day t use data up to t only; the target is day t+1's return (tested in `tests/test_core.py`).
- Daily direction is close to random for most stocks. Expect accuracy near 50-55%.

Educational use only, not financial advice.
