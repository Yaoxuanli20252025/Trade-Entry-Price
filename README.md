# Markdow description of the code and project
Impelemntation of ZIgZag, Neurotrader, and geomatric formation
- Neurotrader
  3-panel chart: price with EMA 9/21/50/200, a color-coded volume bar chart and a 14-period RSI panel
- ZIgZag
  Plots the ZigZag reversal line over price, marks resistance and support line, and flags breakouts and breakdowns
- Geo Formations
  Fits linear resistance/support trendlines to the swings and flags Double Top and Double Bottom patterns, plus overall Uptrend/Downtrend. Detected patterns are also printed to the console.
# How to install
- Python 3.8+
- matplotlib
  Chating
- Numpy
  Numerical operations (EMA, RSI, ZigZag, pattern math)
- yfinance
  Downloading OHLCV price data from Yahoo Finance
  
## How to run
Edit the config block at the top of the script, then run.

# Inputs
- Set these in the CONFIG block, or pass them as arguments to run():
- TICKER - Any stock symbol	AAPL, META, BTC-USD, SPY, INTC
- PERIOD - How far back to fetch data	(1d, 3mo, 1y)
- INTERVAL - Bar/candle interval (1m, 30m, 1h)
- THRESHOLD - ZigZag reversal sensitivity
# Output
- For a given TICKER, each run produces:
- neuro.png — EMA / Volume / RSI panel chart
- zigzag.png — ZigZag chart with breakout/breakdown markers
- geo.png — Support/resistance trendlines + pattern chart
Examples of running

