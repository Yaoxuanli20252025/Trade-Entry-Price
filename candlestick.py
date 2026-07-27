"""
Candlestick chart with Swing High/Low points and Trendlines.

Usage:
  python3 candlestick.py --source file
  python3 candlestick.py --source api --ticker AAPL --period 3mo --interval 1d
"""

import argparse
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--source",   choices=["file", "api"], default="file")
parser.add_argument("--file",     default="data.csv")
parser.add_argument("--ticker",   default="AAPL")
parser.add_argument("--period",   default="3mo")
parser.add_argument("--interval", default="1d")
parser.add_argument("--window",   type=int, default=5,
                    help="Bars each side to confirm a swing point")
args = parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────
def load_file(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], dayfirst=True)
    df = df.rename(columns={"timestamp": "Date", "open": "Open",
                             "high": "High", "low": "Low", "close": "Close"})
    return df.sort_values("Date").reset_index(drop=True)


def load_api(ticker, period, interval):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Run: pip3 install yfinance")
    print(f"Fetching {ticker} ({period}, {interval})…")
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        sys.exit(f"No data for '{ticker}'")
    df = df.reset_index().rename(columns={"Datetime": "Date", "index": "Date"})
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Date", "Open", "High", "Low", "Close"]]


# ── Swing detection ───────────────────────────────────────────────────────────
def find_swings(df, window=5):
    """Return indices of swing highs and swing lows."""
    hi, lo = df["High"].values, df["Low"].values
    n = len(df)
    swing_highs, swing_lows = [], []
    for i in range(window, n - window):
        if hi[i] == max(hi[i - window: i + window + 1]):
            swing_highs.append(i)
        if lo[i] == min(lo[i - window: i + window + 1]):
            swing_lows.append(i)
    return swing_highs, swing_lows


# ── Trendline detection ───────────────────────────────────────────────────────
def find_trendlines(df, swing_idx, col, direction, window=5, top_n=2):
    """
    Find valid trendlines through swing points.
      direction='up'   → support line through ascending swing lows
      direction='down' → resistance line through descending swing highs
    Returns list of dicts: {x_start, x_end, y_start, y_end, score}
    """
    prices = df[col].values
    dates  = df["Date"].values
    price_range = prices.max() - prices.min()
    tol = price_range * 0.003          # 0.3% tolerance for "touching"

    candidates = []
    for a in range(len(swing_idx) - 1):
        for b in range(a + 1, len(swing_idx)):
            i, j = swing_idx[a], swing_idx[b]
            y1, y2 = prices[i], prices[j]
            slope = (y2 - y1) / (j - i)

            if direction == "up"   and slope <= 0: continue
            if direction == "down" and slope >= 0: continue

            # Validate no price breaks through the line between anchors
            valid, touches = True, 0
            for k in range(i, j + 1):
                line_val = y1 + slope * (k - i)
                if direction == "up"   and prices[k] < line_val - tol:
                    valid = False; break
                if direction == "down" and prices[k] > line_val + tol:
                    valid = False; break
                if abs(prices[k] - line_val) <= tol:
                    touches += 1

            if not valid:
                continue

            # Extend line to the last bar
            y_end = y1 + slope * (len(prices) - 1 - i)
            score = touches + (j - i) / len(prices) * 20
            candidates.append(dict(
                x_start=dates[i], x_end=dates[-1],
                y_start=y1,       y_end=y_end,
                score=score,
            ))

    # Deduplicate close anchors, keep top_n
    candidates.sort(key=lambda c: -c["score"])
    selected = []
    for c in candidates:
        if not any(abs(c["x_start"] - s["x_start"]).astype("timedelta64[m]").astype(int) < window * 2
                   for s in selected):
            selected.append(c)
        if len(selected) >= top_n:
            break
    return selected


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_graph(df, title, swing_highs, swing_lows, up_lines, dn_lines):
    graph = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
        subplot_titles=(title, "Price Change"),
    )

    # Candlesticks
    graph.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Swing highs
    sh = df.iloc[swing_highs]
    graph.add_trace(go.Scatter(
        x=sh["Date"], y=sh["High"],
        mode="markers", name="Swing High",
        marker=dict(symbol="triangle-down", color="#ef5350", size=10),
    ), row=1, col=1)

    # Swing lows
    sl = df.iloc[swing_lows]
    graph.add_trace(go.Scatter(
        x=sl["Date"], y=sl["Low"],
        mode="markers", name="Swing Low",
        marker=dict(symbol="triangle-up", color="#26a69a", size=10),
    ), row=1, col=1)

    # Uptrend lines
    for i, line in enumerate(up_lines):
        graph.add_trace(go.Scatter(
            x=[line["x_start"], line["x_end"]],
            y=[line["y_start"], line["y_end"]],
            mode="lines", name=f"Uptrend {i+1}",
            line=dict(color="#26a69a", width=1.5, dash="dash"),
            showlegend=(i == 0),
        ), row=1, col=1)

    # Downtrend lines
    for i, line in enumerate(dn_lines):
        graph.add_trace(go.Scatter(
            x=[line["x_start"], line["x_end"]],
            y=[line["y_start"], line["y_end"]],
            mode="lines", name=f"Downtrend {i+1}",
            line=dict(color="#ef5350", width=1.5, dash="dash"),
            showlegend=(i == 0),
        ), row=1, col=1)

    # Price change bars
    df["change"] = df["Close"].diff()
    graph.add_trace(go.Bar(
        x=df["Date"], y=df["change"].abs(),
        marker_color=["#26a69a" if c >= 0 else "#ef5350" for c in df["change"]],
        name="Price Change", showlegend=False,
    ), row=2, col=1)

    graph.update_layout(
        height=720, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
    )
    graph.update_yaxes(title_text="Price",    row=1, col=1)
    graph.update_yaxes(title_text="|Change|", row=2, col=1)
    return graph


# ── Main ──────────────────────────────────────────────────────────────────────
if args.source == "file":
    df    = load_file(args.file)
    title = f"Candlestick — {args.file}"
else:
    df    = load_api(args.ticker, args.period, args.interval)
    title = f"{args.ticker.upper()}  ({args.period}, {args.interval})"

print(f"Loaded {len(df)} rows | {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")

swing_highs, swing_lows = find_swings(df, window=args.window)
print(f"Swing highs: {len(swing_highs)}  |  Swing lows: {len(swing_lows)}")

up_lines = find_trendlines(df, swing_lows,  "Low",  "up",   window=args.window)
dn_lines = find_trendlines(df, swing_highs, "High", "down", window=args.window)
print(f"Uptrend lines: {len(up_lines)}  |  Downtrend lines: {len(dn_lines)}")

graph = build_graph(df, title, swing_highs, swing_lows, up_lines, dn_lines)
graph.write_html("candlestick.html")
graph.show()
print("Saved → candlestick.html")
