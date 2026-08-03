import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

TICKER = "INTC"

# ── Fetch intraday data ──────────────────────────────────────────────────────
df = yf.download(TICKER, period="30d", interval="5m", auto_adjust=True, progress=False)
if hasattr(df.columns, "levels"):
    df.columns = df.columns.get_level_values(0)
df.dropna(inplace=True)

close = df["Close"].squeeze().values
high  = df["High"].squeeze().values
low   = df["Low"].squeeze().values
x     = np.arange(len(close))

# ── Swing highs / lows (window = 5) ─────────────────────────────────────────
def swing_points(prices, mode="high", window=5):
    idx = []
    for i in range(window, len(prices) - window):
        seg = prices[i - window: i + window + 1]
        if mode == "high" and prices[i] == seg.max():
            idx.append(i)
        elif mode == "low" and prices[i] == seg.min():
            idx.append(i)
    return np.array(idx)

hi_idx = swing_points(high, "high")
lo_idx = swing_points(low,  "low")

# ── Linear trendlines through swing points ───────────────────────────────────
def trendline(x_pts, y_pts, x_full):
    m, b = np.polyfit(x_pts, y_pts, 1)
    return m * x_full + b

resist = trendline(hi_idx, high[hi_idx], x)
support = trendline(lo_idx, low[lo_idx],  x)

# ── Geometric channel fill ───────────────────────────────────────────────────
mid = (resist + support) / 2

# ── Plot (neurotrader888 style) ──────────────────────────────────────────────
times = df.index.strftime("%H:%M")
tick_step = max(1, len(times) // 8)

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.fill_between(x, support, resist, alpha=0.06, color="steelblue")  # channel
ax.plot(x, close,   color="orange", linewidth=1.2, label="Price")
ax.plot(x, support, color="green",  linewidth=1.8, linestyle="--", label="Support Line")
ax.plot(x, resist,  color="red",    linewidth=1.8, linestyle="--", label="Resistance Line")
ax.plot(x, mid,     color="gray",   linewidth=0.8, linestyle=":",  label="Mid Line", alpha=0.6)

# Mark swing highs / lows
ax.scatter(hi_idx, high[hi_idx], marker="v", color="red",   s=50, zorder=5, alpha=0.7)
ax.scatter(lo_idx, low[lo_idx],  marker="^", color="green", s=50, zorder=5, alpha=0.7)

ax.set_title(f"Support & Resistance Trendlines ({TICKER})", fontsize=13)
ax.set_ylabel("Price")
ax.set_xlabel("Time")
ax.set_xticks(x[::tick_step])
ax.set_xticklabels(times[::tick_step], rotation=0)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(f"{TICKER}_neurotrader.png", dpi=150, bbox_inches="tight")
plt.show()
