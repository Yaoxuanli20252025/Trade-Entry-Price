import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

TICKER    = "COIN"
THRESHOLD = 0.015  # min % price reversal to count as a new swing

# ---------- Data ----------
df = yf.download(TICKER, period="1d", interval="5m", auto_adjust=True, progress=False)
if hasattr(df.columns, "levels"):
    df.columns = df.columns.get_level_values(0)
df.dropna(inplace=True)

high  = df["High"].squeeze().values
low   = df["Low"].squeeze().values
close = df["Close"].squeeze().values
times = df.index.strftime("%H:%M")
x     = np.arange(len(close))

# ---------- ZigZag ----------
# Walks through bars; flips direction when price reverses >= THRESHOLD from the last extreme.
# Returns the index and type (+1 peak / -1 trough) of each pivot.
def zigzag(high, low, threshold):
    idx, kind = [0], [1]          # start at bar 0, assume peak
    last = high[0]
    up   = True                   # currently looking for higher high

    for i in range(1, len(high)):
        if up:
            if high[i] > last:                              # new higher high → update pivot
                idx[-1], last = i, high[i]
            elif (last - low[i]) / last >= threshold:       # reversed down → new trough
                up = False;  idx.append(i);  kind.append(-1);  last = low[i]
        else:
            if low[i] < last:                               # new lower low → update pivot
                idx[-1], last = i, low[i]
            elif (high[i] - last) / last >= threshold:      # reversed up → new peak
                up = True;   idx.append(i);  kind.append(1);   last = high[i]

    return np.array(idx), np.array(kind)

piv_idx, piv_kind = zigzag(high, low, THRESHOLD)
peaks   = piv_idx[piv_kind ==  1]
troughs = piv_idx[piv_kind == -1]
zz_y    = np.where(piv_kind == 1, high[piv_idx], low[piv_idx])  # y-values for the zz line

# ---------- Plot ----------
BG = "#0d1117"
fig, ax = plt.subplots(figsize=(17, 6))
fig.patch.set_facecolor(BG);  ax.set_facecolor(BG)

ax.plot(x, close, color="#8888aa", linewidth=0.8, label="Close Price")
ax.plot(piv_idx, zz_y, color="#f0b429", linewidth=2.0, label=f"ZigZag ({THRESHOLD*100:.1f}%)")

# Red dots + labels for peaks
ax.scatter(peaks, high[peaks], color="#ff4d4d", s=70, zorder=5, label="Peak (H)")
for i in peaks:
    ax.annotate(f"{high[i]:.0f}", (i, high[i]), xytext=(0, 7),
                textcoords="offset points", ha="center",
                fontsize=7, color="#ff4d4d", fontweight="bold")

# Green dots + labels for troughs
ax.scatter(troughs, low[troughs], color="#00e676", s=70, zorder=5, label="Trough (L)")
for i in troughs:
    ax.annotate(f"{low[i]:.0f}", (i, low[i]), xytext=(0, -11),
                textcoords="offset points", ha="center",
                fontsize=7, color="#00e676", fontweight="bold")

# Axes & styling
tick_step = max(1, len(times) // 8)
date_str  = df.index[0].strftime("%d %b %Y")
ax.set_title(f"ZigZag Technical Analysis — {date_str} (5-min) — {TICKER}",
             fontsize=12, color="white")
ax.set_xlabel("Time", color="white");  ax.set_ylabel("Price", color="white")
ax.set_xticks(x[::tick_step]);  ax.set_xticklabels(times[::tick_step], color="white")
ax.tick_params(colors="white")
ax.grid(True, alpha=0.15, linestyle="--", color="white")
for spine in ax.spines.values(): spine.set_edgecolor("#333344")
ax.legend(loc="upper right", facecolor="#1a1a2e", edgecolor="#333344",
          labelcolor="white", fontsize=9)

plt.tight_layout()
plt.savefig(f"{TICKER}_zigzag.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
