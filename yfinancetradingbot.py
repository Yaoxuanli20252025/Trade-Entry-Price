import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

TICKER = "AAPL"
CASH   = 10_000

# ── Fetch ────────────────────────────────────────────────────────────────────
df = yf.download(TICKER, period="60d", interval="1d", auto_adjust=True, progress=False)
if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, "levels"):
    df.columns = df.columns.get_level_values(0)
df.dropna(inplace=True)

# ── Indicators ───────────────────────────────────────────────────────────────
close        = df["Close"].squeeze()
df["SMA_20"] = close.rolling(20).mean()
df["SMA_50"] = close.rolling(50).mean()

delta        = close.diff()
gain         = delta.clip(lower=0).rolling(14).mean()
loss         = (-delta.clip(upper=0)).rolling(14).mean()
df["RSI"]    = 100 - 100 / (1 + gain / loss)

ema12        = close.ewm(span=12, adjust=False).mean()
ema26        = close.ewm(span=26, adjust=False).mean()
df["MACD"]   = ema12 - ema26
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# ── Signals ──────────────────────────────────────────────────────────────────
df["Action"] = "HOLD"
buy  = (df["SMA_20"] > df["SMA_50"]) & (df["RSI"] < 40) & (df["MACD"] > df["Signal"])
sell = (df["SMA_20"] < df["SMA_50"]) & (df["RSI"] > 60) & (df["MACD"] < df["Signal"])
df.loc[buy,  "Action"] = "BUY"
df.loc[sell, "Action"] = "SELL"

# ── Backtest ─────────────────────────────────────────────────────────────────
cash, shares, portfolio = CASH, 0.0, []
for _, row in df.iterrows():
    price = float(row["Close"])
    if row["Action"] == "BUY" and cash >= price:
        shares += cash // price
        cash   -= (cash // price) * price
    elif row["Action"] == "SELL" and shares > 0:
        cash  += shares * price
        shares = 0
    portfolio.append(cash + shares * price)

df["Portfolio"] = portfolio
ret    = (portfolio[-1] - CASH) / CASH * 100
bh_ret = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100

print(f"Bot Return   : {ret:+.2f}%")
print(f"Buy & Hold   : {bh_ret:+.2f}%")
print(f"Final Value  : ${portfolio[-1]:,.2f}")
print(f"Last Signal  : {df['Action'].iloc[-1]} @ ${float(close.iloc[-1]):.2f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(f"{TICKER} — AI Trading Bot", fontsize=14, fontweight="bold")

ax1.plot(close,        label="Close",  color="steelblue")
ax1.plot(df["SMA_20"], label="SMA 20", color="orange", linestyle="--")
ax1.plot(df["SMA_50"], label="SMA 50", color="red",    linestyle="--")
ax1.scatter(df[df["Action"]=="BUY"].index,  df[df["Action"]=="BUY"]["Close"],  marker="^", color="green", s=100, zorder=5, label="BUY")
ax1.scatter(df[df["Action"]=="SELL"].index, df[df["Action"]=="SELL"]["Close"], marker="v", color="red",   s=100, zorder=5, label="SELL")
ax1.set_ylabel("Price ($)")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(df["Portfolio"], label="Bot",        color="green")
ax2.plot(CASH * close / float(close.iloc[0]), label="Buy & Hold", color="steelblue", linestyle=":")
ax2.set_ylabel("Portfolio ($)")
ax2.set_xlabel("Date")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{TICKER}_bot.png", dpi=150, bbox_inches="tight")
plt.show()