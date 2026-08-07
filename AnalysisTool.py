import warnings, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, yfinance as yf
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKER    = "INTC"    # any symbol: AAPL  META  BTC-USD  SPY ...
PERIOD    = "1mo"     # 1d  5d  1mo  3mo  6mo  1y
INTERVAL  = "5m"      # 1m  5m  15m  30m  1h  1d
THRESHOLD = 0.015     # ZigZag sensitivity (0.015 = 1.5%)

# ── THEME ─────────────────────────────────────────────────────────────────────
BG = "#0d1117"
C  = dict(close="#8888aa", ema9="#f0b429", ema21="#00e5ff", ema50="#ff6b6b",
          ema200="#a0a0ff", peak="#ff4d4d", trough="#00e676",
          rsi="#00e5ff", vol_up="#2ecc71", vol_dn="#e74c3c",
          breakout="#00ff88", breakdown="#ff4444")
LK = dict(facecolor="#1a1a2e", edgecolor="#333344", labelcolor="white", fontsize=8)

def style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors="white")
    ax.grid(alpha=0.12, ls="--", color="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333344")
    for l in [ax.xaxis.label, ax.yaxis.label, ax.title]: l.set_color("white")

def finish(fig, axes, x, ts, path):
    step = max(1, len(x) // 8)
    axes[-1].set_xticks(x[::step])
    axes[-1].set_xticklabels(ts[::step], color="white", fontsize=7, rotation=20, ha="right")
    for ax in axes: style(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  saved → {path}"); plt.show()

def timestamps(df):
    idx = df.index
    fmt = "%m-%d %H:%M" if idx.normalize().nunique() > 1 else "%H:%M"
    return idx.strftime(fmt)

# ── INDICATORS ────────────────────────────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s):
    d = s.diff(); g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def zigzag(H, L, thr):
    idx, kind, last, up = [0], [1], H[0], True
    for i in range(1, len(H)):
        if up:
            if H[i] > last:                idx[-1], last = i, H[i]
            elif (last - L[i]) / last >= thr: up = False; idx.append(i); kind.append(-1); last = L[i]
        else:
            if L[i] < last:                idx[-1], last = i, L[i]
            elif (H[i] - last) / last >= thr: up = True;  idx.append(i); kind.append(1);  last = H[i]
    return np.array(idx), np.array(kind)

def find_breaks(Cl, H, L, pi, pk):
    """
    Breakout  (^): close breaks above the last swing high (peak).
    Breakdown (v): close breaks below the last swing low  (trough).
    Each level is consumed once a break is confirmed.
    """
    P_set, T_set = set(pi[pk == 1]), set(pi[pk == -1])
    breakouts, breakdowns = [], []
    last_peak = last_trough = None
    for i in range(len(Cl)):
        if i in P_set:   last_peak   = H[i]
        if i in T_set:   last_trough = L[i]
        if last_peak   is not None and Cl[i] > last_peak:   breakouts.append(i);  last_peak   = None
        if last_trough is not None and Cl[i] < last_trough: breakdowns.append(i); last_trough = None
    return breakouts, breakdowns

# ── CHART 1: EMA + Volume + RSI ───────────────────────────────────────────────
def run_neuro(df, ticker):
    close, vol = df["Close"].squeeze(), df["Volume"].squeeze()
    x, ts = np.arange(len(df)), timestamps(df)
    fig = plt.figure(figsize=(17, 9), facecolor=BG)
    ax1, ax2, ax3 = [fig.add_subplot(gridspec.GridSpec(3, 1, height_ratios=[5, 2, 2], hspace=.08)[i]) for i in range(3)]

    ax1.plot(x, close.values, color=C["close"], lw=.8, label="Close")
    for n, col in [(9, C["ema9"]), (21, C["ema21"]), (50, C["ema50"]), (200, C["ema200"])]:
        ax1.plot(x, ema(close, n).values, color=col, lw=1.2, label=f"EMA {n}")
    ax1.set_title(f"EMA Overview — {ticker}", fontsize=12); ax1.legend(**LK, ncol=5)

    vol_colors = [C["vol_up"] if i == 0 or close.values[i] >= close.values[i-1] else C["vol_dn"] for i in range(len(close))]
    ax2.bar(x, vol.values, width=.8, alpha=.75, color=vol_colors)
    ax2.set_ylabel("Volume", fontsize=8)

    r = rsi(close).values; ax3.plot(x, r, color=C["rsi"], lw=1.2)
    for lvl, col in [(70, C["peak"]), (30, C["trough"])]:
        ax3.axhline(lvl, color=col, lw=.8, ls="--", alpha=.7)
        ax3.fill_between(x, r, lvl, where=(r >= lvl if lvl == 70 else r <= lvl), alpha=.2, color=col)
    ax3.set_ylim(0, 100); ax3.set_ylabel("RSI 14", fontsize=8)
    finish(fig, [ax1, ax2, ax3], x, ts, f"{ticker}_neuro.png")

# ── CHART 2: ZigZag + Breakout / Breakdown ────────────────────────────────────
def run_zigzag(df, ticker, thr):
    H, L, Cl = df["High"].squeeze().values, df["Low"].squeeze().values, df["Close"].squeeze().values
    x, ts = np.arange(len(df)), timestamps(df)
    pi, pk = zigzag(H, L, thr)
    P, T   = pi[pk == 1], pi[pk == -1]
    bouts, bdowns = find_breaks(Cl, H, L, pi, pk)

    fig, ax = plt.subplots(figsize=(17, 7), facecolor=BG)
    ax.plot(x, Cl, color=C["close"], lw=.8, label="Close")
    ax.plot(pi, np.where(pk == 1, H[pi], L[pi]), color=C["ema9"], lw=2, label=f"ZigZag {thr*100:.1f}%")

    ax.scatter(P, H[P], color=C["peak"],   s=70, zorder=5, label="Peak (Resistance)")
    ax.scatter(T, L[T], color=C["trough"], s=70, zorder=5, label="Trough (Support)")
    for p in P: ax.annotate(f"{H[p]:.0f}", (p, H[p]), xytext=(0, 7),  textcoords="offset points", ha="center", fontsize=7, color=C["peak"],   fontweight="bold")
    for t in T: ax.annotate(f"{L[t]:.0f}", (t, L[t]), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=7, color=C["trough"], fontweight="bold")

    if bouts:
        ax.scatter(bouts, Cl[bouts], marker="^", color=C["breakout"],  s=150, zorder=6, label=f"Breakout  ({len(bouts)})")
        for i in bouts: ax.annotate("▲ OUT", (i, Cl[i]), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=7, color=C["breakout"], fontweight="bold")
    if bdowns:
        ax.scatter(bdowns, Cl[bdowns], marker="v", color=C["breakdown"], s=150, zorder=6, label=f"Breakdown ({len(bdowns)})")
        for i in bdowns: ax.annotate("▼ DOWN", (i, Cl[i]), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7, color=C["breakdown"], fontweight="bold")

    ax.set_title(f"ZigZag — Breakout & Breakdown — {ticker}", fontsize=12)
    ax.legend(**LK)
    finish(fig, [ax], x, ts, f"{ticker}_zigzag.png")

    print(f"\n[Breakouts]  bars: {bouts}")
    print(f"[Breakdowns] bars: {bdowns}")

# ── CHART 3: Geometric Formations ────────────────────────────────────────────
def find_patterns(pi, pk, H, L, min_gap):
    P, T = pi[pk == 1], pi[pk == -1]
    pH, tL = H[P], L[T]
    out = []
    last_x = -np.inf
    for i in range(len(P) - 1):
        if abs(pH[i] - pH[i+1]) / pH[i] < .003 and (P[i+1] - P[i]) >= min_gap:
            cx = (P[i] + P[i+1]) / 2
            if cx - last_x >= min_gap:
                out.append(("Double Top", P[i], P[i+1], (pH[i] + pH[i+1]) / 2)); last_x = cx
    last_x = -np.inf
    for i in range(len(T) - 1):
        if abs(tL[i] - tL[i+1]) / tL[i] < .003 and (T[i+1] - T[i]) >= min_gap:
            cx = (T[i] + T[i+1]) / 2
            if cx - last_x >= min_gap:
                out.append(("Double Bottom", T[i], T[i+1], (tL[i] + tL[i+1]) / 2)); last_x = cx
    if len(P) >= 2 and len(T) >= 2:
        if   pH[-1] > pH[-2] and tL[-1] > tL[-2]: out.append(("Uptrend",   P[-2], P[-1], None))
        elif pH[-1] < pH[-2] and tL[-1] < tL[-2]: out.append(("Downtrend", T[-2], T[-1], None))
    return sorted(out, key=lambda p: p[1])

def run_geo(df, ticker, thr):
    H, L, Cl = df["High"].squeeze().values, df["Low"].squeeze().values, df["Close"].squeeze().values
    x, ts = np.arange(len(df)), timestamps(df)
    min_gap = max(3, int(len(x) * 0.03))
    pi, pk = zigzag(H, L, thr)
    P, T   = pi[pk == 1], pi[pk == -1]
    pats   = find_patterns(pi, pk, H, L, min_gap)

    fig, ax = plt.subplots(figsize=(17, 7), facecolor=BG)
    ax.plot(x, Cl, color=C["close"], lw=.8, label="Close")
    ax.plot(pi, np.where(pk == 1, H[pi], L[pi]), color=C["ema9"], lw=1.4, alpha=.6, label="ZigZag")
    ax.scatter(P, H[P], color=C["peak"],   s=60, zorder=5)
    ax.scatter(T, L[T], color=C["trough"], s=60, zorder=5)

    for pts, vals, col, lbl in [(P, H[P], C["peak"], "Resistance"), (T, L[T], C["trough"], "Support")]:
        if len(pts) >= 2:
            m, b = np.polyfit(pts, vals, 1)
            ax.plot(x, m * x + b, color=col, lw=1, ls="--", alpha=.45, label=lbl)

    print(f"\n[Geo] {ticker} patterns:")
    pat_color = "#ff9f43"
    last_lbl_x, toggle = -np.inf, 0
    for name, x1, x2, price in pats:
        if price:
            ax.hlines(price, x1, x2, colors=pat_color, lw=1.5, ls="dotted")
            cx = (x1 + x2) / 2
            toggle = toggle + 1 if cx - last_lbl_x < min_gap else 0
            dy = 10 + (toggle % 3) * 16
            last_lbl_x = cx
            ax.annotate(name, (cx, price), xytext=(0, dy), textcoords="offset points",
                        ha="center", fontsize=8, color=pat_color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec=pat_color, alpha=.85),
                        arrowprops=dict(arrowstyle="-", color=pat_color, alpha=.5, lw=.7))
        print(f"  {name}  bars {x1}–{x2}" + (f"  @{price:.2f}" if price else ""))
    if not pats: print("  no patterns detected")

    ax.set_title(f"Geo Formations — {ticker}", fontsize=12); ax.legend(**LK)
    finish(fig, [ax], x, ts, f"{ticker}_geo.png")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def run(ticker=TICKER, period=PERIOD, interval=INTERVAL, threshold=THRESHOLD):
    print(f"\n=== {ticker} | {period} {interval} | threshold={threshold*100:.1f}% ===")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if hasattr(df.columns, "levels"): df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    print(f"  {len(df)} bars loaded")
    run_neuro(df, ticker)
    run_zigzag(df, ticker, threshold)
    run_geo(df, ticker, threshold)

if __name__ == "__main__":
    run()
