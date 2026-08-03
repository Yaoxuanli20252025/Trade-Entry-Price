import warnings, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, yfinance as yf
warnings.filterwarnings("ignore")

# ── CONFIG — edit here ────────────────────────────────────────────────────────
TICKER    = "INTC"       # any Yahoo Finance symbol: AAPL  TSLA  BTC-USD  ETH-USD  SPY ...
PERIOD    = "1mo"        # 1d  5d  1mo  3mo  6mo  1y
INTERVAL  = "5m"         # 1m  5m  15m  30m  1h  1d
THRESHOLD = 0.015        # ZigZag reversal sensitivity  (0.015 = 1.5%)
METHOD    = "all"        # neuro | zigzag | geo | all

# ── THEME ─────────────────────────────────────────────────────────────────────
BG = "#0d1117"
C  = dict(close="#8888aa", ema9="#f0b429", ema21="#00e5ff", ema50="#ff6b6b",
          ema200="#a0a0ff", peak="#ff4d4d", trough="#00e676",
          rsi="#00e5ff",  vol_up="#2ecc71", vol_dn="#e74c3c", pat="#ff9f43")
LK = dict(facecolor="#1a1a2e", edgecolor="#333344", labelcolor="white", fontsize=8)

def style(ax):
    ax.set_facecolor(BG); ax.tick_params(colors="white")
    ax.grid(alpha=0.12, ls="--", color="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333344")
    for l in [ax.xaxis.label, ax.yaxis.label, ax.title]: l.set_color("white")

def finish(fig, axes, x, ts, path):
    step = max(1, len(x) // 8)
    axes[-1].set_xticks(x[::step]); axes[-1].set_xticklabels(ts[::step], color="white", fontsize=7)
    for ax in axes: style(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  saved → {path}"); plt.show()

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
            if H[i] > last:               idx[-1], last = i, H[i]
            elif (last-L[i])/last >= thr: up=False; idx.append(i); kind.append(-1); last=L[i]
        else:
            if L[i] < last:               idx[-1], last = i, L[i]
            elif (H[i]-last)/last >= thr: up=True;  idx.append(i); kind.append(1);  last=H[i]
    return np.array(idx), np.array(kind)

def find_patterns(pi, pk, H, L):
    P, T = pi[pk==1], pi[pk==-1]; pH, tL = H[P], L[T]; out=[]
    for i in range(len(P)-1):
        if abs(pH[i]-pH[i+1])/pH[i] < .005: out.append(("Double Top",    P[i],P[i+1],(pH[i]+pH[i+1])/2))
    for i in range(len(T)-1):
        if abs(tL[i]-tL[i+1])/tL[i] < .005: out.append(("Double Bottom", T[i],T[i+1],(tL[i]+tL[i+1])/2))
    if len(P)>=2 and len(T)>=2:
        if   pH[-1]>pH[-2] and tL[-1]>tL[-2]: out.append(("Uptrend",   P[-2],P[-1],None))
        elif pH[-1]<pH[-2] and tL[-1]<tL[-2]: out.append(("Downtrend", T[-2],T[-1],None))
    return out

# ── METHOD 1 — NEUROTRADER888 ─────────────────────────────────────────────────
def run_neuro(df, ticker):
    close, vol = df["Close"].squeeze(), df["Volume"].squeeze()
    x, ts = np.arange(len(df)), df.index.strftime("%H:%M")
    fig = plt.figure(figsize=(17,9), facecolor=BG)
    ax1,ax2,ax3 = [fig.add_subplot(gridspec.GridSpec(3,1,height_ratios=[5,2,2],hspace=.08)[i]) for i in range(3)]

    ax1.plot(x, close.values, color=C["close"], lw=.8, label="Close")
    for n,col in [(9,C["ema9"]),(21,C["ema21"]),(50,C["ema50"]),(200,C["ema200"])]:
        ax1.plot(x, ema(close,n).values, color=col, lw=1.2, label=f"EMA {n}")
    ax1.set_title(f"Neurotrader888 — {ticker}", fontsize=12); ax1.legend(**LK, ncol=5)

    ax2.bar(x, vol.values, width=.8, alpha=.75,
            color=[C["vol_up"] if close.values[i]>=close.values[i-1] else C["vol_dn"] for i in range(len(close))])
    ax2.set_ylabel("Volume", fontsize=8)

    r = rsi(close).values; ax3.plot(x, r, color=C["rsi"], lw=1.2)
    for lvl,col in [(70,C["peak"]),(30,C["trough"])]:
        ax3.axhline(lvl, color=col, lw=.8, ls="--", alpha=.7)
        ax3.fill_between(x, r, lvl, where=(r>=lvl if lvl==70 else r<=lvl), alpha=.2, color=col)
    ax3.set_ylim(0,100); ax3.set_ylabel("RSI 14", fontsize=8)
    finish(fig, [ax1,ax2,ax3], x, ts, f"{ticker}_neuro.png")

# ── METHOD 2 — ZIGZAG ────────────────────────────────────────────────────────
def run_zigzag(df, ticker, thr):
    H,L,Cl = df["High"].squeeze().values, df["Low"].squeeze().values, df["Close"].squeeze().values
    x, ts  = np.arange(len(df)), df.index.strftime("%H:%M")
    pi,pk  = zigzag(H,L,thr); P,T = pi[pk==1], pi[pk==-1]

    fig,ax = plt.subplots(figsize=(17,6), facecolor=BG)
    ax.plot(x, Cl, color=C["close"], lw=.8, label="Close")
    ax.plot(pi, np.where(pk==1,H[pi],L[pi]), color=C["ema9"], lw=2, label=f"ZigZag {thr*100:.1f}%")
    ax.scatter(P,H[P], color=C["peak"],   s=70, zorder=5, label="Peak")
    ax.scatter(T,L[T], color=C["trough"], s=70, zorder=5, label="Trough")
    for i,y,col,dy in [(p,H[p],C["peak"],7) for p in P]+[(t,L[t],C["trough"],-11) for t in T]:
        ax.annotate(f"{y:.0f}",(i,y),xytext=(0,dy),textcoords="offset points",
                    ha="center",fontsize=7,color=col,fontweight="bold")
    ax.set_title(f"ZigZag — {ticker}", fontsize=12); ax.legend(**LK)
    finish(fig, [ax], x, ts, f"{ticker}_zigzag.png")

# ── METHOD 3 — GEOMETRIC FORMATIONS ──────────────────────────────────────────
def run_geo(df, ticker, thr):
    H,L,Cl = df["High"].squeeze().values, df["Low"].squeeze().values, df["Close"].squeeze().values
    x, ts  = np.arange(len(df)), df.index.strftime("%H:%M")
    pi,pk  = zigzag(H,L,thr); P,T = pi[pk==1], pi[pk==-1]
    pats   = find_patterns(pi,pk,H,L)

    fig,ax = plt.subplots(figsize=(17,7), facecolor=BG)
    ax.plot(x, Cl, color=C["close"], lw=.8, label="Close")
    ax.plot(pi, np.where(pk==1,H[pi],L[pi]), color=C["ema9"], lw=1.4, alpha=.6, label="ZigZag")
    ax.scatter(P,H[P],color=C["peak"],  s=60,zorder=5)
    ax.scatter(T,L[T],color=C["trough"],s=60,zorder=5)
    for pts,vals,col,lbl in [(P,H[P],C["peak"],"Resistance"),(T,L[T],C["trough"],"Support")]:
        if len(pts)>=2:
            m,b=np.polyfit(pts,vals,1); ax.plot(x,m*x+b,color=col,lw=1,ls="--",alpha=.45,label=lbl)
    print(f"\n[Geo] {ticker} patterns:")
    for name,x1,x2,price in pats:
        if price:
            ax.hlines(price,x1,x2,colors=C["pat"],lw=1.5,ls="dotted")
            ax.annotate(name,((x1+x2)/2,price),xytext=(0,10),textcoords="offset points",
                        ha="center",fontsize=8,color=C["pat"],fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2",fc="#1a1a2e",ec=C["pat"],alpha=.85))
        print(f"  {name}  bars {x1}–{x2}" + (f"  @{price:.2f}" if price else ""))
    if not pats: print("  no patterns detected")
    ax.set_title(f"Geo Formations — {ticker}", fontsize=12); ax.legend(**LK)
    finish(fig, [ax], x, ts, f"{ticker}_geo.png")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def run(ticker=TICKER, period=PERIOD, interval=INTERVAL, threshold=THRESHOLD, method=METHOD):
    print(f"\n=== {ticker} | {period} {interval} | {method} ===")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if hasattr(df.columns, "levels"): df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    print(f"  {len(df)} bars loaded")
    if method in ("neuro",  "all"): run_neuro(df, ticker)
    if method in ("zigzag", "all"): run_zigzag(df, ticker, threshold)
    if method in ("geo",    "all"): run_geo(df, ticker, threshold)

if __name__ == "__main__":
    run()
