import requests
import matplotlib.pyplot as plt

# --- Fetch World Bank Data ---
def fetch(indicator, country="US"):
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=30"
    data = requests.get(url).json()[1]
    pairs = [(int(d["date"]), d["value"]) for d in data if d["value"]]
    return sorted(pairs)  # oldest first

# --- Signal: value above average = BUY, below = SELL ---
def signals(pairs):
    years, vals = zip(*pairs)
    avg = sum(vals) / len(vals)
    sigs = ["BUY" if v > avg else "SELL" for v in vals]
    return years, vals, sigs, avg

# --- Plot one indicator ---
def plot(pairs, title, ylabel):
    years, vals, sigs, avg = signals(pairs)
    plt.figure(figsize=(10, 4))
    plt.plot(years, vals, marker="o", color="steelblue", label=ylabel)
    for y, v, s in zip(years, vals, sigs):
        color = "green" if s == "BUY" else "red"
        plt.annotate(s, (y, v), textcoords="offset points", xytext=(0, 8),
                     fontsize=7, color=color)
    plt.axhline(avg, color="gray", linestyle="--", label=f"Avg: {avg:.2f}")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

# GDP (current USD)
plot(fetch("NY.GDP.MKTP.CD"), "US GDP (current USD)", "USD")

# Inflation (CPI %)
plot(fetch("FP.CPI.TOTL.ZG"), "US Inflation (%)", "%")

# Unemployment (%)
plot(fetch("SL.UEM.TOTL.ZS"), "US Unemployment (%)", "%")
