import requests
import matplotlib.pyplot as plt
from typing import List, Tuple

# Global endpoint template for World Bank API
URL_DATA = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=30"


def fetch(indicator: str, country: str = "US") -> List[Tuple[int, float]]:
    """Fetch recent values for an indicator from the World Bank API.

    Args:
        indicator: World Bank indicator code, e.g. 'NY.GDP.MKTP.CD'
        country:  Country ISO code (default 'US').

    Returns:
        Sorted list of (year, value) pairs (oldest first).

    Raises:
        RuntimeError: if the API response is invalid or no data is returned.
    """
    url = URL_DATA.format(country=country, indicator=indicator)
    resp = requests.get(url)
    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(f"Invalid JSON response from {url}")

    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected response structure from {url}")

    data = payload[1]
    pairs: List[Tuple[int, float]] = []
    for d in data:
        if d.get("value") is None:
            continue
        try:
            year = int(d["date"])
            val = float(d["value"])
        except (TypeError, ValueError):
            continue
        pairs.append((year, val))

    if not pairs:
        raise RuntimeError(f"No data returned for {indicator} / {country}")

    return sorted(pairs)


def signals(pairs: List[Tuple[int, float]]) -> Tuple[List[int], List[float], List[str], float]:
    """Compute simple BUY/SELL signals based on whether a value is above the series average.

    Args:
        pairs: list of (year, value) tuples, oldest first.

    Returns:
        years, values, signals, average
        - years: list of years
        - values: list of values
        - signals: list of 'BUY' or 'SELL' strings for each value
        - average: average of the values
    """
    years = [y for y, v in pairs]
    vals = [v for y, v in pairs]
    avg = sum(vals) / len(vals)
    sigs = ["BUY" if v > avg else "SELL" for v in vals]
    return years, vals, sigs, avg


def plot_indicator(pairs: List[Tuple[int, float]], title: str, ylabel: str) -> None:
    """Plot a single indicator with annotated BUY/SELL signals.

    Args:
        pairs: list of (year, value) tuples
        title: plot title
        ylabel: y axis label
    """
    years, vals, sigs, avg = signals(pairs)

    plt.figure(figsize=(10, 4))
    plt.plot(years, vals, marker="o", color="steelblue", label=ylabel)
    for y, v, s in zip(years, vals, sigs):
        color = "green" if s == "BUY" else "red"
        plt.annotate(s, (y, v), textcoords="offset points", xytext=(0, 8),
                     fontsize=7, color=color, ha="center")

    plt.axhline(avg, color="gray", linestyle="--", label=f"Avg: {avg:.2f}")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # GDP (current USD)
    gdp = fetch("NY.GDP.MKTP.CD")
    plot_indicator(gdp, "US GDP (current USD)", "USD")

    # Inflation (CPI %)
    cpi = fetch("FP.CPI.TOTL.ZG")
    plot_indicator(cpi, "US Inflation (%)", "%")

    # Unemployment (%)
    uem = fetch("SL.UEM.TOTL.ZS")
    plot_indicator(uem, "US Unemployment (%)", "%")
