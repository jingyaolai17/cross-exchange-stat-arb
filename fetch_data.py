"""
fetch_data_kraken_binance.py
----------------------------
Download synchronized market data from Tardis for
Kraken (spot) and Binance (perpetual futures), suitable for
cross-venue spread / arbitrage analysis.

Design principles:
- Explicit exchange + instrument separation
- Exchange-specific symbol mapping
- Deterministic directory structure
- Safe to reuse for research & production pipelines
"""

import os
import requests
import gzip
import pandas as pd
from datetime import datetime, timedelta

# ==================================================
# CONFIG
# ==================================================

# Exchanges & instruments
EXCHANGES = {
    "binance-futures": {
        "instrument": "perp",
        "symbols": ["BTCUSDT"],
    },
    "kraken-spot": {
        "instrument": "spot",
        "symbols": ["BTC-USD"],
    },
}

# Tardis dataset types needed for spread analysis
DATA_TYPES = [
    "book_snapshot_25",
    "trades",
    "incremental_book_L2",
]

FROM_DATE = "2025-12-15"
TO_DATE   = "2025-01-05"

# Tardis API key
API_KEY = os.getenv(
    "TARDIS_API_KEY",
    "TD.vWJuOhlrKNyeMPNN.nu9ILFfEckIu-IG.Oc6B0o3-T4GWpTd.EUtGu4ThzdSCzvn.bs5r0f0QuXthEeU.bk6y"
)

BASE_URL = "https://datasets.tardis.dev/v1"


# ==================================================
# DOWNLOAD FUNCTION
# ==================================================
def download_tardis_csv(
    exchange: str,
    instrument: str,
    data_type: str,
    symbol: str,
    date: str,
    api_key: str,
) -> str | None:
    """
    Download a single Tardis CSV file and save it under:

    data/{exchange}/{instrument}/{data_type}/{date}_{symbol}.csv.gz
    """
    year, month, day = date.split("-")

    url = (
        f"{BASE_URL}/{exchange}/{data_type}/"
        f"{year}/{month}/{day}/{symbol}.csv.gz"
    )

    headers = {"Authorization": f"Bearer {api_key}"}

    save_dir = f"data/{exchange}/{instrument}/{data_type}"
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, f"{date}_{symbol}.csv.gz")

    print(f"[FETCH] {exchange} | {instrument} | {data_type} | {symbol} | {date}")

    response = requests.get(url, headers=headers, timeout=60)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"  ✓ Saved → {filename}")
        return filename
    else:
        print(
            f"  ✗ Failed ({response.status_code}) → {exchange} {symbol} {date}\n"
            f"    {response.text[:200]}"
        )
        return None


# ==================================================
# MAIN
# ==================================================
def main():
    start = datetime.fromisoformat(FROM_DATE)
    end   = datetime.fromisoformat(TO_DATE)

    for exchange, cfg in EXCHANGES.items():
        instrument = cfg["instrument"]
        symbols = cfg["symbols"]

        print(f"\n=== Exchange: {exchange} ({instrument}) ===")

        for symbol in symbols:
            print(f"\n--- Symbol: {symbol} ---")

            for data_type in DATA_TYPES:
                print(f"\n  Dataset: {data_type}")

                current = start
                downloaded_files = []

                while current <= end:
                    date_str = current.strftime("%Y-%m-%d")

                    f = download_tardis_csv(
                        exchange=exchange,
                        instrument=instrument,
                        data_type=data_type,
                        symbol=symbol,
                        date=date_str,
                        api_key=API_KEY,
                    )

                    if f:
                        downloaded_files.append(f)

                    current += timedelta(days=1)

                # --------------------------------------------------
                # Quick sanity preview
                # --------------------------------------------------
                if downloaded_files:
                    try:
                        preview_file = downloaded_files[0]
                        print(f"\n  Preview → {preview_file}")
                        with gzip.open(preview_file, "rt") as gz:
                            df_preview = pd.read_csv(gz, nrows=5)
                            print(df_preview.head())
                    except Exception as e:
                        print(f"  Preview failed: {e}")

    print("\nAll downloads complete.")
    print("Data saved under ./data/{exchange}/{instrument}/")


if __name__ == "__main__":
    main()