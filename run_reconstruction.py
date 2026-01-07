import subprocess
import sys

PYTHON = sys.executable  # ensures same venv

JOBS = [
    # # Binance perp
    {
        "exchange": "binance-futures/perp",
        "symbol": "BTCUSDT",
        "start": "2025-12-29",
        "end": "2026-01-04",
        "freq": "250ms",
    },
    # Kraken spot
    {
        "exchange": "kraken/spot",
        "symbol": "XBT-USD",
        "start": "2025-12-29",
        "end": "2026-01-04",
        "freq": "250ms",
    },
]

for job in JOBS:
    cmd = [
        PYTHON,
        "reconstruction_features.py",
        "--exchange", job["exchange"],
        "--symbol", job["symbol"],
        "--start", job["start"],
        "--end", job["end"],
        "--freq", job["freq"],
        "--topn", "25",
        "--base", "data",
        "--out", "features",
    ]

    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)