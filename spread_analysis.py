"""
spread_analysis.py
------------------
Cross-exchange spread analysis for:
- Binance Futures PERP: BTCUSDT
- Kraken Spot: XBT-USD

Run:
    python spread_analysis.py

Inputs:
- features/binance-futures/perp/BTCUSDT/2025-12-15_to_2025-12-21_BTCUSDT_250ms.parquet
- features/kraken/spot/XBT-USD/2025-12-15_to_2025-12-21_XBT-USD_250ms.parquet

Outputs:
- reports/spread_analysis_<START>_to_<END>/
"""

from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# CONFIG (single source of truth)
# =========================

BINANCE_FEATURE = Path(
    "features/binance-futures/perp/BTCUSDT/"
    "2025-12-15_to_2025-12-21_BTCUSDT_250ms.parquet"
)

KRAKEN_FEATURE = Path(
    "features/kraken/spot/XBT-USD/"
    "2025-12-15_to_2025-12-21_XBT-USD_250ms.parquet"
)

FREQ_MS = 250
MERGE_TOL = pd.Timedelta("1s")

Z_WINDOW_SEC = 30
Z_ENTRY = 3.0
Z_EXIT = 0.5

SPREAD_BPS_CAP = 10  # liquidity sanity filter


# =========================
# Helpers
# =========================

def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path).sort_index()
    return df[~df.index.duplicated(keep="last")]


def merge_asof(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    l = left.reset_index().rename(columns={"ts": "t", "index": "t"})
    r = right.reset_index().rename(columns={"ts": "t", "index": "t"})

    m = pd.merge_asof(
        l.sort_values("t"),
        r.sort_values("t"),
        on="t",
        direction="backward",
        tolerance=MERGE_TOL,
        suffixes=("_kr", "_bn"),
    )
    return m.set_index("t").sort_index()


def event_study(z: pd.Series) -> pd.DataFrame:
    zvals = z.values
    t = z.index

    events = []
    i, n = 0, len(zvals)

    while i < n:
        if np.isfinite(zvals[i]) and abs(zvals[i]) >= Z_ENTRY:
            t0 = t[i]
            j = i + 1
            while j < n and (not np.isfinite(zvals[j]) or abs(zvals[j]) > Z_EXIT):
                j += 1
            if j < n:
                events.append({
                    "start_time": t0,
                    "end_time": t[j],
                    "duration_sec": (t[j] - t0).total_seconds(),
                })
            i = j
        else:
            i += 1

    return pd.DataFrame(events)


# =========================
# Main
# =========================

def main():
    print("\n=== Loading reconstructed features ===")

    bn = load_features(BINANCE_FEATURE)
    kr = load_features(KRAKEN_FEATURE)

    start = max(bn.index.min(), kr.index.min()).date()
    end = min(bn.index.max(), kr.index.max()).date()

    report_dir = Path(f"reports/spread_analysis_{start}_to_{end}")
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Binance file: {BINANCE_FEATURE}")
    print(f"Kraken  file: {KRAKEN_FEATURE}")
    print(f"Date range : {start} → {end}")
    print(f"Reports →  {report_dir}")

    # ---------------------
    # Align
    # ---------------------
    merged = merge_asof(kr, bn)
    merged = merged.dropna(subset=["mid_kr", "mid_bn"])

    # ---------------------
    # Basis
    # ---------------------
    merged["basis"] = merged["mid_bn"] - merged["mid_kr"]
    merged["basis_bps"] = merged["basis"] / merged["mid_kr"] * 1e4

    merged = merged[
        (merged["spread_bps_kr"] < SPREAD_BPS_CAP) &
        (merged["spread_bps_bn"] < SPREAD_BPS_CAP)
    ]

    # ---------------------
    # Z-score
    # ---------------------
    bars_per_sec = int(1000 / FREQ_MS)
    win = Z_WINDOW_SEC * bars_per_sec

    mu = merged["basis_bps"].rolling(win).mean()
    sd = merged["basis_bps"].rolling(win).std()
    merged["z"] = (merged["basis_bps"] - mu) / sd

    # ---------------------
    # Event study
    # ---------------------
    events = event_study(merged["z"])

    # ---------------------
    # Save outputs
    # ---------------------
    merged["basis_bps"].describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    ).to_csv(report_dir / "basis_distribution.csv")

    if len(events) > 0:
        events.to_csv(report_dir / "event_study.csv", index=False)

    with open(report_dir / "summary.txt", "w") as f:
        f.write("Cross-Exchange Spread Analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Date range: {start} → {end}\n")
        f.write(f"Freq      : {FREQ_MS} ms\n")
        f.write(f"Z-window  : {Z_WINDOW_SEC} sec\n\n")

        f.write("Basis (bps):\n")
        f.write(str(merged["basis_bps"].describe()) + "\n\n")

        f.write("Z-score exceedance rates:\n")
        for zthr in [2, 3, 4]:
            rate = (merged["z"].abs() > zthr).mean()
            f.write(f"|z| > {zthr}: {rate:.4%}\n")

        f.write("\nEvent study (|z| > 3 → < 0.5):\n")
        if len(events) > 0:
            f.write(str(events["duration_sec"].describe()) + "\n")
        else:
            f.write("No events detected.\n")

    print("\n=== Spread analysis complete ===\n")


if __name__ == "__main__":
    main()