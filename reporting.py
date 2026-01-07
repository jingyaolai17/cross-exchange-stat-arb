# core/reporting.py
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

def save_freeze(cfg: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "freeze_is.json", "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

def save_reports(trades: pd.DataFrame, equity: pd.DataFrame, attribution: pd.DataFrame, live: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    trades.to_csv(out_dir / "trades.csv", index=False)
    equity.to_csv(out_dir / "equity_curve.csv")
    attribution.to_csv(out_dir / "attribution.csv", index=False)

    with open(out_dir / "live_metrics.json", "w") as f:
        json.dump(live, f, indent=2, sort_keys=True)

    # cumulative pnl plot
    plt.figure()
    equity["equity_usd"].plot()
    plt.title("Cross-Exchange StatArb | Cumulative Realized PnL (USD)")
    plt.xlabel("Time")
    plt.ylabel("PnL (USD)")
    plt.tight_layout()
    plt.savefig(out_dir / "cumulative_pnl.png", dpi=150)
    plt.close()