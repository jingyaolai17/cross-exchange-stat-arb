# run/IS_backtest.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from exec_model import ExecParams
from backtest_engine import StrategyParams, run_engine
from reporting import save_freeze, save_reports

def load_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna()
    return df

def normalize_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    if "ts" in df.columns:
        df = df.rename(columns={"ts": "t"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "t"})
    elif "index" in df.columns:
        df = df.rename(columns={"index": "t"})
    else:
        raise ValueError("No recognizable timestamp column")
    return df.sort_values("t")

def merge_asof_kr_to_bn(kr: pd.DataFrame, bn: pd.DataFrame, tol: str) -> pd.DataFrame:
    bn2 = normalize_time_index(bn)
    kr2 = normalize_time_index(kr)

    kr2["t_kr_src"] = kr2["t"]

    m = pd.merge_asof(
        kr2, bn2, on="t", direction="backward",
        tolerance=pd.Timedelta(tol), suffixes=("_kr","_bn")
    ).set_index("t").sort_index()
    return m

def add_latency_fill_view(df: pd.DataFrame, latency_ms: int) -> pd.DataFrame:
    lat = pd.Timedelta(milliseconds=latency_ms)
    fill = df.copy()
    fill.index = fill.index + lat

    out = pd.merge_asof(
        df.reset_index().rename(columns={"t":"t_dec","index":"t_dec"}).sort_values("t_dec"),
        fill.reset_index().rename(columns={"t":"t_fill","index":"t_fill"}).sort_values("t_fill"),
        left_on="t_dec", right_on="t_fill",
        direction="forward", tolerance=lat*2, suffixes=("","_fill")
    ).set_index("t_dec").sort_index()
    return out

def rolling_zscore(x: pd.Series, win: int) -> pd.Series:
    mu = x.rolling(win).mean()
    sd = x.rolling(win).std()
    return (x - mu) / sd

def main():
    cfg_path = Path("configs/freeze_is.json")
    cfg = json.loads(cfg_path.read_text())

    bn = load_features(cfg["paths"]["binance_features"])
    kr = load_features(cfg["paths"]["kraken_features"])

    df = merge_asof_kr_to_bn(kr, bn, cfg["timing"]["merge_tol"])

    # staleness (BUGFIX: use .dt.total_seconds())
    kr_src = pd.to_datetime(df["t_kr_src"], errors="coerce")
    df["kr_staleness_sec"] = (df.index - kr_src).dt.total_seconds()
    df["kr_staleness_sec"] = df["kr_staleness_sec"].clip(lower=0.0)

    # basis + exec edges
    df = df.dropna(subset=["mid_kr","mid_bn","best_bid_kr","best_ask_kr","best_bid_bn","best_ask_bn","spread_bps_kr","spread_bps_bn"])
    df["basis_bps"] = (df["mid_bn"] - df["mid_kr"]) / df["mid_kr"] * 1e4
    df["exec_edge_short_basis_bps"] = (df["best_bid_bn"] - df["best_ask_kr"]) / df["mid_kr"] * 1e4
    df["exec_edge_long_basis_bps"]  = (df["best_bid_kr"] - df["best_ask_bn"]) / df["mid_kr"] * 1e4

    # latency fill view
    df = add_latency_fill_view(df, cfg["timing"]["latency_ms"])
    df = df.dropna(subset=["best_bid_kr_fill","best_ask_kr_fill","best_bid_bn_fill","best_ask_bn_fill"])

    # zscore
    bar_ms = int(cfg["timing"]["bar_ms"])
    bars_per_sec = int(1000 / bar_ms)
    win = int(cfg["signal"]["z_window_sec"] * bars_per_sec)
    df["z"] = rolling_zscore(df["basis_bps"], win)

    # params
    sp = StrategyParams(
        z_entry=float(cfg["signal"]["z_entry"]),
        z_exit=float(cfg["signal"]["z_exit"]),
        stop_z=float(cfg["signal"]["stop_z"]),
        max_hold_sec=float(cfg["risk"]["max_hold_sec"]),
        cooldown_sec=float(cfg["risk"]["cooldown_sec"]),
        min_hold_sec=float(cfg["risk"]["min_hold_sec"]),
        notional_usd=float(cfg["risk"]["notional_usd"]),
        spread_bps_cap=float(cfg["filters"]["spread_bps_cap"]),
        staleness_cap_sec=float(cfg["filters"]["staleness_cap_sec"]),
        min_exec_edge_bps=float(cfg["filters"]["min_exec_edge_bps"]),
    )

    ep = ExecParams(
        latency_ms=int(cfg["timing"]["latency_ms"]),
        slip_bps_per_leg=float(cfg["fees"]["slip_bps_per_leg"]),
        fee_bps_binance=float(cfg["fees"]["binance_taker_bps"]),
        fee_bps_kraken=float(cfg["fees"]["kraken_taker_bps"]),
    )

    trades, equity, pack = run_engine(df, sp, ep)

    start = df.index.min().date()
    end = df.index.max().date()
    out_dir = Path(f"reports/IS_reports_cross_{start}_to_{end}")

    save_freeze(cfg, out_dir)
    save_reports(trades, equity, pack["attribution"], pack["live"], out_dir)

    print("[OK] Saved reports →", out_dir)
    print(pack["live"])

if __name__ == "__main__":
    main()