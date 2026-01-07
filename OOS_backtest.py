"""
oos_backtest_BTCUSDT.py
----------------------
Out-of-sample validation for BTCUSDT PERP Flow-Control strategy.

Design principles:
- OOS consumes ONLY freeze_is.json (no re-tuning)
- Pipeline order identical to IS_backtest_BTCUSDT.py
- Re-runs alpha module (microprice+OFI) using frozen alpha knobs
- Optional latency stress test grid

Outputs (in OOS_reports_BTCUSDT_PERP/):
- latency_grid_summary.csv
- oos_attribution_latency_{X}ms.csv
- oos_live_metrics_latency_{X}ms.json
- oos_pnl_latency_{X}ms.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from strategy_core_patched import (
    StrategyParams,
    SignalParams, RegimeParams, RiskParams,
    Fees,
    build_base_features,
    apply_regime_scalers,
    apply_alpha_gating,
    apply_latency_to_decision,
    execute_account,
    apply_killswitch_and_flatten,
    compute_attribution_table,
    compute_live_metrics,
    assert_invariants,
    to_bps,
)

from alpha.microprice_ofi_alpha import compute_microprice_ofi_alpha


# --------------------------------------------------
# Paths
# --------------------------------------------------
FEATURE_PATH = "data/binance-futures/features/2025-12-17_to_2025-12-19_BTCUSDT_100ms.parquet"
FREEZE_PATH  = "IS_reports_BTCUSDT_PERP/freeze_is.json"
REPORT_DIR   = "OOS_reports_BTCUSDT_PERP"
os.makedirs(REPORT_DIR, exist_ok=True)


# --------------------------------------------------
# Helpers (match IS)
# --------------------------------------------------
def safe_ffill(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).ffill().bfill()

def q(series: pd.Series):
    return series.quantile([0.5, 0.9, 0.99]).to_dict()

def hold_intent(intent: pd.Series, every_n: int) -> pd.Series:
    """Update intent only every N bars; forward-fill in between."""
    intent = intent.astype(float)
    if every_n <= 1:
        return intent
    idx = np.arange(len(intent))
    mask = (idx % int(every_n) == 0)
    held = pd.Series(np.where(mask, intent.values, np.nan), index=intent.index).ffill().fillna(0.0)
    return held.astype(float)

def apply_maker_preference(df: pd.DataFrame, intent_col: str, maker_min: float, penalty_power: float) -> pd.DataFrame:
    """Apply maker preference exactly once, post-gating (same as IS)."""
    df = df.copy()
    if float(penalty_power) > 0.0:
        df[intent_col] = (
            df[intent_col].astype(float)
            * df["f_maker"].astype(float).clip(0.0, 1.0).pow(float(penalty_power))
        ).astype(float)
    df.loc[df["f_maker"].astype(float) < float(maker_min), intent_col] = 0.0
    return df

def force_btcusd_usd_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    BTCUSDT is already USD-quoted. Force mid_usd/spread_usd/spread_bps to be consistent
    to avoid any accidental FX scaling.
    """
    df = df.copy()
    mid = df["mid"].astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    spread = df["spread"].astype(float).clip(lower=0.0).fillna(0.0)

    df["mid_usd"] = mid
    df["spread_usd"] = spread
    df["spread_bps"] = to_bps((spread / mid.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)).fillna(0.0)
    df["half_spread_bps"] = 0.5 * df["spread_bps"]
    return df


# --------------------------------------------------
# Load freeze (single source of truth)
# --------------------------------------------------
with open(FREEZE_PATH, "r") as f:
    fr = json.load(f)

# Build StrategyParams explicitly to avoid __init__ unexpected keys like "version"
params = StrategyParams(
    fees=Fees(**fr["fees"]),
    signal=SignalParams(**fr["signal"]),
    regime=RegimeParams(
        vol_pct_hi=fr["filters"]["vol_pct_hi"],
        toxic_mult=fr["filters"]["toxic_mult"],
        scaler_combine=fr["scaler"]["combine"],
        scaler_weights=tuple(fr["scaler"]["weights"]),
    ),
    risk=RiskParams(
        capital_usd=fr["risk"]["capital_usd"],
        notional_usd=fr["risk"]["notional_usd"],
        inv_cap_units=fr["risk"]["inv_cap_units"],
        inv_decay_lambda=fr["risk"]["inv_decay_lambda"],
    ),
)

# Frozen alpha/intent/maker knobs
alpha_knobs = fr["alpha_modules"]["microprice_ofi"]
intent_knobs = fr["alpha_modules"]["intent"]
maker_knobs = fr["alpha_modules"]["maker"]

MP_OFI_ALPHA_SCALE_BPS = float(alpha_knobs["alpha_scale_bps"])
MP_EDGE_CLIP           = float(alpha_knobs["mp_clip"])
OFI_FAST_SPAN          = int(alpha_knobs["ofi_fast_span"])
OFI_SLOW_SPAN          = int(alpha_knobs["ofi_slow_span"])
ALPHA_CLIP_BPS         = float(alpha_knobs["alpha_clip_bps"])
MP_DEADZONE_HS         = float(alpha_knobs.get("mp_deadzone_hs", 0.25))
TREND_STRENGTH_THRESH  = float(alpha_knobs["trend_strength_thresh"])
ONE_SIDED              = bool(alpha_knobs["one_sided"])

INTENT_DENOM_BPS       = float(intent_knobs["denom_bps"])
HOLD_BARS              = int(intent_knobs["hold_bars"])

MAKER_MIN              = float(maker_knobs["maker_min"])
MAKER_PENALTY_POWER    = float(maker_knobs["penalty_power"])


# --------------------------------------------------
# Load OOS features
# --------------------------------------------------
df_raw = pd.read_parquet(FEATURE_PATH)
df_raw.index = pd.to_datetime(df_raw.index)
if getattr(df_raw.index, "tz", None) is not None:
    df_raw.index = df_raw.index.tz_localize(None)
df_raw = df_raw.sort_index()

need_cols = ["mid", "spread", "microprice", "ofi", "ofi_z", "qi", "bid_size", "ask_size"]
missing = [c for c in need_cols if c not in df_raw.columns]
if missing:
    raise ValueError(f"Missing required columns in OOS features: {missing}")

df_raw["spread"] = df_raw["spread"].astype(float).abs().clip(lower=0.0)
df_raw["mid"] = safe_ffill(df_raw["mid"].astype(float))
df_raw["microprice"] = safe_ffill(df_raw["microprice"].astype(float))
df_raw["ofi"] = df_raw["ofi"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
df_raw["ofi_z"] = df_raw["ofi_z"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
df_raw["qi"] = df_raw["qi"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.5)


# --------------------------------------------------
# Latency grid (can change)
# --------------------------------------------------
LATENCY_GRID_MS = [0, 50, 100, 300, 500]
rows = []

for latency_ms in LATENCY_GRID_MS:
    print(f"\n=== OOS run | latency = {latency_ms} ms ===")

    df = df_raw.copy()

    # ----------------------------
    # 1) base features
    # ----------------------------
    df = build_base_features(df, params)
    df["mid"] = safe_ffill(df["mid"].astype(float))
    df["spread"] = df["spread"].astype(float).fillna(0.0).clip(lower=0.0)

    # Force BTCUSDT USD consistency (same as IS)
    df = force_btcusd_usd_consistency(df)

    # Optional scaling sanity (same logic as IS)
    p99 = float(df["spread_bps"].quantile(0.99))
    if p99 > 80.0:
        scale = 1.0
        while p99 / scale > 80.0 and scale < 1e6:
            scale *= 10.0
        print(f"[WARN] spread_bps p99={p99:.2f} too large. Auto-normalizing by /{scale:g}")
        df["spread_bps"] = df["spread_bps"] / scale
        df["half_spread_bps"] = df["half_spread_bps"] / scale
        if "spread_usd" in df.columns and "mid_usd" in df.columns:
            df["spread_usd"] = (df["spread_bps"] / 1e4) * df["mid_usd"]

    # ----------------------------
    # 2) alpha module (MUST exist in OOS)
    # ----------------------------
    alpha_df = compute_microprice_ofi_alpha(
        df,
        alpha_scale_bps=MP_OFI_ALPHA_SCALE_BPS,
        mp_clip=MP_EDGE_CLIP,
        ofi_fast_span=OFI_FAST_SPAN,
        ofi_slow_span=OFI_SLOW_SPAN,
        mp_deadzone_hs=MP_DEADZONE_HS,
        trend_fast_span=200,
        trend_slow_span=2000,
        trend_strength_thresh=TREND_STRENGTH_THRESH,
        one_sided=ONE_SIDED,
    )

    df["alpha_bps"] = alpha_df["alpha_bps"].astype(float).clip(-ALPHA_CLIP_BPS, ALPHA_CLIP_BPS)
    df["mp_edge"] = alpha_df.get("mp_edge", pd.Series(0.0, index=df.index)).astype(float)
    df["ofi_norm"] = alpha_df.get("ofi_norm", pd.Series(0.0, index=df.index)).astype(float)
    df["trend_strength"] = alpha_df.get("trend_strength", pd.Series(0.0, index=df.index)).astype(float)
    df["trend_dir"] = alpha_df.get("trend_dir", pd.Series(0.0, index=df.index)).astype(float)

    # ----------------------------
    # 3) alpha -> sig_intent (same as IS)
    # ----------------------------
    df["sig_intent"] = (
        float(params.signal.sig_strength)
        * np.tanh(df["alpha_bps"] / float(INTENT_DENOM_BPS))
    ).fillna(0.0).astype(float)

    # ----------------------------
    # 4) maker/taker split (same as IS)
    # ----------------------------
    maker_edge = df["alpha_bps"] - float(params.signal.join_thresh) * df["half_spread_bps"].astype(float)
    df["join_maker"] = maker_edge >= 0.0

    if "p_maker_fill" not in df.columns:
        raise ValueError("build_base_features() must produce 'p_maker_fill'. Missing in OOS.")

    df["f_maker"] = (df["join_maker"].astype(float) * df["p_maker_fill"].astype(float)).clip(0.0, 1.0)
    df["f_taker"] = (1.0 - df["f_maker"]).clip(0.0, 1.0)

    # ----------------------------
    # 5) flow-control overlay (same as IS)
    # ----------------------------
    df = apply_regime_scalers(df, params, improved=True)

    # ----------------------------
    # 6) gating
    # ----------------------------
    df = apply_alpha_gating(
        df, params,
        intent_col_in="sig_intent_scaled",
        intent_col_out="intent_gated",
        alpha_col="alpha_bps",
    )

    # ----------------------------
    # 7) hold + maker pref (same as IS)
    # ----------------------------
    df["intent_gated"] = hold_intent(df["intent_gated"].astype(float), HOLD_BARS)
    df = apply_maker_preference(df, intent_col="intent_gated", maker_min=MAKER_MIN, penalty_power=MAKER_PENALTY_POWER)

    # ----------------------------
    # 8) latency + execution + killswitch
    # ----------------------------
    df = apply_latency_to_decision(df, params, intent_col="intent_gated", latency_ms=latency_ms)
    df = execute_account(df, params, intent_col="intent_gated")
    df = apply_killswitch_and_flatten(df, params, intent_col="intent_gated")
    assert_invariants(df, params, intent_col="intent_gated")

    # ----------------------------
    # Metrics + save artifacts
    # ----------------------------
    att = compute_attribution_table(df, params)
    live = compute_live_metrics(df, params, intent_col="intent_gated")

    # Robust sharpe extraction (in case metric label differs)
    def get_metric(att_df, name: str, default=np.nan):
        m = att_df.loc[att_df["Metric"] == name, "Value"]
        return float(m.values[0]) if len(m) else float(default)

    sharpe_daily = get_metric(att, "Sharpe (daily)", default=np.nan)
    max_dd = get_metric(att, "Max Drawdown (USD)", default=np.nan)

    rows.append({
        "latency_ms": latency_ms,
        "daily_sharpe": sharpe_daily,
        "max_drawdown_usd": max_dd,
        "turnover_usd": float(live.get("turnover_usd", 0.0)),
        "net_edge_bps_on_traded": float(live.get("net_edge_bps_on_traded", np.nan)),
        "total_pnl_usd": float(live.get("total_pnl_after_usd", 0.0)),
        "total_pnl_bps": float(live.get("total_pnl_after_bps_on_capital", 0.0)),
        "alpha_nonzero_rate": float((df["alpha_bps"].abs() > 0).mean()),
        "intent_nonzero_rate": float((df["intent_gated"].abs() > 0).mean()),
    })

    # Save attribution + live metrics per latency
    att_path = os.path.join(REPORT_DIR, f"oos_attribution_latency_{latency_ms}ms.csv")
    att.to_csv(att_path, index=False)

    live_path = os.path.join(REPORT_DIR, f"oos_live_metrics_latency_{latency_ms}ms.json")
    with open(live_path, "w") as f:
        json.dump(live, f, indent=2)

    # Plot
    plt.figure(figsize=(11, 4))
    df["pnl_after_bps"].cumsum().plot()
    plt.title(f"OOS BTCUSDT PERP | Latency = {latency_ms} ms")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL (bps)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, f"oos_pnl_latency_{latency_ms}ms.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("alpha_nonzero_rate:", float((df["alpha_bps"].abs() > 0).mean()))
    print("intent_nonzero_rate:", float((df["intent_gated"].abs() > 0).mean()))
    print("turnover_usd:", float(live.get("turnover_usd", 0.0)))
    print("total_pnl_bps:", float(live.get("total_pnl_after_bps_on_capital", 0.0)))


# --------------------------------------------------
# Save summary
# --------------------------------------------------
summary = pd.DataFrame(rows)
summary_path = os.path.join(REPORT_DIR, "latency_grid_summary.csv")
summary.to_csv(summary_path, index=False)

print("\nOOS validation complete.")
print(f"Latency summary saved → {summary_path}")
print(f"Reports written to → {REPORT_DIR}")