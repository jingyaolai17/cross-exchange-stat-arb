import numpy as np
import pandas as pd

def compute_microprice_ofi_alpha(
    df: pd.DataFrame,
    alpha_scale_bps: float = 8.0,
    mp_clip: float = 3.0,
    ofi_fast_span: int = 3,
    ofi_slow_span: int = 30,
    vol_span: int = 20,
    mp_deadzone_hs: float = 0.25,
    trend_fast_span: int = 200,
    trend_slow_span: int = 2000,
    trend_strength_thresh: float = 0.25,
    one_sided: bool = True,
) -> pd.DataFrame:

    out = pd.DataFrame(index=df.index)
    eps = 1e-12

    mid = df["mid"].astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    mp  = df["microprice"].astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    spread = df["spread"].astype(float).clip(lower=0.0).fillna(0.0)

    # --- vol (for normalization/debug) ---
    ret = np.log(mid.replace(0.0, np.nan)).diff()
    vol = (
        ret.ewm(span=vol_span, adjust=False)
        .std()
        .replace([0.0, np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(1e-6)
        .clip(lower=1e-6)
    )
    out["vol_bps"] = (vol * 1e4).clip(lower=0.05)

    # --- microprice edge in half-spread units ---
    half_spread = (0.5 * spread).replace(0.0, np.nan).ffill()
    mp_edge_hs = ((mp - mid) / (half_spread + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mp_edge_hs = mp_edge_hs.clip(-mp_clip, mp_clip)
    out["mp_edge"] = mp_edge_hs

    # --- OFI persistence ---
    ofi = df["ofi"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ofi_fast = ofi.ewm(span=ofi_fast_span, adjust=False).mean()
    ofi_slow = ofi.ewm(span=ofi_slow_span, adjust=False).mean()
    ofi_norm = (ofi_fast - ofi_slow).clip(-5.0, 5.0)
    out["ofi_norm"] = ofi_norm

    # --- trend direction/strength ---
    ema_fast = mid.ewm(span=trend_fast_span, adjust=False).mean()
    ema_slow = mid.ewm(span=trend_slow_span, adjust=False).mean()
    trend_raw = (ema_fast - ema_slow)

    trend_dir = np.sign(trend_raw).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    denom = (mid.abs() * vol + eps)
    trend_strength = (trend_raw.abs() / denom).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out["trend_dir"] = trend_dir
    out["trend_strength"] = trend_strength

    # --- causal features ---
    mp_e = out["mp_edge"].shift(1).fillna(0.0)
    ofi_p = out["ofi_norm"].shift(1).fillna(0.0)
    tdir = out["trend_dir"].shift(1).fillna(0.0)
    tstr = out["trend_strength"].shift(1).fillna(0.0)

    # =========================
    # NEW: smoother alpha build
    # =========================

    # (A) microprice size weight: soft deadzone ramp
    x = mp_e.abs()
    dz = float(mp_deadzone_hs)
    mp_w = np.clip((x - dz) / (0.35 + eps), 0.0, 1.0)  # ramp width

    # (B) OFI amplitude: allow >1 when OFI supports direction
    ofi_amp = 1.0 + 0.6 * np.tanh(0.8 * ofi_p)  # in [0.4, 1.6]

    # (C) agreement weight: don't hard-zero disagreement
    same = (np.sign(mp_e) == np.sign(ofi_p))
    agree_w = np.where(same, 1.0, 0.35)  # start 0.35

    # (D) trend weight: smooth; used as multiplier not gate
    th = float(trend_strength_thresh)
    trend_w = np.clip((tstr - th) / (0.25 + eps), 0.0, 1.0)  # width controls how fast it turns on

    # (E) alignment weight: only punish counter-trend strongly when trend is strong
    if one_sided:
        counter = (np.sign(mp_e) != np.sign(tdir)) & (tdir.abs() > 0.0)
        align_w = np.where(counter, 1.0 - 0.9 * trend_w, 1.0)  # 0.1 at strong trend, ~1 when weak trend
    else:
        align_w = 1.0

    alpha_raw = mp_e * mp_w * ofi_amp * agree_w * (0.35 + 0.65 * trend_w) * align_w

    out["alpha_bps"] = (alpha_raw * float(alpha_scale_bps)).clip(-12.0, 12.0)
    return out