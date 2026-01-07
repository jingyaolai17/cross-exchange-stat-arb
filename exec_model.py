# core/exec_model.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import math

class Side(str, Enum):
    SHORT_BASIS = "SHORT_BASIS"  # short perp, long spot
    LONG_BASIS  = "LONG_BASIS"   # long perp, short spot

@dataclass
class ExecParams:
    latency_ms: int
    slip_bps_per_leg: float
    fee_bps_binance: float
    fee_bps_kraken: float

def bps_to_mult(bps: float) -> float:
    return 1.0 + bps / 1e4

def bps_frac(bps: float) -> float:
    return bps / 1e4

def fees_usd(notional_usd: float, fee_bps: float) -> float:
    return notional_usd * bps_frac(fee_bps)

def entry_fill_prices(row, side: Side, slip_bps: float) -> Tuple[float, float]:
    """
    Return (spot_px, perp_px) for ENTRY.
    Uses *_fill quotes and applies adverse slippage.
    """
    bid_kr, ask_kr = float(row["best_bid_kr_fill"]), float(row["best_ask_kr_fill"])
    bid_bn, ask_bn = float(row["best_bid_bn_fill"]), float(row["best_ask_bn_fill"])

    if side == Side.SHORT_BASIS:
        # long spot => buy at ask_kr (worse + slip)
        spot_px = ask_kr * bps_to_mult(slip_bps)
        # short perp => sell at bid_bn (worse - slip)
        perp_px = bid_bn * (1.0 - slip_bps / 1e4)
    else:
        # short spot => sell at bid_kr (worse - slip)
        spot_px = bid_kr * (1.0 - slip_bps / 1e4)
        # long perp => buy at ask_bn (worse + slip)
        perp_px = ask_bn * bps_to_mult(slip_bps)

    return spot_px, perp_px

def exit_fill_prices(row, side: Side, slip_bps: float) -> Tuple[float, float]:
    """
    Return (spot_px, perp_px) for EXIT (REVERSE trades).
    This is the critical bugfix vs your current pipeline.
    """
    bid_kr, ask_kr = float(row["best_bid_kr_fill"]), float(row["best_ask_kr_fill"])
    bid_bn, ask_bn = float(row["best_bid_bn_fill"]), float(row["best_ask_bn_fill"])

    if side == Side.SHORT_BASIS:
        # exit: SELL spot, BUY perp
        spot_px = bid_kr * (1.0 - slip_bps / 1e4)
        perp_px = ask_bn * bps_to_mult(slip_bps)
    else:
        # exit: BUY spot, SELL perp
        spot_px = ask_kr * bps_to_mult(slip_bps)
        perp_px = bid_bn * (1.0 - slip_bps / 1e4)

    return spot_px, perp_px