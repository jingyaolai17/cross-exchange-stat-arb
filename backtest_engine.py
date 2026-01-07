# core/backtest_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from exec_model import Side, ExecParams, entry_fill_prices, exit_fill_prices, fees_usd

@dataclass
class StrategyParams:
    z_entry: float
    z_exit: float
    stop_z: float
    max_hold_sec: float
    cooldown_sec: float
    min_hold_sec: float
    notional_usd: float
    spread_bps_cap: float
    staleness_cap_sec: float
    min_exec_edge_bps: float

@dataclass
class Position:
    side: Side
    entry_time: pd.Timestamp
    spot_qty: float
    perp_qty: float
    spot_entry_px: float
    perp_entry_px: float
    entry_fees_usd: float

def decide_side(z: float) -> Optional[Side]:
    if z != z:
        return None
    if z >= 0:
        return Side.SHORT_BASIS
    return Side.LONG_BASIS

def run_engine(df: pd.DataFrame, sp: StrategyParams, ep: ExecParams) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    df must contain:
      basis_bps, z,
      best_bid/ask_{kr,bn}, spread_bps_{kr,bn}, kr_staleness_sec,
      best_bid/ask_{kr,bn}_fill (fill-time quotes)
      exec_edge_short_basis_bps, exec_edge_long_basis_bps
    """
    pos: Optional[Position] = None
    last_exit_t: Optional[pd.Timestamp] = None

    logs: List[Dict[str, Any]] = []
    equity: List[Dict[str, Any]] = []
    realized_pnl = 0.0

    traded_notional = 0.0
    gross_edge_bps_sum = 0.0
    cost_bps_sum = 0.0

    for t, row in df.iterrows():
        z = float(row["z"]) if row["z"] == row["z"] else np.nan

        # compute current exec edge (directional)
        edge_short = float(row["exec_edge_short_basis_bps"])
        edge_long  = float(row["exec_edge_long_basis_bps"])
        side_hint = decide_side(z)
        exec_edge = edge_short if side_hint == Side.SHORT_BASIS else edge_long

        # filters
        if float(row["spread_bps_bn"]) > sp.spread_bps_cap or float(row["spread_bps_kr"]) > sp.spread_bps_cap:
            liquidity_ok = False
        else:
            liquidity_ok = True

        stale_ok = float(row["kr_staleness_sec"]) <= sp.staleness_cap_sec

        # time since exit
        since_exit = (t - last_exit_t).total_seconds() if last_exit_t is not None else 1e9

        # manage position
        if pos is None:
            # entry conditions
            if (
                z == z
                and abs(z) >= sp.z_entry
                and liquidity_ok
                and stale_ok
                and since_exit >= sp.cooldown_sec
                and exec_edge >= sp.min_exec_edge_bps
            ):
                side = decide_side(z)
                if side is None:
                    continue

                spot_px, perp_px = entry_fill_prices(row, side, ep.slip_bps_per_leg)
                spot_qty = sp.notional_usd / spot_px
                perp_qty = sp.notional_usd / perp_px

                entry_fee = fees_usd(sp.notional_usd, ep.fee_bps_kraken) + fees_usd(sp.notional_usd, ep.fee_bps_binance)

                pos = Position(
                    side=side,
                    entry_time=t,
                    spot_qty=float(spot_qty),
                    perp_qty=float(perp_qty),
                    spot_entry_px=float(spot_px),
                    perp_entry_px=float(perp_px),
                    entry_fees_usd=float(entry_fee),
                )

                logs.append({
                    "time": t, "event": "ENTER", "side": side.value,
                    "z": z, "basis_bps": float(row["basis_bps"]),
                    "exec_edge_bps": float(exec_edge),
                    "reason": "z_entry+exec_edge",
                })

                # attribution: gross edge on traded
                traded_notional += 2.0 * sp.notional_usd
                gross_edge_bps_sum += float(exec_edge) * (2.0 * sp.notional_usd)

                # costs (fees + slippage) per round-trip will be accounted at exit; here store per-entry if you want
        else:
            hold_sec = (t - pos.entry_time).total_seconds()

            # exit conditions (don’t allow instant flip-flop)
            exit_ok = False
            reason = ""

            if hold_sec >= sp.max_hold_sec:
                exit_ok = True; reason = "max_hold"
            elif abs(z) >= sp.stop_z:
                exit_ok = True; reason = "stop_z"
            elif hold_sec >= sp.min_hold_sec and abs(z) <= sp.z_exit:
                exit_ok = True; reason = "z_exit"

            if exit_ok:
                spot_exit_px, perp_exit_px = exit_fill_prices(row, pos.side, ep.slip_bps_per_leg)

                # pnl calc by leg
                if pos.side == Side.SHORT_BASIS:
                    # long spot pnl
                    spot_pnl = pos.spot_qty * (spot_exit_px - pos.spot_entry_px)
                    # short perp pnl
                    perp_pnl = pos.perp_qty * (pos.perp_entry_px - perp_exit_px)
                else:
                    # short spot pnl
                    spot_pnl = pos.spot_qty * (pos.spot_entry_px - spot_exit_px)
                    # long perp pnl
                    perp_pnl = pos.perp_qty * (perp_exit_px - pos.perp_entry_px)

                exit_fee = fees_usd(sp.notional_usd, ep.fee_bps_kraken) + fees_usd(sp.notional_usd, ep.fee_bps_binance)
                total_fees = pos.entry_fees_usd + exit_fee

                realized = float(spot_pnl + perp_pnl - total_fees)
                realized_pnl += realized

                logs.append({
                    "time": t, "event": "EXIT", "side": pos.side.value,
                    "z": z, "basis_bps": float(row["basis_bps"]),
                    "realized_pnl_usd": realized,
                    "fees_usd": float(total_fees),
                    "hold_sec": float(hold_sec),
                    "reason": reason,
                })

                # attribution costs on traded
                # cost_bps computed vs 2*notional
                cost_usd = float(total_fees)
                cost_bps = (cost_usd / (2.0 * sp.notional_usd)) * 1e4
                cost_bps_sum += cost_bps * (2.0 * sp.notional_usd)

                pos = None
                last_exit_t = t

        equity.append({"time": t, "equity_usd": realized_pnl})

    trades = pd.DataFrame(logs)
    eq = pd.DataFrame(equity).set_index("time")

    # live metrics / attribution
    exits = trades[trades["event"] == "EXIT"].copy()
    n_trades = int(len(exits))

    avg_fees = float(exits["fees_usd"].mean()) if n_trades > 0 else 0.0
    hit_rate = float((exits["realized_pnl_usd"] > 0).mean()) if n_trades > 0 else 0.0
    avg_hold = float(exits["hold_sec"].mean()) if n_trades > 0 else 0.0

    gross_edge_bps_on_traded = (gross_edge_bps_sum / traded_notional) if traded_notional > 0 else 0.0
    cost_bps_on_traded = (cost_bps_sum / traded_notional) if traded_notional > 0 else 0.0
    net_edge_bps_on_traded = gross_edge_bps_on_traded - cost_bps_on_traded

    live = {
        "turnover_usd": traded_notional,
        "gross_edge_bps_on_traded": gross_edge_bps_on_traded,
        "cost_bps_on_traded": cost_bps_on_traded,
        "net_edge_bps_on_traded": net_edge_bps_on_traded,
        "total_pnl_after_usd": float(realized_pnl),
        "trades": n_trades,
        "hit_rate": hit_rate,
        "avg_hold_sec": avg_hold,
        "avg_fees_usd": avg_fees,
    }

    attribution = pd.DataFrame([{
        "Metric": "Gross edge (bps on traded)", "Value": gross_edge_bps_on_traded
    },{
        "Metric": "Cost (bps on traded)", "Value": cost_bps_on_traded
    },{
        "Metric": "Net edge (bps on traded)", "Value": net_edge_bps_on_traded
    },{
        "Metric": "Turnover (USD)", "Value": traded_notional
    },{
        "Metric": "Total PnL after (USD)", "Value": float(realized_pnl)
    },{
        "Metric": "Trades", "Value": n_trades
    },{
        "Metric": "Hit rate", "Value": hit_rate
    },{
        "Metric": "Avg hold (sec)", "Value": avg_hold
    },{
        "Metric": "Avg fees (USD)", "Value": avg_fees
    }])

    return trades, eq, {"live": live, "attribution": attribution}