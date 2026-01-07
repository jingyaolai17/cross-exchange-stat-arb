"""
reconstruction_features.py
--------------------------------
Cross-exchange, exchange-agnostic L2 order book reconstruction + standardized
microstructure feature generation (NO alpha, NO strategy assumptions).

Works for:
- Binance-perps (snapshot + incremental L2 + trades)  => full replay
- Kraken-spot (snapshot + trades; incremental L2 may be missing) => snapshot-only fallback

Output schema is consistent across venues. When L2 deltas are missing:
- has_l2 = False
- n_l2_updates_in_bar = 0
- ofi = 0
- qi/microprice still computed from current book state (static between snapshots)
"""

import os
import gzip
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, Iterator, List, Tuple, Optional

import numpy as np
import pandas as pd

from ob_core import BookLadder, extract_wide_updates, extract_long_update


# =========================
# Helpers / Types
# =========================

@dataclass
class Update:
    side: str   # "bid" or "ask"
    price: float
    size: float


def _parse_ts(series: pd.Series) -> pd.Series:
    """
    Robust timestamp parsing:
    - If numeric: assume microseconds epoch (Tardis default) => unit='us'
    - Else: parse as datetime string
    Returns tz-naive pandas datetime.
    """
    s = series
    if np.issubdtype(s.dtype, np.number):
        ts = pd.to_datetime(s, unit="us", errors="coerce")
    else:
        ts = pd.to_datetime(s, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    return ts


def daterange(start_date: str, end_date: str) -> Iterable[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    for n in range((end - start).days + 1):
        yield (start + timedelta(days=n)).strftime("%Y-%m-%d")


def _default_kind_map(exchange: str) -> Dict[str, str]:
    return {
        "snapshot": "book_snapshot_25",
        "depth": "incremental_book_L2",
        "trades": "trades",
    }


def _path(base: str, exchange: str, kind: str, date: str, symbol: str, kind_map: Dict[str, str]) -> str:
    folder = kind_map[kind]
    return os.path.join(base, exchange, folder, f"{date}_{symbol}.csv.gz")


def _read_csv_gz(path: str) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f)
    df.columns = [c.lower() for c in df.columns]
    return df


# =========================
# Loaders
# =========================

def load_snapshot_wide(
    base: str,
    exchange: str,
    symbol: str,
    date: str,
    kind_map: Dict[str, str],
    topn: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    path = _path(base, exchange, "snapshot", date, symbol, kind_map)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    print(f"[{exchange} {symbol}] Loading snapshot: {path}")

    snap = _read_csv_gz(path)
    if len(snap) == 0:
        raise ValueError(f"Empty snapshot file: {path}")

    row0 = snap.iloc[0].to_dict()
    ups = extract_wide_updates(row0)
    if not ups:
        raise ValueError("Snapshot row did not yield any updates via extract_wide_updates().")

    asks = [(u.price, u.size) for u in ups if u.side == "ask"]
    bids = [(u.price, u.size) for u in ups if u.side == "bid"]

    if "timestamp" in snap.columns:
        t0 = _parse_ts(snap["timestamp"]).iloc[0]
    elif "ts" in snap.columns:
        t0 = pd.to_datetime(snap["ts"], errors="coerce").iloc[0]
    else:
        raise ValueError("Snapshot missing timestamp column (expected 'timestamp' or 'ts').")

    if pd.isna(t0):
        raise ValueError("Failed to parse snapshot timestamp.")

    asks_df = (
        pd.DataFrame(asks, columns=["price", "size"])
        .sort_values("price", ascending=True)
        .head(topn)
        .reset_index(drop=True)
    )
    bids_df = (
        pd.DataFrame(bids, columns=["price", "size"])
        .sort_values("price", ascending=False)
        .head(topn)
        .reset_index(drop=True)
    )
    return asks_df, bids_df, pd.Timestamp(t0)


def load_trades(
    base: str,
    exchange: str,
    symbol: str,
    date: str,
    kind_map: Dict[str, str],
    freq: str
) -> pd.DataFrame:
    path = _path(base, exchange, "trades", date, symbol, kind_map)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    print(f"[{exchange} {symbol}] Loading trades: {path}")

    t = _read_csv_gz(path)
    if len(t) == 0:
        return pd.DataFrame(columns=["buy", "sell", "n_trades"])

    if "timestamp" not in t.columns:
        raise ValueError("Trades missing 'timestamp' column.")

    t["ts"] = _parse_ts(t["timestamp"])
    t = t.dropna(subset=["ts"]).sort_values("ts")

    if "side" not in t.columns:
        t["side"] = "unknown"
    t["side"] = t["side"].astype(str).str.lower()

    vol_col = "amount" if "amount" in t.columns else ("quantity" if "quantity" in t.columns else None)
    if vol_col is None:
        t["vol"] = 0.0
        vol_col = "vol"

    t["bucket"] = t["ts"].dt.floor(freq)

    side_map = {"b": "buy", "s": "sell", "bid": "buy", "ask": "sell"}
    t["side_norm"] = t["side"].map(side_map).fillna(t["side"])

    agg_vol = t.groupby(["bucket", "side_norm"])[vol_col].sum().unstack(fill_value=0.0)
    if "buy" not in agg_vol.columns:
        agg_vol["buy"] = 0.0
    if "sell" not in agg_vol.columns:
        agg_vol["sell"] = 0.0

    agg_cnt = t.groupby("bucket").size().rename("n_trades")

    out = agg_vol[["buy", "sell"]].join(agg_cnt, how="outer").fillna(0.0).sort_index()
    return out


def iterate_l2_deltas(
    base: str,
    exchange: str,
    symbol: str,
    date: str,
    kind_map: Dict[str, str],
    chunksize: int = 200_000
) -> Iterator[Tuple[pd.Timestamp, List[Update]]]:
    """
    Iterator over L2 delta updates. Raises FileNotFoundError if file missing.
    Caller decides whether to fallback to snapshot-only mode.
    """
    path = _path(base, exchange, "depth", date, symbol, kind_map)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    print(f"[{exchange} {symbol}] Loading incremental L2 deltas: {path}")

    for chunk in pd.read_csv(gzip.open(path), chunksize=chunksize):
        chunk.columns = [c.lower() for c in chunk.columns]
        if "timestamp" not in chunk.columns:
            raise ValueError("L2 deltas missing 'timestamp' column.")

        chunk["ts"] = _parse_ts(chunk["timestamp"])
        chunk = chunk.dropna(subset=["ts"]).sort_values("ts", kind="mergesort")

        if {"price", "side"}.issubset(set(chunk.columns)):
            chunk = chunk.drop_duplicates(subset=["ts", "price", "side"], keep="last")

        for _, row in chunk.iterrows():
            d = row.to_dict()

            ups = extract_wide_updates(d)
            if not ups:
                ups = extract_long_update(d)
            if not ups:
                continue

            converted: List[Update] = []
            for u in ups:
                try:
                    converted.append(Update(side=str(u.side).lower(), price=float(u.price), size=float(u.size)))
                except Exception:
                    continue

            if converted:
                yield pd.Timestamp(row["ts"]), converted


# =========================
# Reconstruction + Features
# =========================

def build_feature_table(
    base_dir: str,
    exchange: str,
    symbol: str,
    date: str,
    freq: str,
    topn: int,
    kind_map: Dict[str, str],
    require_depth: bool = False,
) -> pd.DataFrame:
    print(f"\n=== [{exchange}] Building features for {symbol} | {date} ===")

    asks0, bids0, t0 = load_snapshot_wide(base_dir, exchange, symbol, date, kind_map, topn=topn)

    book = BookLadder(topn)
    for side, df0 in (("ask", asks0), ("bid", bids0)):
        for p, s in df0.itertuples(index=False):
            book.apply_updates([Update(side=side, price=float(p), size=float(s))])

    trades = load_trades(base_dir, exchange, symbol, date, kind_map, freq=freq)

    # -------- L2 depth iterator (optional) --------
    has_l2 = True
    depth_iter: Optional[Iterator[Tuple[pd.Timestamp, List[Update]]]] = None
    ts_next: Optional[pd.Timestamp] = None
    ups_next: Optional[List[Update]] = None

    try:
        depth_iter = iterate_l2_deltas(base_dir, exchange, symbol, date, kind_map)
        ts_next, ups_next = next(depth_iter)
    except FileNotFoundError as e:
        has_l2 = False
        if require_depth:
            raise
        print(f"[{exchange} {symbol}] No incremental L2 deltas → snapshot-only mode ({e})")
        depth_iter = None
        ts_next, ups_next = None, None
    except StopIteration:
        # file exists but empty
        has_l2 = True
        ts_next, ups_next = None, None

    # end of grid
    if len(trades) > 0:
        end_ts = trades.index.max() + pd.Timedelta(freq)
    else:
        end_ts = t0 + pd.Timedelta(hours=24)

    grid = pd.date_range(t0.floor(freq), end_ts, freq=freq)
    feat_rows: List[Dict] = []

    prev_bb, prev_ba = book.best_bid(), book.best_ask()
    prev_sizes = (
        float(prev_bb[1]) if prev_bb[1] is not None else None,
        float(prev_ba[1]) if prev_ba[1] is not None else None,
    )

    updates_in_bar = 0
    current_bar = grid[0] if len(grid) else t0

    for t in grid:
        if t != current_bar:
            updates_in_bar = 0
            current_bar = t

        if has_l2 and depth_iter is not None:
            while ts_next is not None and ts_next <= t:
                book.apply_updates(ups_next)
                updates_in_bar += len(ups_next) if ups_next is not None else 0
                try:
                    ts_next, ups_next = next(depth_iter)
                except StopIteration:
                    ts_next, ups_next = None, None
                    break

        bb, ba = book.best_bid(), book.best_ask()
        spread, mid = book.spread_mid()
        micro = book.microprice()
        qi = book.queue_imbalance()

        # OFI proxy
        ofi = 0.0
        if has_l2:
            if prev_sizes[0] is not None and prev_sizes[1] is not None and bb[1] is not None and ba[1] is not None:
                ofi = (float(bb[1]) - float(prev_sizes[0])) - (float(ba[1]) - float(prev_sizes[1]))
                prev_sizes = (float(bb[1]), float(ba[1]))
        else:
            # snapshot-only: no meaningfully defined "update-driven" OFI
            ofi = 0.0
            updates_in_bar = 0

        # trades on same bucket
        buys = sells = n_trades = 0.0
        if len(trades) > 0 and t in trades.index:
            buys = float(trades.loc[t, "buy"]) if "buy" in trades.columns else 0.0
            sells = float(trades.loc[t, "sell"]) if "sell" in trades.columns else 0.0
            n_trades = float(trades.loc[t, "n_trades"]) if "n_trades" in trades.columns else 0.0

        feat_rows.append(
            {
                "ts": t,
                "exchange": exchange,
                "symbol": symbol,
                "date": date,
                "has_l2": bool(has_l2),
                "best_bid": float(bb[0]) if bb[0] is not None else np.nan,
                "best_ask": float(ba[0]) if ba[0] is not None else np.nan,
                "bid_size": float(bb[1]) if bb[1] is not None else np.nan,
                "ask_size": float(ba[1]) if ba[1] is not None else np.nan,
                "spread": float(spread) if spread is not None else np.nan,
                "mid": float(mid) if mid is not None else np.nan,
                "microprice": float(micro) if micro is not None else np.nan,
                "qi": float(qi) if qi is not None else np.nan,
                "ofi": float(ofi),
                "buy_vol": float(buys),
                "sell_vol": float(sells),
                "trade_imb": float(buys - sells),
                "n_trades": float(n_trades),
                "n_l2_updates_in_bar": float(updates_in_bar),
            }
        )

    df = pd.DataFrame(feat_rows).set_index("ts").sort_index()

    # guards
    df["spread"] = df["spread"].astype(float).clip(lower=0.0)
    df["mid"] = df["mid"].astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df["microprice"] = df["microprice"].astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()

    df["ret_bps"] = (df["mid"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0) * 1e4).clip(-200, 200)
    df["spread_bps"] = ((df["spread"] / df["mid"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 1e4).clip(0, 500)

    return df


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", type=str, required=True, help="e.g. binance-futures/perp, kraken/spot")
    ap.add_argument("--symbol", type=str, required=True, help="e.g. BTCUSDT, XBT-USD")
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--freq", type=str, default="250ms", help="bar frequency, e.g. 100ms, 250ms, 1s")
    ap.add_argument("--topn", type=int, default=25, help="book ladder depth")
    ap.add_argument("--base", type=str, default="data", help="base data folder")
    ap.add_argument("--out", type=str, default="features", help="output features folder")
    ap.add_argument(
        "--require_depth",
        action="store_true",
        help="If set, fail when incremental_book_L2 is missing (no snapshot-only fallback).",
    )
    args = ap.parse_args()

    exchange = args.exchange
    symbol = args.symbol
    start_date = args.start
    end_date = args.end
    freq = args.freq
    topn = int(args.topn)
    base_dir = args.base
    out_dir = args.out
    require_depth = bool(args.require_depth)

    kind_map = _default_kind_map(exchange)

    out_sym_dir = os.path.join(out_dir, exchange, symbol)
    os.makedirs(out_sym_dir, exist_ok=True)

    merged: List[pd.DataFrame] = []
    for date in daterange(start_date, end_date):
        try:
            df_day = build_feature_table(
                base_dir=base_dir,
                exchange=exchange,
                symbol=symbol,
                date=date,
                freq=freq,
                topn=topn,
                kind_map=kind_map,
                require_depth=require_depth,
            )
            merged.append(df_day)

            daily_path = os.path.join(out_sym_dir, f"{date}_{symbol}_{freq}.parquet")
            df_day.to_parquet(daily_path)
            print(f"[OK] Saved daily features → {daily_path} | rows={len(df_day):,}")

        except FileNotFoundError as e:
            print(f"[SKIP] {date} missing inputs: {e}")
            continue
        except Exception as e:
            print(f"[WARN] Skipping {date} due to error: {e}")
            continue

    if not merged:
        print("\nNo days processed successfully. Check your input folders / symbols / exchange names.")
        return

    all_df = pd.concat(merged).sort_index()

    merged_path = os.path.join(out_sym_dir, f"{start_date}_to_{end_date}_{symbol}_{freq}.parquet")
    all_df.to_parquet(merged_path)

    print("\n==============================")
    print("Per-venue feature build complete.")
    print(f"Exchange : {exchange}")
    print(f"Symbol   : {symbol}")
    print(f"Range    : {start_date} → {end_date}")
    print(f"Freq     : {freq}")
    print(f"Rows     : {len(all_df):,}")
    print(f"Saved    : {merged_path}")
    print("==============================\n")


if __name__ == "__main__":
    main()