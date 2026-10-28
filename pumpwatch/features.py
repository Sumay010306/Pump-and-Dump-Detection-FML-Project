"""Causal features: all values available at the close of the current candle."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ["return_1m", "return_5m", "return_15m", "acceleration",
            "volume_z", "volume_ratio", "volatility_5m", "volatility_15m",
            "range_ratio", "body_ratio", "upper_wick", "lower_wick",
            "trade_z", "mean_trade_size", "buy_share", "activity_available"]
DESCRIPTIONS = {
    "return_1m": "Price return over one minute", "return_5m": "Price return over five minutes",
    "return_15m": "Price return over fifteen minutes", "acceleration": "Change in one-minute return",
    "volume_z": "Current volume versus previous 60-minute mean and deviation",
    "volume_ratio": "Current volume relative to previous 60-minute mean",
    "volatility_5m": "Five-minute return standard deviation",
    "volatility_15m": "Fifteen-minute return standard deviation",
    "range_ratio": "High-low range divided by open price", "body_ratio": "Signed candle body divided by open",
    "upper_wick": "Upper wick divided by open", "lower_wick": "Lower wick divided by open",
    "trade_z": "Trade count versus its preceding 60-minute baseline",
    "mean_trade_size": "Base-asset volume per trade", "buy_share": "Taker-buy share of volume",
    "activity_available": "Whether trade count and taker-buy volume were supplied",
}


def _segment(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    r = d.close.pct_change(fill_method=None)
    for n in [1, 5, 15]:
        f[f"return_{n}m"] = d.close.pct_change(n, fill_method=None)
    f["acceleration"] = r.diff()
    previous = d.volume.shift(1).rolling(60, min_periods=60)
    # A floor handles genuinely inactive markets; no future-dependent scaling.
    mean, std = previous.mean(), previous.std(ddof=0)
    f["volume_z"] = ((d.volume - mean) / std.clip(lower=1e-8)).clip(-100, 100)
    f["volume_ratio"] = (d.volume / mean.clip(lower=1e-8)).clip(0, 1000)
    for n in [5, 15]:
        f[f"volatility_{n}m"] = r.rolling(n).std(ddof=0)
    f["range_ratio"] = (d.high - d.low) / d.open
    f["body_ratio"] = (d.close - d.open) / d.open
    f["upper_wick"] = (d.high - d[["open", "close"]].max(axis=1)) / d.open
    f["lower_wick"] = (d[["open", "close"]].min(axis=1) - d.low) / d.open
    activity = "trades" in d and "taker_buy_volume" in d
    f["activity_available"] = float(activity)
    if activity:
        prior_trades = d.trades.shift(1).rolling(60, min_periods=60)
        f["trade_z"] = ((d.trades - prior_trades.mean()) / prior_trades.std(ddof=0).clip(lower=1)).clip(-100, 100)
        f["mean_trade_size"] = d.volume / d.trades.clip(lower=1)
        f["buy_share"] = np.where(d.volume > 0, d.taker_buy_volume / d.volume.clip(lower=1e-8), 0)
    else:
        f["trade_z"] = f["mean_trade_size"] = f["buy_share"] = 0.0
    f["ready"] = np.arange(len(d)) >= 60
    return f


def engineer(candles: pd.DataFrame) -> pd.DataFrame:
    d = candles.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    # Every discontinuity restarts all rolling windows, preventing gap bridging.
    breaks = d.symbol.ne(d.symbol.shift()) | d.timestamp.diff().ne(pd.Timedelta(minutes=1))
    d["segment"] = breaks.cumsum()
    blocks = [_segment(g) for _, g in d.groupby("segment", sort=False)]
    f = pd.concat(blocks).sort_index()
    d = pd.concat([d, f], axis=1)
    d["prediction_time"] = d.timestamp + pd.Timedelta(minutes=1)
    d["ready"] &= np.isfinite(d[FEATURES]).all(axis=1)
    return d


def label_windows(frame: pd.DataFrame, events: pd.DataFrame, announcements: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["target"] = 0
    d["event_id"] = ""
    # Unlabeled vicinity of any known signal is excluded, not called normal.
    for row in announcements.itertuples():
        t = pd.Timestamp(row.announced)
        mask = (d.symbol == row.symbol + "BTC") & (d.timestamp.between(t - pd.Timedelta(minutes=60), t + pd.Timedelta(minutes=60)))
        d.loc[mask, "target"] = -1
    for row in events.itertuples():
        t = pd.Timestamp(row.onset).floor("min")
        same = d.symbol == row.pair
        d.loc[same & d.timestamp.between(t-pd.Timedelta(minutes=60), t+pd.Timedelta(minutes=60)), "target"] = -1
    for row in events.itertuples():
        t = pd.Timestamp(row.onset).floor("min")
        mask = (d.symbol == row.pair) & (d.timestamp >= t) & (d.timestamp < t + pd.Timedelta(minutes=3))
        d.loc[mask, "target"] = 1
        d.loc[mask, "event_id"] = row.event_id
    return d
