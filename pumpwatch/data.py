"""Acquisition, provenance, strict candle validation, and research onset labels."""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESEARCH = "https://raw.githubusercontent.com/SystemsLab-Sapienza/pump-and-dump-dataset/master/"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]


def fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=45)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Download did not complete")


def validate_candles(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One pair, open-time timestamps, completed 1-minute candles only.

    Reject corrupt records; never fill gaps or fabricate zero-volume candles.
    Exact duplicates are removed; conflicting timestamps are rejected.
    """
    d = frame.copy()
    missing = set(REQUIRED) - set(d.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if "symbol" in d and d.symbol.nunique() != 1:
        raise ValueError("Upload exactly one trading pair at a time.")
    if d.empty:
        raise ValueError("No candle records found.")
    t = d.timestamp
    numeric = pd.to_numeric(t, errors="coerce")
    if numeric.notna().all():
        magnitude = float(numeric.abs().median())
        unit = "us" if magnitude > 1e14 else "ms" if magnitude > 1e11 else "s"
        d["timestamp"] = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        d["timestamp"] = pd.to_datetime(t, utc=True, errors="coerce", format="mixed")
    if d.timestamp.isna().any():
        raise ValueError("Invalid timestamps. Supply ISO UTC dates or epoch seconds/ms/us.")
    if (d.timestamp != d.timestamp.dt.floor("min")).any():
        raise ValueError("Timestamps must mark one-minute candle OPEN times.")
    numeric_cols = [c for c in COLUMNS[1:6] + ["quote_volume", "trades", "taker_buy_volume"] if c in d]
    for c in numeric_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        if not np.isfinite(d[c]).all():
            raise ValueError(f"Column {c} contains missing or non-finite values.")
    if (d[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive.")
    if (d[[c for c in numeric_cols if c not in ["open", "high", "low", "close"]]] < 0).any().any():
        raise ValueError("Volumes and trade counts cannot be negative.")
    if ((d.high < d[["open", "close", "low"]].max(axis=1)) |
        (d.low > d[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid candle: high/low do not contain open and close.")
    if "taker_buy_volume" in d and (d.taker_buy_volume > d.volume + 1e-8).any():
        raise ValueError("Taker-buy volume cannot exceed total volume.")
    if "trades" in d and (d.trades % 1 != 0).any():
        raise ValueError("Trade counts must be whole numbers.")
    before = len(d)
    d = d.drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    if d.timestamp.duplicated().any():
        raise ValueError("Conflicting records share a timestamp.")
    intervals = d.timestamp.diff().dropna().dt.total_seconds()
    if len(intervals) and intervals.min() < 60:
        raise ValueError("Only one-minute candles are supported.")
    if len(intervals) and not (intervals == 60).any():
        raise ValueError("No consecutive one-minute candles found; check the interval.")
    if (d.timestamp + pd.Timedelta(minutes=1) > pd.Timestamp.now(tz="UTC")).any():
        raise ValueError("Only completed candles are accepted.")
    return d, {"rows": len(d), "duplicates_removed": before - len(d),
               "gaps": int((intervals > 60).sum())}


def acquire() -> None:
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ["pump_telegram.csv", "labeled_features/features_25S.csv.gz", "LICENSE"]:
        path = raw / Path(name).name
        if not path.exists():
            path.write_bytes(fetch(RESEARCH + name))
    signals = pd.read_csv(raw / "pump_telegram.csv")
    signals = signals[signals.exchange == "binance"].copy()
    signals["announced"] = pd.to_datetime(signals.date + " " + signals.hour, utc=True)
    known = signals[["symbol", "announced"]].drop_duplicates()
    known.to_csv(DATA / "known_announcements.csv", index=False)
    labeled = pd.read_csv(raw / "features_25S.csv.gz")
    events = labeled.loc[labeled["gt"] == 1, ["date", "pump_index", "symbol"]].copy()
    events["onset"] = pd.to_datetime(events.date, utc=True)
    events = events.drop_duplicates(["symbol", "onset"]).sort_values("onset")
    offsets = []
    for row in events.itertuples():
        candidates = known.loc[known.symbol == row.symbol, "announced"]
        offsets.append(float((candidates - row.onset).abs().min().total_seconds()) if len(candidates) else None)
    events["announcement_distance_seconds"] = offsets
    # Match source labels to an independently recorded Binance announcement.
    events["included"] = events.announcement_distance_seconds.le(300)
    events["event_id"] = events.pump_index.map(lambda n: f"event_{n:03d}")
    events["pair"] = events.symbol + "BTC"
    events.to_csv(DATA / "event_audit.csv", index=False)
    eligible = events[events.included].copy()
    jobs = set()
    for r in eligible.itertuples():
        jobs.add((r.pair, r.onset.strftime("%Y-%m-%d")))
        jobs.add((r.pair, (r.onset - pd.Timedelta(days=7)).strftime("%Y-%m-%d")))
    archive = raw / "binance"
    archive.mkdir(exist_ok=True)

    def download(job):
        pair, day = job
        name = f"{pair}-1m-{day}.zip"
        path = archive / name
        url = f"https://data.binance.vision/data/spot/daily/klines/{pair}/1m/{name}"
        record = {"pair": pair, "day": day, "url": url}
        try:
            if not path.exists():
                path.write_bytes(fetch(url))
            body = path.read_bytes()
            digest = hashlib.sha256(body).hexdigest()
            checksum_file = path.with_suffix(".zip.CHECKSUM")
            if not checksum_file.exists():
                checksum_file.write_bytes(fetch(url + ".CHECKSUM"))
            expected = checksum_file.read_text().split()[0]
            if expected != digest:
                raise ValueError("Archive checksum mismatch")
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                d = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLUMNS)
            d, quality = validate_candles(d)
            d["symbol"] = pair
            record.update(status="ok", sha256=digest, **quality)
            return record, d
        except Exception as exc:
            record.update(status="excluded", reason=str(exc))
            return record, None

    records, frames = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, j) for j in sorted(jobs)]
        for i, future in enumerate(as_completed(futures), 1):
            rec, frame = future.result()
            records.append(rec)
            if frame is not None:
                frames.append(frame)
            if i % 50 == 0 or i == len(futures):
                print(f"Audited {i}/{len(futures)} archives; usable {len(frames)}", flush=True)
    pd.DataFrame(records).sort_values(["pair", "day"]).to_csv(DATA / "download_manifest.csv", index=False)
    if not frames:
        raise RuntimeError("No usable raw candles. See download_manifest.csv")
    candles = pd.concat(frames).sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])
    candles.to_csv(DATA / "candles.csv.gz", index=False, compression="gzip")
    eligible.to_csv(DATA / "events.csv", index=False)
    audit = {"research_positive_rows": int((labeled['gt'] == 1).sum()),
             "binance_announcement_rows": len(signals), "unique_announcements": len(known),
             "matched_onsets": len(eligible), "unmatched_onsets": int((~events.included).sum()),
             "archives_requested": len(jobs), "archives_usable": len(frames),
             "candle_rows": len(candles), "coins": candles.symbol.nunique(),
             "first_candle": str(candles.timestamp.min()), "last_candle": str(candles.timestamp.max()),
             "label_policy": "Manually labeled 25-second onset mapped to its minute; onset minute and next two minutes positive. Other candles within 60 minutes of known onsets/announcements excluded from binary metrics.",
             "collection_policy": "Full UTC event day and same pair seven days earlier. First 60 minutes of each disconnected segment are warm-up; no gap filling.",
             "sources": [RESEARCH, "https://github.com/binance/binance-public-data"]}
    (DATA / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    acquire()
