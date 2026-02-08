# build_auction_snapshot.py
# ============================================================
# Build AUCTION snapshots from RAW (drafts/parquet + picks/parquet)
# - Mirrors the ADP "snapshot" pattern you already use
# - Outputs:
#     sleeper_dynasty_adp/data/snapshots/auction_price_series/season=YYYY/auction_price_series.parquet
#     sleeper_dynasty_adp/data/snapshots/auction_price_series/auction_price_series_ALL.parquet   (optional)
#     sleeper_dynasty_adp/data/snapshots/auction_draft_catalog/season=YYYY/auction_draft_catalog.parquet
#     sleeper_dynasty_adp/data/snapshots/auction_draft_catalog/auction_draft_catalog_ALL.parquet (optional)
#
# Notes:
# - Auction price comes from pick.metadata.amount -> stored in RAW picks as md_amount
# - Time axis is anchored on DRAFT start_time -> start_dt/start_month (safe conversion)
# - This script does NOT require API calls; it operates purely on local RAW parquet files
# ============================================================

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
ROOT_DIR = "sleeper_dynasty_adp"
SEASONS = [2026]  # set seasons you want to build
KEEP_DYNASTY_CLASSES = {"startup", "rookie"}  # same idea as ADP

DIR_RAW_DRAFTS = os.path.join(ROOT_DIR, "data", "raw", "drafts")
DIR_RAW_PICKS  = os.path.join(ROOT_DIR, "data", "raw", "picks")

DIR_SNAP_AUCTION_TS  = os.path.join(ROOT_DIR, "data", "snapshots", "auction_price_series")
DIR_SNAP_AUCTION_CAT = os.path.join(ROOT_DIR, "data", "snapshots", "auction_draft_catalog")

os.makedirs(DIR_SNAP_AUCTION_TS, exist_ok=True)
os.makedirs(DIR_SNAP_AUCTION_CAT, exist_ok=True)


# -----------------------------
# SAFE TIME CONVERSION (fixes overflow)
# -----------------------------
def safe_ms_to_datetime_utc(ms_series: pd.Series, *, label: str = "start_time") -> pd.Series:
    """
    Convert epoch-ms -> UTC datetime safely.
    Masks out-of-range ms values so pandas/numpy never overflows internally.
    """
    s = pd.to_numeric(ms_series, errors="coerce")

    lower = pd.Timestamp("2010-01-01", tz="UTC").value // 1_000_000  # ns -> ms
    upper = pd.Timestamp("2036-12-31", tz="UTC").value // 1_000_000

    bad = s.notna() & ((s < lower) | (s > upper))
    if int(bad.sum()) > 0:
        examples = ms_series[bad].head(5).tolist()
        print(f"[warn] {label}: found {int(bad.sum()):,} out-of-range ms values. Examples: {examples}")

    s = s.mask(bad, np.nan)
    return pd.to_datetime(s, unit="ms", utc=True, errors="coerce")


# -----------------------------
# IO helpers
# -----------------------------
def season_raw_paths(season: int) -> Tuple[str, str]:
    drafts_path = os.path.join(DIR_RAW_DRAFTS, f"drafts_{season}.parquet")
    picks_path  = os.path.join(DIR_RAW_PICKS,  f"picks_{season}.parquet")
    return drafts_path, picks_path


def load_raw_season(season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    drafts_path, picks_path = season_raw_paths(season)

    if not os.path.exists(drafts_path):
        raise FileNotFoundError(f"Missing RAW drafts parquet: {drafts_path}")
    if not os.path.exists(picks_path):
        raise FileNotFoundError(f"Missing RAW picks parquet:  {picks_path}")

    drafts = pd.read_parquet(drafts_path)
    picks  = pd.read_parquet(picks_path)

    # normalize id types
    for c in ["draft_id", "league_id"]:
        if c in drafts.columns:
            drafts[c] = drafts[c].astype(str)

    if "draft_id" in picks.columns:
        picks["draft_id"] = picks["draft_id"].astype(str)
    if "player_id" in picks.columns:
        picks["player_id"] = picks["player_id"].astype(str)

    return drafts, picks


# -----------------------------
# Catalog building (similar to 01)
# -----------------------------
def build_draft_catalog(drafts_df: pd.DataFrame) -> pd.DataFrame:
    df = drafts_df.copy()

    for c in ["draft_id", "league_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    num_cols = [
        "created", "start_time", "last_picked",
        "st_teams", "st_rounds", "st_pick_timer", "st_reversal_round",
        "st_slots_qb", "st_slots_rb", "st_slots_wr", "st_slots_te",
        "st_slots_flex", "st_slots_super_flex", "st_slots_def", "st_slots_k",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["start_dt"] = safe_ms_to_datetime_utc(df.get("start_time", pd.Series([], dtype="float64")), label="start_time")
    df["start_date"] = df["start_dt"].dt.date.astype("string")
    df["start_month"] = df["start_dt"].dt.strftime("%Y-%m").astype("string")

    df["is_dynasty"] = df.get("md_scoring_type", "").astype(str).str.contains("dynasty", case=False, na=False)
    df["is_superflex"] = (df.get("st_slots_super_flex", 0).fillna(0) > 0) | df.get("md_scoring_type", "").astype(str).str.contains(
        "2qb|superflex", case=False, na=False
    )

    def _dynasty_class(row) -> str:
        if not bool(row.get("is_dynasty", False)):
            return "non_dynasty"
        rounds = row.get("st_rounds", np.nan)
        if pd.notna(rounds) and rounds <= 6:
            return "rookie"
        if pd.notna(rounds) and rounds >= 14:
            return "startup"
        return "other"

    df["dynasty_class"] = df.apply(_dynasty_class, axis=1)

    return df


def build_auction_draft_catalog(draft_catalog: pd.DataFrame) -> pd.DataFrame:
    if draft_catalog.empty:
        return pd.DataFrame()
    if "type" not in draft_catalog.columns:
        return pd.DataFrame()
    return draft_catalog[draft_catalog["type"].astype(str).str.lower() == "auction"].copy()


# -----------------------------
# Auction price series
# -----------------------------
def compute_auction_price_series(
    picks_df: pd.DataFrame,
    auction_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate auction prices by month (start_month) + player_id + format fields.
    Requires picks_df to have md_amount (written by 01_ingest_historical.py).
    """
    if picks_df.empty or auction_catalog.empty:
        return pd.DataFrame()

    p = picks_df.copy()

    for c in ["draft_id", "player_id"]:
        if c in p.columns:
            p[c] = p[c].astype(str)

    # md_amount should exist if you ingested auction metadata in 01
    if "md_amount" not in p.columns:
        print("[warn] picks_df has no md_amount column. Did you update 01_ingest_historical.py and re-run ingestion?")
        return pd.DataFrame()

    p["amount"] = pd.to_numeric(p["md_amount"], errors="coerce")

    d = auction_catalog.copy()
    d["draft_id"] = d["draft_id"].astype(str)

    keep_cols = [
        "draft_id", "season", "start_dt", "start_month",
        "dynasty_class", "type", "md_scoring_type", "st_teams", "st_rounds", "is_superflex",
        "draft_status"
    ]
    keep_cols = [c for c in keep_cols if c in d.columns]

    m = p.merge(d[keep_cols], on="draft_id", how="inner")

    # only completed auction drafts (mirrors how you use picks)
    status_col = "draft_status" if "draft_status" in m.columns else "status"
    if status_col in m.columns:
        m = m[m[status_col].astype(str).str.lower() == "complete"].copy()

    m = m[m["player_id"].notna()].copy()
    m = m[m["start_month"].notna()].copy()
    m = m[m["amount"].notna()].copy()

    if "dynasty_class" in m.columns:
        m = m[m["dynasty_class"].isin(list(KEEP_DYNASTY_CLASSES))].copy()

    out = (
        m.groupby(
            ["season", "start_month", "player_id", "dynasty_class", "md_scoring_type", "st_teams", "st_rounds", "is_superflex"],
            dropna=False
        )
        .agg(
            drafts=("draft_id", "nunique"),
            sales=("amount", "size"),
            avg_price=("amount", "mean"),
            med_price=("amount", "median"),
            min_price=("amount", "min"),
            max_price=("amount", "max"),
        )
        .reset_index()
    )

    for c in ["avg_price", "med_price", "min_price", "max_price"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    out["drafts"] = pd.to_numeric(out["drafts"], errors="coerce").fillna(0).astype(int)
    out["sales"] = pd.to_numeric(out["sales"], errors="coerce").fillna(0).astype(int)

    return out


# -----------------------------
# Writer helpers
# -----------------------------
def write_season_parquet(df: pd.DataFrame, base_dir: str, season: int, filename: str) -> str:
    out_dir = os.path.join(base_dir, f"season={season}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_parquet(path, index=False)
    return path


# -----------------------------
# MAIN
# -----------------------------
all_auc_ts_parts: List[pd.DataFrame] = []
all_auc_cat_parts: List[pd.DataFrame] = []

for season in SEASONS:
    print(f"\n=== BUILD AUCTION SNAPSHOT season={season} ===")

    drafts, picks = load_raw_season(season)
    if drafts.empty:
        print(f"[season={season}] drafts empty. Skipping.")
        continue

    draft_catalog = build_draft_catalog(drafts)
    auction_catalog = build_auction_draft_catalog(draft_catalog)

    if auction_catalog.empty:
        print(f"[season={season}] No auction drafts found in RAW drafts_{season}.parquet")
        continue

    # write auction catalog snapshot
    cat_path = write_season_parquet(
        auction_catalog,
        DIR_SNAP_AUCTION_CAT,
        season,
        "auction_draft_catalog.parquet",
    )
    print(f"[season={season}] wrote auction catalog -> {cat_path} | shape={auction_catalog.shape}")
    all_auc_cat_parts.append(auction_catalog)

    # compute & write auction price series
    auc_ts = compute_auction_price_series(picks, auction_catalog)
    if auc_ts.empty:
        print(f"[season={season}] auction price series empty. (No completed auction picks with md_amount?)")
        continue

    ts_path = write_season_parquet(
        auc_ts,
        DIR_SNAP_AUCTION_TS,
        season,
        "auction_price_series.parquet",
    )
    print(f"[season={season}] wrote auction price series -> {ts_path} | shape={auc_ts.shape}")
    all_auc_ts_parts.append(auc_ts)

# combined outputs
if all_auc_ts_parts:
    auc_all = pd.concat(all_auc_ts_parts, ignore_index=True)
    auc_all_path = os.path.join(DIR_SNAP_AUCTION_TS, "auction_price_series_ALL.parquet")
    auc_all.to_parquet(auc_all_path, index=False)
    print(f"\n[OK] wrote combined auction price series -> {auc_all_path} | shape={auc_all.shape}")

if all_auc_cat_parts:
    ac_all = pd.concat(all_auc_cat_parts, ignore_index=True)
    ac_all_path = os.path.join(DIR_SNAP_AUCTION_CAT, "auction_draft_catalog_ALL.parquet")
    ac_all.to_parquet(ac_all_path, index=False)
    print(f"[OK] wrote combined auction draft catalog -> {ac_all_path} | shape={ac_all.shape}")
