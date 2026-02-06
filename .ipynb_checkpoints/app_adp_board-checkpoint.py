# app_adp_board.py
# ============================================================
# Interactive Sleeper Dynasty ADP Board (RAW-first)
# - Loads drafts/leagues/picks from data/raw by season
# - Optional: writes snapshots under data/snapshots
# - Interactive board + player distribution + monthly ADP trend
# - Robust start_time parsing (fixes overflow)
# - Adds PPG using Sleeper stats endpoint
# ============================================================

import os
import time
import math
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Sleeper ADP Board", layout="wide")

# ----------------------------
# Sleeper API
# ----------------------------
BASE = "https://api.sleeper.app/v1"
session = requests.Session()
session.headers.update({"User-Agent": "Sleeper-Dynasty-ADP/1.0"})


def get_json(url: str, timeout: int = 30, retries: int = 4, backoff: float = 1.8) -> Any:
    last_err = None
    for i in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 429:
                time.sleep(min(30, (backoff ** i) + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(min(30, (backoff ** i) + 0.5))
    raise RuntimeError(f"GET failed: {url}\nLast error: {last_err}")


def url_players_nfl() -> str:
    return f"{BASE}/players/nfl"


def url_stats_nfl_regular(season: int) -> str:
    return f"{BASE}/stats/nfl/regular/{season}"


# ----------------------------
# Path helpers (ROBUST)
# ----------------------------
def find_data_root_candidates(start_dir: str) -> List[str]:
    """
    Search upwards from start_dir for a folder that contains data/raw.
    Also check for a nested sleeper_dynasty_adp/data/raw.
    """
    candidates = []
    cur = os.path.abspath(start_dir)

    for _ in range(8):  # up to 8 levels
        # direct: <cur>/data/raw
        dr = os.path.join(cur, "data", "raw")
        if os.path.isdir(dr):
            candidates.append(os.path.join(cur, "data"))

        # nested: <cur>/sleeper_dynasty_adp/data/raw
        nested = os.path.join(cur, "sleeper_dynasty_adp", "data", "raw")
        if os.path.isdir(nested):
            candidates.append(os.path.join(cur, "sleeper_dynasty_adp", "data"))

        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # de-dup while preserving order
    out = []
    seen = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def pick_best_data_dir() -> Tuple[str, str, str]:
    """
    Choose a data directory. Default is auto-detected based on this file location.
    Returns: (project_root, raw_dir, snapshots_dir)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = find_data_root_candidates(here)

    if not candidates:
        # fallback: assume repo structure beside this script
        # (user can override in sidebar)
        project_root = os.path.abspath(os.path.join(here, ".."))
        data_dir = os.path.join(project_root, "data")
    else:
        data_dir = candidates[0]
        project_root = os.path.dirname(data_dir)

    raw_dir = os.path.join(data_dir, "raw")
    snapshots_dir = os.path.join(data_dir, "snapshots")
    return project_root, raw_dir, snapshots_dir


# ----------------------------
# Time parsing (FIX overflow)
# ----------------------------
def add_time_columns(drafts: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly add start_dt/start_date/start_month from start_time (ms epoch).
    Avoids pandas/numpy overflow when start_time is float/scientific notation.
    """
    df = drafts.copy()

    # Prefer start_time; fallback to created
    if "start_time" in df.columns:
        ts = df["start_time"]
    elif "created" in df.columns:
        ts = df["created"]
    else:
        df["start_dt"] = pd.NaT
        df["start_date"] = pd.Series(pd.NA, index=df.index, dtype="string")
        df["start_month"] = pd.Series(pd.NA, index=df.index, dtype="string")
        return df

    ms = pd.to_numeric(ts, errors="coerce")

    # plausible ms epoch range (2000 to 2035)
    lo = 946684800000
    hi = 2051222400000
    ms = ms.where((ms >= lo) & (ms <= hi))

    # round then nullable int
    ms_int = ms.round().astype("Int64")

    # convert to datetime (masking invalid)
    start_dt = pd.to_datetime(ms_int.astype("float64"), unit="ms", utc=True, errors="coerce")

    df["start_dt"] = start_dt
    df["start_date"] = df["start_dt"].dt.strftime("%Y-%m-%d").astype("string")
    df["start_month"] = df["start_dt"].dt.strftime("%Y-%m").astype("string")

    return df


# ----------------------------
# Loading RAW season files
# ----------------------------
def season_raw_paths(raw_dir: str, season: int) -> Tuple[str, str, str]:
    """
    RAW file layout you have:
      data/raw/drafts/drafts_YYYY.parquet
      data/raw/leagues/leagues_YYYY.parquet
      data/raw/picks/picks_YYYY.parquet
    """
    drafts_path = os.path.join(raw_dir, "drafts", f"drafts_{season}.parquet")
    leagues_path = os.path.join(raw_dir, "leagues", f"leagues_{season}.parquet")
    picks_path = os.path.join(raw_dir, "picks", f"picks_{season}.parquet")
    return drafts_path, picks_path, leagues_path


@st.cache_data(show_spinner=False)
def load_raw_season(raw_dir: str, season: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drafts_path, picks_path, leagues_path = season_raw_paths(raw_dir, season)

    if not os.path.exists(drafts_path):
        raise FileNotFoundError(f"Missing RAW drafts parquet:\n{drafts_path}")
    if not os.path.exists(picks_path):
        raise FileNotFoundError(f"Missing RAW picks parquet:\n{picks_path}")
    if not os.path.exists(leagues_path):
        # leagues are useful, but not strictly required for most app features
        leagues = pd.DataFrame()
    else:
        leagues = pd.read_parquet(leagues_path)

    drafts = pd.read_parquet(drafts_path)
    picks = pd.read_parquet(picks_path)

    # normalize ids
    for col in ["draft_id", "league_id"]:
        if col in drafts.columns:
            drafts[col] = drafts[col].astype(str)
    if "draft_id" in picks.columns:
        picks["draft_id"] = picks["draft_id"].astype(str)
    if not leagues.empty:
        if "league_id" in leagues.columns:
            leagues["league_id"] = leagues["league_id"].astype(str)

    drafts = add_time_columns(drafts)

    return drafts, picks, leagues


def ensure_snapshot_dirs(snapshots_dir: str) -> None:
    os.makedirs(snapshots_dir, exist_ok=True)
    os.makedirs(os.path.join(snapshots_dir, "drafts"), exist_ok=True)
    os.makedirs(os.path.join(snapshots_dir, "picks"), exist_ok=True)
    os.makedirs(os.path.join(snapshots_dir, "leagues"), exist_ok=True)


def write_snapshots_for_season(
    snapshots_dir: str,
    season: int,
    drafts: pd.DataFrame,
    picks: pd.DataFrame,
    leagues: pd.DataFrame,
) -> Tuple[str, str, str]:
    ensure_snapshot_dirs(snapshots_dir)
    ddir = os.path.join(snapshots_dir, "drafts", f"season={season}")
    pdir = os.path.join(snapshots_dir, "picks", f"season={season}")
    ldir = os.path.join(snapshots_dir, "leagues", f"season={season}")
    os.makedirs(ddir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(ldir, exist_ok=True)

    drafts_path = os.path.join(ddir, "drafts.parquet")
    picks_path = os.path.join(pdir, "picks.parquet")
    leagues_path = os.path.join(ldir, "leagues.parquet")

    drafts.to_parquet(drafts_path, index=False)
    picks.to_parquet(picks_path, index=False)
    if not leagues.empty:
        leagues.to_parquet(leagues_path, index=False)

    return drafts_path, picks_path, leagues_path


# ----------------------------
# Players + PPG
# ----------------------------
@st.cache_data(show_spinner=False)
def load_players_df() -> pd.DataFrame:
    players = get_json(url_players_nfl())
    players_raw = (
        pd.DataFrame.from_dict(players, orient="index")
        .reset_index()
        .rename(columns={"index": "sleeper_player_id"})
    )
    if "player_id" in players_raw.columns:
        players_raw["player_id"] = players_raw["player_id"].where(
            players_raw["player_id"].notna(), players_raw["sleeper_player_id"]
        )
    else:
        players_raw["player_id"] = players_raw["sleeper_player_id"]

    players_raw["player_id"] = players_raw["player_id"].astype(str)

    want = ["player_id", "full_name", "position", "team", "years_exp", "rookie_year", "status"]
    have = [c for c in want if c in players_raw.columns]
    df = players_raw[have].copy()

    if "years_exp" in df.columns:
        df["years_exp"] = pd.to_numeric(df["years_exp"], errors="coerce")
    if "rookie_year" in df.columns:
        df["rookie_year"] = pd.to_numeric(df["rookie_year"], errors="coerce")

    # normalize cols if missing
    if "full_name" not in df.columns and "first_name" in players_raw.columns and "last_name" in players_raw.columns:
        df["full_name"] = (players_raw["first_name"].fillna("") + " " + players_raw["last_name"].fillna("")).str.strip()

    return df


def safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df)), index=df.index, dtype="float64")


def calc_fantasy_points(df: pd.DataFrame, scoring: Dict[str, float]) -> pd.Series:
    pts = pd.Series(np.zeros(len(df)), index=df.index, dtype="float64")
    for stat, w in scoring.items():
        pts = pts + safe_col(df, stat) * float(w)
    return pts


@st.cache_data(show_spinner=False)
def load_ppg(season: int) -> pd.DataFrame:
    # scoring baseline (you can tweak later)
    SCORING = {
        "pass_td": 4.0,
        "rush_td": 6.0,
        "rec_td": 6.0,
        "pass_yd": 0.04,
        "rush_yd": 0.10,
        "rec_yd": 0.10,
        "rec": 1.0,
        "pass_int": -2.0,
        "fum_lost": -2.0,
        "pass_2pt": 2.0,
        "rush_2pt": 2.0,
        "rec_2pt": 2.0,
    }

    data = get_json(url_stats_nfl_regular(season))
    if not isinstance(data, dict) or len(data) == 0:
        return pd.DataFrame(columns=["player_id", "ppg"])

    stats_df = pd.DataFrame.from_dict(data, orient="index").reset_index().rename(columns={"index": "player_id"})
    stats_df["player_id"] = stats_df["player_id"].astype(str)

    players_df = load_players_df().copy()
    players_df["player_id"] = players_df["player_id"].astype(str)

    merged = players_df.merge(stats_df, on="player_id", how="left")

    merged["fantasy_pts"] = calc_fantasy_points(merged, SCORING)

    if "gp" in merged.columns:
        gp = pd.to_numeric(merged["gp"], errors="coerce")
    elif "g" in merged.columns:
        gp = pd.to_numeric(merged["g"], errors="coerce")
    else:
        gp = pd.Series(np.nan, index=merged.index)

    merged["games_played"] = gp.replace(0, np.nan)
    merged["ppg"] = merged["fantasy_pts"] / merged["games_played"]

    out = merged[["player_id", "ppg", "games_played", "fantasy_pts"]].copy()
    out["ppg"] = pd.to_numeric(out["ppg"], errors="coerce")
    return out


# ----------------------------
# Core computations
# ----------------------------
def infer_pick_no(picks: pd.DataFrame) -> pd.Series:
    """
    Try to get overall pick number from common Sleeper pick exports.
    """
    for c in ["pick_no", "overall_pick", "pick", "draft_pick", "pick_number"]:
        if c in picks.columns:
            s = pd.to_numeric(picks[c], errors="coerce")
            if s.notna().any():
                return s
    # fallback: reconstruct from round + draft_slot if possible
    if "round" in picks.columns and "draft_slot" in picks.columns:
        r = pd.to_numeric(picks["round"], errors="coerce")
        ds = pd.to_numeric(picks["draft_slot"], errors="coerce")
        # this is not perfect without league size, but better than nothing:
        return (r - 1) * 12 + ds
    return pd.Series(np.nan, index=picks.index)


def compute_player_pick_stats(
    picks: pd.DataFrame,
    players_df: pd.DataFrame,
    drafts_filtered: pd.DataFrame,
    ppg_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Returns a player-level pool with:
      player_id, full_name, team, position, ppg,
      drafts, picks, adp, min_pick, max_pick
    """
    p = picks.copy()

    if "draft_id" not in p.columns or "player_id" not in p.columns:
        raise RuntimeError("picks parquet must include at least: draft_id, player_id")

    p["draft_id"] = p["draft_id"].astype(str)
    p["player_id"] = p["player_id"].astype(str)

    # limit to filtered drafts
    keep_draft_ids = set(drafts_filtered["draft_id"].astype(str).unique())
    p = p[p["draft_id"].isin(keep_draft_ids)].copy()

    # pick number
    p["pick_no_calc"] = infer_pick_no(p)
    p["pick_no_calc"] = pd.to_numeric(p["pick_no_calc"], errors="coerce")
    p = p[p["pick_no_calc"].notna()].copy()

    # attach player metadata
    pl = players_df.copy()
    pl["player_id"] = pl["player_id"].astype(str)
    df = p.merge(pl, on="player_id", how="left")

    # keep skill positions
    if "position" in df.columns:
        df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()

    # compute pool stats
    out = (
        df.groupby("player_id", as_index=False)
          .agg(
              picks=("pick_no_calc", "size"),
              drafts=("draft_id", pd.Series.nunique),
              adp=("pick_no_calc", "mean"),
              min_pick=("pick_no_calc", "min"),
              max_pick=("pick_no_calc", "max"),
          )
    )

    # add player meta back (full_name/team/position)
    meta_cols = [c for c in ["player_id", "full_name", "team", "position", "years_exp"] if c in pl.columns]
    meta = pl[meta_cols].drop_duplicates("player_id")
    out = out.merge(meta, on="player_id", how="left")

    # PPG merge
    if ppg_df is not None and not ppg_df.empty:
        out = out.merge(ppg_df[["player_id", "ppg", "games_played", "fantasy_pts"]], on="player_id", how="left")

    out["adp"] = pd.to_numeric(out["adp"], errors="coerce").round(2)
    out["min_pick"] = pd.to_numeric(out["min_pick"], errors="coerce")
    out["max_pick"] = pd.to_numeric(out["max_pick"], errors="coerce")
    out["picks"] = pd.to_numeric(out["picks"], errors="coerce").fillna(0).astype(int)
    out["drafts"] = pd.to_numeric(out["drafts"], errors="coerce").fillna(0).astype(int)

    # cleanup
    out = out[out["drafts"] > 0].copy()
    out = out.sort_values("adp").reset_index(drop=True)
    return out


def add_pos_rank(pool: pd.DataFrame) -> pd.DataFrame:
    df = pool.copy()
    df["adp"] = pd.to_numeric(df["adp"], errors="coerce")
    df["position"] = df["position"].astype(str)
    df = df.sort_values(["position", "adp"], ascending=[True, True]).reset_index(drop=True)
    df["pos_rank"] = df.groupby("position").cumcount() + 1
    return df


def snake_picks(num_teams: int, num_rounds: int) -> List[Dict[str, int]]:
    picks = []
    for r in range(1, num_rounds + 1):
        for pick_in_round in range(1, num_teams + 1):
            team = pick_in_round if (r % 2 == 1) else (num_teams - pick_in_round + 1)
            picks.append({"round": r, "pick_in_round": pick_in_round, "team": team})
    return picks


def build_snake_board(pool: pd.DataFrame, num_teams: int, num_rounds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      board_wide: index Round 1.., columns Team 1.. with pretty labels
      board_long: Pick, Player, Team, Position, PositionalRank, ADP, Min, Max, Drafts, Picks, PPG
    """
    need = num_teams * num_rounds
    ranked = pool.sort_values("adp").head(need).reset_index(drop=True)

    board = pd.DataFrame(
        index=[f"Round {r}" for r in range(1, num_rounds + 1)],
        columns=[f"Team {t}" for t in range(1, num_teams + 1)],
    )

    long_rows = []
    pick_map = snake_picks(num_teams, num_rounds)

    for i, pck in enumerate(pick_map):
        if i >= len(ranked):
            break
        row = ranked.iloc[i]
        r = pck["round"]
        team_slot = pck["team"]
        pir = pck["pick_in_round"]
        pick_label = f"{r}.{pir:02d}"

        # pretty cell label (keeps your old style inside board)
        cell = f"{pick_label} {row.get('full_name','')}, {row.get('team','')} {row.get('position','')} ({int(row.get('pos_rank',0) or 0)})"
        board.loc[f"Round {r}", f"Team {team_slot}"] = cell

        long_rows.append({
            "Pick": pick_label,
            "player_id": row["player_id"],
            "Player": row.get("full_name", ""),
            "Team": row.get("team", ""),
            "Position": row.get("position", ""),
            "PositionalRank": int(row.get("pos_rank", 0) or 0),
            "ADP": float(row.get("adp", np.nan)) if pd.notna(row.get("adp", np.nan)) else np.nan,
            "MinPick": float(row.get("min_pick", np.nan)) if pd.notna(row.get("min_pick", np.nan)) else np.nan,
            "MaxPick": float(row.get("max_pick", np.nan)) if pd.notna(row.get("max_pick", np.nan)) else np.nan,
            "Drafts": int(row.get("drafts", 0) or 0),
            "Picks": int(row.get("picks", 0) or 0),
            "PPG": float(row.get("ppg", np.nan)) if pd.notna(row.get("ppg", np.nan)) else np.nan,
        })

    board_long = pd.DataFrame(long_rows)
    return board, board_long


def build_linear_board(pool: pd.DataFrame, num_teams: int, num_rounds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    need = num_teams * num_rounds
    ranked = pool.sort_values("adp").head(need).reset_index(drop=True)

    board = pd.DataFrame(
        index=[f"Round {r}" for r in range(1, num_rounds + 1)],
        columns=[f"Team {t}" for t in range(1, num_teams + 1)],
    )

    long_rows = []
    idx = 0
    for r in range(1, num_rounds + 1):
        for t in range(1, num_teams + 1):
            if idx >= len(ranked):
                break
            row = ranked.iloc[idx]
            pick_label = f"{r}.{t:02d}"
            cell = f"{pick_label} {row.get('full_name','')}, {row.get('team','')} {row.get('position','')} ({int(row.get('pos_rank',0) or 0)})"
            board.loc[f"Round {r}", f"Team {t}"] = cell

            long_rows.append({
                "Pick": pick_label,
                "player_id": row["player_id"],
                "Player": row.get("full_name", ""),
                "Team": row.get("team", ""),
                "Position": row.get("position", ""),
                "PositionalRank": int(row.get("pos_rank", 0) or 0),
                "ADP": float(row.get("adp", np.nan)) if pd.notna(row.get("adp", np.nan)) else np.nan,
                "MinPick": float(row.get("min_pick", np.nan)) if pd.notna(row.get("min_pick", np.nan)) else np.nan,
                "MaxPick": float(row.get("max_pick", np.nan)) if pd.notna(row.get("max_pick", np.nan)) else np.nan,
                "Drafts": int(row.get("drafts", 0) or 0),
                "Picks": int(row.get("picks", 0) or 0),
                "PPG": float(row.get("ppg", np.nan)) if pd.notna(row.get("ppg", np.nan)) else np.nan,
            })

            idx += 1

    board_long = pd.DataFrame(long_rows)
    return board, board_long


def filter_drafts(
    drafts: pd.DataFrame,
    leagues: pd.DataFrame,
    season: int,
    draft_status: List[str],
    draft_type: List[str],
    scoring_types: List[str],
    league_sizes: List[int],
    min_rounds: Optional[int],
    max_rounds: Optional[int],
    date_min: Optional[pd.Timestamp],
    date_max: Optional[pd.Timestamp],
    te_premium_only: bool,
) -> pd.DataFrame:
    df = drafts.copy()

    # season
    if "season" in df.columns:
        df = df[pd.to_numeric(df["season"], errors="coerce") == season].copy()

    # status/type/scoring/size
    if "draft_status" in df.columns and draft_status:
        df = df[df["draft_status"].isin(draft_status)].copy()

    if "type" in df.columns and draft_type:
        df = df[df["type"].isin(draft_type)].copy()

    if "md_scoring_type" in df.columns and scoring_types:
        df = df[df["md_scoring_type"].isin(scoring_types)].copy()

    if "st_teams" in df.columns and league_sizes:
        df["st_teams"] = pd.to_numeric(df["st_teams"], errors="coerce")
        df = df[df["st_teams"].isin(league_sizes)].copy()

    if "st_rounds" in df.columns:
        df["st_rounds"] = pd.to_numeric(df["st_rounds"], errors="coerce")
        if min_rounds is not None:
            df = df[df["st_rounds"] >= min_rounds].copy()
        if max_rounds is not None:
            df = df[df["st_rounds"] <= max_rounds].copy()

    # date filter
    if "start_dt" in df.columns:
        if date_min is not None:
            df = df[df["start_dt"] >= date_min].copy()
        if date_max is not None:
            df = df[df["start_dt"] <= date_max].copy()

    # TE Premium filter (from leagues: scoring_settings.bonus_rec_te)
    if te_premium_only and (not leagues.empty) and ("league_id" in df.columns) and ("league_id" in leagues.columns):
        # find a plausible TE premium column
        te_cols = [c for c in leagues.columns if c.endswith("scoring_settings.bonus_rec_te") or c == "scoring_settings.bonus_rec_te"]
        if te_cols:
            te_col = te_cols[0]
            lg = leagues[["league_id", te_col]].copy()
            lg[te_col] = pd.to_numeric(lg[te_col], errors="coerce").fillna(0)
            df = df.merge(lg, on="league_id", how="left")
            df = df[pd.to_numeric(df[te_col], errors="coerce").fillna(0) > 0].copy()

    return df


def player_monthly_trend(picks: pd.DataFrame, drafts_filtered: pd.DataFrame, player_id: str, last_n_months: int = 5) -> pd.DataFrame:
    if "draft_id" not in picks.columns or "player_id" not in picks.columns:
        return pd.DataFrame(columns=["start_month", "adp"])

    d = drafts_filtered[["draft_id", "start_month", "start_dt"]].copy()
    d["draft_id"] = d["draft_id"].astype(str)

    p = picks.copy()
    p["draft_id"] = p["draft_id"].astype(str)
    p["player_id"] = p["player_id"].astype(str)

    p = p[p["player_id"] == str(player_id)].copy()
    p = p.merge(d, on="draft_id", how="inner")

    p["pick_no_calc"] = infer_pick_no(p)
    p["pick_no_calc"] = pd.to_numeric(p["pick_no_calc"], errors="coerce")
    p = p[p["pick_no_calc"].notna()].copy()
    p = p[p["start_month"].notna()].copy()

    # last N months present
    months = sorted([m for m in p["start_month"].unique() if m and m != "nan"])
    if not months:
        return pd.DataFrame(columns=["start_month", "adp", "picks"])
    keep = months[-last_n_months:]

    agg = (
        p[p["start_month"].isin(keep)]
        .groupby("start_month", as_index=False)
        .agg(adp=("pick_no_calc", "mean"), picks=("pick_no_calc", "size"))
    )
    agg["adp"] = pd.to_numeric(agg["adp"], errors="coerce").round(2)
    agg = agg.sort_values("start_month")
    return agg


# ============================================================
# UI
# ============================================================
st.title("Sleeper Dynasty ADP Board (Interactive)")

# Resolve data dirs (auto)
project_root_auto, raw_dir_auto, snapshots_dir_auto = pick_best_data_dir()

with st.sidebar:
    st.header("Data / Paths")

    st.write(f"**Auto project root:** {project_root_auto}")
    st.write(f"**Auto raw dir:** {raw_dir_auto}")
    st.write(f"**Auto snapshots dir:** {snapshots_dir_auto}")

    st.caption("If auto-detection is wrong, override below (point to the folder that contains raw/drafts, raw/picks, raw/leagues).")

    raw_override = st.text_input(
        "Override RAW dir (optional)",
        value=raw_dir_auto,
        help="Example: C:\\Users\\lgilb\\fantasyfootball\\sleeper_dynasty_adp\\scripts_or_notebooks\\sleeper_dynasty_adp\\data\\raw",
    )
    raw_dir = raw_override.strip()

    snapshots_override = st.text_input(
        "Override snapshots dir (optional)",
        value=snapshots_dir_auto,
        help="Example: ...\\data\\snapshots",
    )
    snapshots_dir = snapshots_override.strip()

    write_snaps = st.checkbox("Write snapshots for selected season", value=False)

    st.divider()
    st.header("Board settings")

    season = st.number_input("Season", min_value=2015, max_value=2030, value=2026, step=1)
    board_kind = st.selectbox("Board type", ["Startup (snake)", "Rookie (linear)"], index=0)

    num_teams = st.number_input("League size", min_value=4, max_value=32, value=12, step=1)
    if board_kind.startswith("Startup"):
        num_rounds = st.number_input("Rounds", min_value=1, max_value=60, value=35, step=1)
    else:
        num_rounds = st.number_input("Rounds", min_value=1, max_value=20, value=5, step=1)

    st.divider()
    st.header("Filters")

    # load drafts just to populate filter choices
    # (we'll load full data in main below)
    filter_draft_status = st.multiselect("Draft status", ["complete", "pre_draft", "drafting", "paused"], default=["complete"])
    filter_draft_type = st.multiselect("Draft type", ["snake", "linear", "auction"], default=["snake", "linear"])
    # scoring types from your sample (safe defaults)
    filter_scoring = st.multiselect(
        "Scoring type (md_scoring_type)",
        ["dynasty_2qb", "dynasty_ppr", "dynasty_half_ppr", "dynasty_std", "2qb", "ppr", "half_ppr", "std", "idp", "idp_1qb"],
        default=["dynasty_2qb"],
    )
    te_premium_only = st.checkbox("TE Premium only (league scoring_settings.bonus_rec_te > 0)", value=False)

    min_rounds = st.number_input("Min st_rounds (optional)", min_value=0, max_value=80, value=0, step=1)
    max_rounds = st.number_input("Max st_rounds (optional)", min_value=0, max_value=80, value=0, step=1)

    st.caption("Set Min/Max to 0 to disable")
    if min_rounds == 0:
        min_rounds_val = None
    else:
        min_rounds_val = int(min_rounds)
    if max_rounds == 0:
        max_rounds_val = None
    else:
        max_rounds_val = int(max_rounds)

    ppg_season = st.number_input("PPG season (regular)", min_value=2015, max_value=2030, value=2025, step=1)

# Main load
try:
    drafts, picks, leagues = load_raw_season(raw_dir, int(season))
except Exception as e:
    st.error(f"Failed to load RAW season files.\n\n{e}")
    st.stop()

# Optionally write snapshots
if write_snaps:
    try:
        dpath, ppath, lpath = write_snapshots_for_season(snapshots_dir, int(season), drafts, picks, leagues)
        st.success("Snapshots written.")
        st.caption(f"Drafts: {dpath}")
        st.caption(f"Picks:  {ppath}")
        if leagues is not None and not leagues.empty:
            st.caption(f"Leagues:{lpath}")
    except Exception as e:
        st.warning(f"Could not write snapshots:\n{e}")

# Date range slider (derive from drafts)
drafts_valid_dt = drafts[drafts["start_dt"].notna()].copy()
if len(drafts_valid_dt) > 0:
    dt_min = drafts_valid_dt["start_dt"].min()
    dt_max = drafts_valid_dt["start_dt"].max()
else:
    dt_min = pd.Timestamp("2000-01-01", tz="UTC")
    dt_max = pd.Timestamp("2000-01-02", tz="UTC")

# Streamlit date_input uses date objects
default_start = dt_min.date() if isinstance(dt_min, pd.Timestamp) else date(2000, 1, 1)
default_end = dt_max.date() if isinstance(dt_max, pd.Timestamp) else date(2000, 1, 2)

c1, c2, c3 = st.columns([2, 2, 6])
with c1:
    date_from = st.date_input("Drafts from (date)", value=default_start)
with c2:
    date_to = st.date_input("Drafts to (date)", value=default_end)
with c3:
    st.caption("Date filters use **draft start_time** (not file timestamps).")

date_min = pd.Timestamp(datetime.combine(date_from, datetime.min.time()), tz="UTC")
date_max = pd.Timestamp(datetime.combine(date_to, datetime.max.time()), tz="UTC")

# Filter drafts
league_sizes = [int(num_teams)]
drafts_f = filter_drafts(
    drafts=drafts,
    leagues=leagues,
    season=int(season),
    draft_status=filter_draft_status,
    draft_type=filter_draft_type,
    scoring_types=filter_scoring,
    league_sizes=league_sizes,
    min_rounds=min_rounds_val,
    max_rounds=max_rounds_val,
    date_min=date_min,
    date_max=date_max,
    te_premium_only=te_premium_only,
)

st.subheader("Filtered draft set")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Drafts", f"{len(drafts_f):,}")
m2.metric("Picks (rows)", f"{len(picks):,}")
m3.metric("Draft months", f"{drafts_f['start_month'].nunique() if 'start_month' in drafts_f.columns else 0:,}")
bad_dt = int(drafts["start_dt"].isna().sum())
m4.metric("Drafts missing start_dt", f"{bad_dt:,}")

if len(drafts_f) == 0:
    st.warning("No drafts match your filters. Relax filters and try again.")
    st.stop()

# Load players + ppg
players_df = load_players_df()

with st.spinner("Loading PPG (Sleeper stats)…"):
    try:
        ppg_df = load_ppg(int(ppg_season))
    except Exception as e:
        ppg_df = pd.DataFrame(columns=["player_id", "ppg"])
        st.warning(f"Could not load PPG for {ppg_season}: {e}")

# Compute pool
pool = compute_player_pick_stats(picks=picks, players_df=players_df, drafts_filtered=drafts_f, ppg_df=ppg_df)
pool = add_pos_rank(pool)

# Board build (startup snake vs rookie linear)
if board_kind.startswith("Startup"):
    board_wide, board_long = build_snake_board(pool, num_teams=int(num_teams), num_rounds=int(num_rounds))
else:
    # rookie-ish filter: years_exp 0-1 if available
    if "years_exp" in pool.columns:
        pool2 = pool[pd.to_numeric(pool["years_exp"], errors="coerce").isin([0, 1])].copy()
    else:
        pool2 = pool.copy()
    board_wide, board_long = build_linear_board(pool2, num_teams=int(num_teams), num_rounds=int(num_rounds))

# Layout: board + long table
st.subheader("ADP Board")

left, right = st.columns([2.2, 1.3], gap="large")

with left:
    st.caption("Board cells are ADP-ranked players mapped into draft slots.")
    st.dataframe(board_wide, use_container_width=True, height=520)

    st.caption("Long format (your requested output). Downloadable.")
    st.dataframe(
        board_long.drop(columns=["player_id"]),
        use_container_width=True,
        height=360,
    )
    csv_bytes = board_long.drop(columns=["player_id"]).to_csv(index=False).encode("utf-8")
    st.download_button("Download board (long CSV)", data=csv_bytes, file_name=f"adp_board_long_{season}.csv", mime="text/csv")

with right:
    st.subheader("Player lookup")

    # Build a stable label list
    pool_disp = pool.copy()
    pool_disp["label"] = (
        pool_disp["full_name"].fillna("")
        + " | "
        + pool_disp["team"].fillna("")
        + " "
        + pool_disp["position"].fillna("")
        + " | ADP "
        + pool_disp["adp"].astype(str)
    )
    pool_disp = pool_disp.sort_values("adp")
    labels = pool_disp["label"].tolist()

    # Default selection: first label if any
    default_idx = 0 if len(labels) else None
    selected_label = st.selectbox("Select a player", options=labels, index=default_idx if default_idx is not None else 0)

    if not selected_label:
        st.info("Pick a player to view details.")
        st.stop()

    row = pool_disp[pool_disp["label"] == selected_label].head(1)
    if row.empty:
        st.warning("Selection not found in pool (filters may have changed).")
        st.stop()

    selected_pid = row["player_id"].iloc[0]
    selected_name = row["full_name"].iloc[0]
    selected_team = row["team"].iloc[0]
    selected_pos = row["position"].iloc[0]

    # Summary stats
    st.markdown(f"### {selected_name} ({selected_team} {selected_pos})")
    s1, s2 = st.columns(2)
    s1.metric("ADP", f"{row['adp'].iloc[0]:.2f}")
    s2.metric("Pos rank", f"{int(row['pos_rank'].iloc[0])}")

    s3, s4 = st.columns(2)
    s3.metric("Drafts", f"{int(row['drafts'].iloc[0]):,}")
    s4.metric("Picks", f"{int(row['picks'].iloc[0]):,}")

    s5, s6 = st.columns(2)
    s5.metric("Min pick", f"{float(row['min_pick'].iloc[0]):.0f}")
    s6.metric("Max pick", f"{float(row['max_pick'].iloc[0]):.0f}")

    if "ppg" in row.columns and pd.notna(row["ppg"].iloc[0]):
        st.metric(f"PPG ({ppg_season})", f"{float(row['ppg'].iloc[0]):.2f}")
    else:
        st.caption("PPG not available for this player/season.")

    st.divider()

    # Build player picks distribution (using filtered drafts)
    p_sub = picks.copy()
    p_sub["draft_id"] = p_sub["draft_id"].astype(str)
    p_sub["player_id"] = p_sub["player_id"].astype(str)
    p_sub = p_sub[(p_sub["player_id"] == str(selected_pid)) & (p_sub["draft_id"].isin(set(drafts_f["draft_id"].astype(str))))].copy()

    p_sub["pick_no_calc"] = infer_pick_no(p_sub)
    p_sub["pick_no_calc"] = pd.to_numeric(p_sub["pick_no_calc"], errors="coerce")
    p_sub = p_sub[p_sub["pick_no_calc"].notna()].copy()

    st.markdown("#### Draft position distribution")
    if len(p_sub) == 0:
        st.info("No pick rows found for this player in the filtered draft set.")
    else:
        fig = plt.figure()
        plt.hist(p_sub["pick_no_calc"].values, bins=30)
        plt.xlabel("Overall pick")
        plt.ylabel("Count")
        st.pyplot(fig, clear_figure=True)

    st.markdown("#### ADP trend by month (last 5 months present)")
    trend = player_monthly_trend(picks, drafts_f, str(selected_pid), last_n_months=5)
    if trend.empty:
        st.info("No monthly trend available (missing start_month or not enough drafts).")
    else:
        fig2 = plt.figure()
        plt.plot(trend["start_month"].astype(str), trend["adp"].astype(float), marker="o")
        plt.gca().invert_yaxis()  # lower ADP = earlier pick
        plt.xlabel("Draft month")
        plt.ylabel("ADP (lower is earlier)")
        st.pyplot(fig2, clear_figure=True)
        st.dataframe(trend, use_container_width=True, height=180)

st.caption("Tip: if you see missing start_dt counts, those drafts will be excluded by date filters (they have invalid start_time).")
