# app_adp_board.py
# ============================================================
# Sleeper Dynasty ADP Board (RAW-first) — VISUAL GRID + IN-APP POPUP (NO STICKY REOPEN) + GRAPHS INSIDE POPUP
#
# Fixes requested:
# ✅ Clicking a player should NOT feel like a full "site refresh"
#    - Removed JS window.location.href (which forces a hard navigation)
#    - Use a normal same-tab link (target="_self") so Streamlit handles it
#    - NOTE: Streamlit will still rerun the script on interaction (that’s normal),
#            but this avoids the “hard refresh” behavior you were seeing.
#
# ✅ When you close the popup, it stays closed even after filter changes
#    - Uses st.session_state["dialog_open"] and clears query param on close
#    - Popup only opens when you *newly* click a player (pid changes)
#
# ✅ ADP graphs live with the player card (inside the popup), not under the board
#
# ✅ Streamlit deprecation fixed: st.dataframe(..., width="stretch")
# ============================================================

import os
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Reduce noisy Streamlit runtime warnings (internal)
# ------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"coroutine 'expire_cache' was never awaited",
)

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Sleeper ADP Board", layout="wide")

# ----------------------------
# Styling + Position colors
# ----------------------------
POSITION_COLORS = {
    "QB": "#a855f7",   # purple
    "RB": "#22c55e",   # green
    "WR": "#3b82f6",   # blue
    "TE": "#f59e0b",   # amber
    "K":  "#9ca3af",   # gray
    "RDP": "#ef4444",  # red (rookie pick)
    "UNK": "#64748b",  # slate
}

POSITION_TEXT = {
    "QB": "QB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
    "K": "K",
    "RDP": "Rookie Pick",
    "UNK": "?",
}

APP_CSS = """
<style>
  /* Page polish */
  .block-container { padding-top: 1.2rem; padding-bottom: 3rem; }
  h1, h2, h3 { letter-spacing: -0.02em; }

  /* Board wrapper */
  .adp-board-wrap {
    width: 100%;
    overflow-x: auto;
    padding: 12px 6px 10px 6px;
    border-radius: 18px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  }

  /* HTML table grid */
  table.adp-board {
    border-collapse: separate;
    border-spacing: 10px;
    width: max-content;
    min-width: 100%;
  }
  table.adp-board th {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    color: rgba(255,255,255,0.78);
    letter-spacing: 0.06em;
    padding: 2px 6px;
    text-align: center;
    white-space: nowrap;
  }
  table.adp-board th.roundhdr {
    text-align: left;
    padding-left: 6px;
    position: sticky;
    left: 0;
    background: rgba(15, 23, 42, 0.0);
    z-index: 3;
  }

  table.adp-board td.roundlbl {
    font-size: 12px;
    font-weight: 800;
    color: rgba(255,255,255,0.75);
    padding: 0 8px;
    white-space: nowrap;
    position: sticky;
    left: 0;
    z-index: 2;
    background: rgba(2,6,23,0.40);
    border-radius: 12px;
  }

  /* Pick cell */
  a.pickcell {
    display: block;
    width: 130px;
    height: 128px;
    border-radius: 18px;
    overflow: hidden;
    text-decoration: none !important;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 6px 18px rgba(0,0,0,0.22);
    transition: transform 120ms ease, box-shadow 120ms ease, border 120ms ease;
    position: relative;
    background: rgba(255,255,255,0.02);
    cursor: pointer;
  }
  a.pickcell:hover {
    transform: translateY(-2px);
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 10px 28px rgba(0,0,0,0.30);
  }

  /* Top "image" region */
  .cell-top {
    height: 72px;
    width: 100%;
    background: rgba(255,255,255,0.04);
    position: relative;
  }
  .cell-top img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: 50% 18%;
    display: block;
    filter: contrast(1.02) saturate(1.05);
  }
  /* subtle gradient to help text */
  .cell-top:after {
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(to bottom, rgba(0,0,0,0.05), rgba(0,0,0,0.40));
    pointer-events: none;
  }

  /* Bottom region: position color */
  .cell-bottom {
    height: 56px;
    width: 100%;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 4px;
    color: #0b1220;
    font-weight: 800;
  }
  .cell-bottom .name {
    font-size: 12px;
    line-height: 1.05;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: rgba(255,255,255,0.95);
    text-shadow: 0 1px 10px rgba(0,0,0,0.45);
  }
  .cell-bottom .meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    font-weight: 800;
    color: rgba(255,255,255,0.92);
    text-shadow: 0 1px 10px rgba(0,0,0,0.45);
  }
  .pill {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(0,0,0,0.18);
    border: 1px solid rgba(255,255,255,0.14);
    font-size: 10px;
    letter-spacing: 0.04em;
  }

  /* Selected highlight */
  a.pickcell.selected {
    outline: 2px solid rgba(255,255,255,0.75);
    outline-offset: 2px;
  }

  /* Rookie pick placeholder (no image) */
  .rp-top {
    height: 72px;
    width: 100%;
    background: radial-gradient(circle at 30% 25%, rgba(255,255,255,0.18), rgba(255,255,255,0.04));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    color: rgba(255,255,255,0.92);
  }

  /* Small caption vibe */
  .muted {
    color: rgba(255,255,255,0.62);
    font-size: 12px;
  }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

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
# Display helpers
# ----------------------------
def sleeper_headshot_url(player_id: str) -> str:
    pid = str(player_id)
    return f"https://sleepercdn.com/content/nfl/players/{pid}.jpg"


def normalize_pos(pos: Any) -> str:
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return "UNK"
    p = str(pos).strip().upper()
    if p in ["RDP", "ROOKIE_PICK", "ROOKIE PICK", "PICK"]:
        return "RDP"
    if p not in POSITION_COLORS:
        return "UNK"
    return p


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


# ----------------------------
# Path helpers (ROBUST)
# ----------------------------
def find_data_root_candidates(start_dir: str) -> List[str]:
    candidates = []
    cur = os.path.abspath(start_dir)

    for _ in range(10):
        dr = os.path.join(cur, "data", "raw")
        if os.path.isdir(dr):
            candidates.append(os.path.join(cur, "data"))

        nested = os.path.join(cur, "sleeper_dynasty_adp", "data", "raw")
        if os.path.isdir(nested):
            candidates.append(os.path.join(cur, "sleeper_dynasty_adp", "data"))

        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    out, seen = [], set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def pick_best_data_dir() -> Tuple[str, str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = find_data_root_candidates(here)

    if not candidates:
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
    df = drafts.copy()

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

    lo = 946684800000   # 2000-01-01
    hi = 2051222400000  # 2035-01-01
    ms = ms.where((ms >= lo) & (ms <= hi))

    ms_int = ms.round().astype("Int64")
    start_dt = pd.to_datetime(ms_int.astype("float64"), unit="ms", utc=True, errors="coerce")

    df["start_dt"] = start_dt
    df["start_date"] = df["start_dt"].dt.strftime("%Y-%m-%d").astype("string")
    df["start_month"] = df["start_dt"].dt.strftime("%Y-%m").astype("string")
    return df


# ----------------------------
# Loading RAW season files
# ----------------------------
def season_raw_paths(raw_dir: str, season: int) -> Tuple[str, str, str]:
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

    drafts = pd.read_parquet(drafts_path)
    picks = pd.read_parquet(picks_path)
    leagues = pd.read_parquet(leagues_path) if os.path.exists(leagues_path) else pd.DataFrame()

    for col in ["draft_id", "league_id"]:
        if col in drafts.columns:
            drafts[col] = drafts[col].astype(str)
    if "draft_id" in picks.columns:
        picks["draft_id"] = picks["draft_id"].astype(str)
    if "player_id" in picks.columns:
        picks["player_id"] = picks["player_id"].astype(str)
    if not leagues.empty and "league_id" in leagues.columns:
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

    if "position" in df.columns:
        df["position"] = df["position"].map(normalize_pos)

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
        return pd.DataFrame(columns=["player_id", "ppg", "games_played", "fantasy_pts"])

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
    for c in ["pick_no", "overall_pick", "pick", "draft_pick", "pick_number"]:
        if c in picks.columns:
            s = pd.to_numeric(picks[c], errors="coerce")
            if s.notna().any():
                return s
    if "round" in picks.columns and "draft_slot" in picks.columns:
        r = pd.to_numeric(picks["round"], errors="coerce")
        ds = pd.to_numeric(picks["draft_slot"], errors="coerce")
        return (r - 1) * 12 + ds
    return pd.Series(np.nan, index=picks.index)


def add_pos_rank(pool: pd.DataFrame) -> pd.DataFrame:
    df = pool.copy()
    df["adp"] = pd.to_numeric(df["adp"], errors="coerce")
    df["position"] = df["position"].map(normalize_pos)
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


def derive_round_from_overall_pick(pick_no: pd.Series, st_teams: pd.Series) -> pd.Series:
    p = pd.to_numeric(pick_no, errors="coerce")
    t = pd.to_numeric(st_teams, errors="coerce")
    t = t.where(t.notna() & (t > 0), np.nan)
    out = (np.floor((p - 1) / t) + 1)
    return pd.to_numeric(out, errors="coerce")


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

    if "season" in df.columns:
        df = df[pd.to_numeric(df["season"], errors="coerce") == season].copy()

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

    if "start_dt" in df.columns:
        if date_min is not None:
            df = df[df["start_dt"] >= date_min].copy()
        if date_max is not None:
            df = df[df["start_dt"] <= date_max].copy()

    if te_premium_only and (not leagues.empty) and ("league_id" in df.columns) and ("league_id" in leagues.columns):
        te_cols = [c for c in leagues.columns if c.endswith("scoring_settings.bonus_rec_te") or c == "scoring_settings.bonus_rec_te"]
        if te_cols:
            te_col = te_cols[0]
            lg = leagues[["league_id", te_col]].copy()
            lg[te_col] = pd.to_numeric(lg[te_col], errors="coerce").fillna(0)
            df = df.merge(lg, on="league_id", how="left")
            df = df[pd.to_numeric(df[te_col], errors="coerce").fillna(0) > 0].copy()

    return df


def player_monthly_trend(
    picks: pd.DataFrame,
    drafts_filtered: pd.DataFrame,
    player_id: str,
    last_n_months: int = 5
) -> pd.DataFrame:
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


# ----------------------------
# Rookie pick placeholders via early kickers
# ----------------------------
def build_rookie_pick_placeholders(
    picks: pd.DataFrame,
    drafts_filtered: pd.DataFrame,
    players_df: pd.DataFrame,
    early_rounds: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if picks.empty or drafts_filtered.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "st_teams" not in drafts_filtered.columns:
        return pd.DataFrame(), pd.DataFrame()

    dmini = drafts_filtered[["draft_id", "st_teams"]].copy()
    dmini["draft_id"] = dmini["draft_id"].astype(str)
    dmini["st_teams"] = pd.to_numeric(dmini["st_teams"], errors="coerce")

    p = picks.copy()
    p["draft_id"] = p["draft_id"].astype(str)
    p["player_id"] = p["player_id"].astype(str)

    keep_draft_ids = set(dmini["draft_id"].unique())
    p = p[p["draft_id"].isin(keep_draft_ids)].copy()
    p = p.merge(dmini, on="draft_id", how="left")

    p["pick_no_calc"] = infer_pick_no(p)
    p["pick_no_calc"] = pd.to_numeric(p["pick_no_calc"], errors="coerce")
    p = p[p["pick_no_calc"].notna()].copy()

    pl = players_df[["player_id", "position"]].copy()
    pl["player_id"] = pl["player_id"].astype(str)
    p = p.merge(pl, on="player_id", how="left")

    if "round" in p.columns:
        rnd = pd.to_numeric(p["round"], errors="coerce")
        if rnd.notna().mean() < 0.50:
            p["_round_calc"] = derive_round_from_overall_pick(p["pick_no_calc"], p["st_teams"])
        else:
            p["_round_calc"] = rnd
    else:
        p["_round_calc"] = derive_round_from_overall_pick(p["pick_no_calc"], p["st_teams"])

    p["_round_calc"] = pd.to_numeric(p["_round_calc"], errors="coerce")
    p = p[p["_round_calc"].notna()].copy()

    early_k = p[(p["position"] == "K") & (p["_round_calc"] <= int(early_rounds))].copy()
    if early_k.empty:
        return pd.DataFrame(), pd.DataFrame()

    qualifying_draft_ids = set(early_k["draft_id"].unique())

    pk = p[(p["draft_id"].isin(qualifying_draft_ids)) & (p["position"] == "K")].copy()
    if pk.empty:
        return pd.DataFrame(), pd.DataFrame()

    pk = pk.sort_values(["draft_id", "pick_no_calc"]).copy()
    pk["_k_seq"] = pk.groupby("draft_id").cumcount() + 1

    st_teams = pd.to_numeric(pk["st_teams"], errors="coerce").fillna(12).astype(int)
    seq0 = pk["_k_seq"] - 1
    rp_round = (seq0 // st_teams) + 1
    rp_pir = (seq0 % st_teams) + 1

    pk["_rp_round"] = rp_round.astype(int)
    pk["_rp_pir"] = rp_pir.astype(int)
    pk["_rp_label"] = pk["_rp_round"].astype(str) + "." + pk["_rp_pir"].map(lambda x: f"{int(x):02d}")

    pk["_rp_id"] = "ROOKIE_PICK_" + pk["_rp_label"].astype(str)
    pk["_rp_name"] = "Rookie Pick " + pk["_rp_label"].astype(str)

    picks_rp = pk.copy()
    picks_rp["player_id"] = picks_rp["_rp_id"].astype(str)

    rp_meta = (
        picks_rp[["player_id", "_rp_name"]]
        .drop_duplicates("player_id")
        .rename(columns={"_rp_name": "full_name"})
    )
    rp_meta["team"] = ""
    rp_meta["position"] = "RDP"
    rp_meta["years_exp"] = np.nan
    rp_meta["is_rookie_pick"] = True

    keep_cols = [c for c in picks.columns if c in picks_rp.columns]
    for c in ["pick_no_calc", "_rp_label", "_rp_id", "_round_calc", "st_teams"]:
        if c in picks_rp.columns and c not in keep_cols:
            keep_cols.append(c)

    return picks_rp[keep_cols].copy(), rp_meta


def compute_player_pick_stats(
    picks: pd.DataFrame,
    players_df: pd.DataFrame,
    drafts_filtered: pd.DataFrame,
    ppg_df: Optional[pd.DataFrame] = None,
    include_positions: Optional[List[str]] = None,
    extra_meta: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    p = picks.copy()

    if "draft_id" not in p.columns or "player_id" not in p.columns:
        raise RuntimeError("picks parquet must include at least: draft_id, player_id")

    p["draft_id"] = p["draft_id"].astype(str)
    p["player_id"] = p["player_id"].astype(str)

    keep_draft_ids = set(drafts_filtered["draft_id"].astype(str).unique())
    p = p[p["draft_id"].isin(keep_draft_ids)].copy()

    p["pick_no_calc"] = infer_pick_no(p)
    p["pick_no_calc"] = pd.to_numeric(p["pick_no_calc"], errors="coerce")
    p = p[p["pick_no_calc"].notna()].copy()

    pl = players_df.copy()
    pl["player_id"] = pl["player_id"].astype(str)

    df = p.merge(pl, on="player_id", how="left")

    df["is_rookie_pick"] = pd.Series(False, index=df.index, dtype="boolean")

    if extra_meta is not None and not extra_meta.empty:
        em = extra_meta.copy()
        em["player_id"] = em["player_id"].astype(str)

        df = df.merge(
            em[["player_id", "full_name", "team", "position", "years_exp", "is_rookie_pick"]],
            on="player_id",
            how="left",
            suffixes=("", "_extra"),
        )

        for col in ["full_name", "team", "position", "years_exp"]:
            extra_col = f"{col}_extra"
            if extra_col in df.columns:
                df[col] = df[col].where(df[col].notna(), df[extra_col])
                df = df.drop(columns=[extra_col])

        if "is_rookie_pick_extra" in df.columns:
            df["is_rookie_pick"] = (
                df["is_rookie_pick_extra"]
                .astype("boolean")
                .fillna(False)
                | df["is_rookie_pick"].astype("boolean").fillna(False)
            )
            df = df.drop(columns=["is_rookie_pick_extra"])

    if include_positions is None:
        include_positions = ["QB", "RB", "WR", "TE"]

    if "position" in df.columns:
        df["position"] = df["position"].map(normalize_pos)
        allowed = [normalize_pos(x) for x in include_positions]
        df = df[df["position"].isin(allowed)].copy()

    out = (
        df.groupby("player_id", as_index=False)
          .agg(
              picks=("pick_no_calc", "size"),
              drafts=("draft_id", pd.Series.nunique),
              adp=("pick_no_calc", "mean"),
              min_pick=("pick_no_calc", "min"),
              max_pick=("pick_no_calc", "max"),
              full_name=("full_name", "first"),
              team=("team", "first"),
              position=("position", "first"),
              years_exp=("years_exp", "first"),
              is_rookie_pick=("is_rookie_pick", "first"),
          )
    )

    if ppg_df is not None and not ppg_df.empty:
        out = out.merge(ppg_df[["player_id", "ppg", "games_played", "fantasy_pts"]], on="player_id", how="left")

    out["adp"] = pd.to_numeric(out["adp"], errors="coerce").round(2)
    out["min_pick"] = pd.to_numeric(out["min_pick"], errors="coerce")
    out["max_pick"] = pd.to_numeric(out["max_pick"], errors="coerce")
    out["picks"] = pd.to_numeric(out["picks"], errors="coerce").fillna(0).astype(int)
    out["drafts"] = pd.to_numeric(out["drafts"], errors="coerce").fillna(0).astype(int)

    out = out[out["drafts"] > 0].copy()
    out = out.sort_values("adp").reset_index(drop=True)
    return out


# ----------------------------
# Build board mapping (for rendering)
# ----------------------------
def build_board_map_snake(pool: pd.DataFrame, num_teams: int, num_rounds: int) -> Dict[Tuple[int, int], Dict[str, Any]]:
    need = num_teams * num_rounds
    ranked = pool.sort_values("adp").head(need).reset_index(drop=True)

    mapping: Dict[Tuple[int, int], Dict[str, Any]] = {}
    pick_map = snake_picks(num_teams, num_rounds)

    for i, pck in enumerate(pick_map):
        if i >= len(ranked):
            break
        row = ranked.iloc[i].to_dict()
        r = int(pck["round"])
        team_col = int(pck["team"])
        mapping[(r, team_col)] = row

    return mapping


def build_board_map_linear(pool: pd.DataFrame, num_teams: int, num_rounds: int) -> Dict[Tuple[int, int], Dict[str, Any]]:
    need = num_teams * num_rounds
    ranked = pool.sort_values("adp").head(need).reset_index(drop=True)

    mapping: Dict[Tuple[int, int], Dict[str, Any]] = {}
    idx = 0
    for r in range(1, num_rounds + 1):
        for t in range(1, num_teams + 1):
            if idx >= len(ranked):
                break
            mapping[(r, t)] = ranked.iloc[idx].to_dict()
            idx += 1
    return mapping


def get_query_pid() -> Optional[str]:
    try:
        qp = st.query_params
        pid = qp.get("pid", None)
        if isinstance(pid, list):
            return pid[0] if pid else None
        return pid
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            pid = qp.get("pid", [None])[0]
            return pid
        except Exception:
            return None


def clear_selected_pid() -> None:
    try:
        if "pid" in st.query_params:
            del st.query_params["pid"]
        return
    except Exception:
        pass
    try:
        st.experimental_set_query_params()
    except Exception:
        pass


def render_adp_board_html(
    mapping: Dict[Tuple[int, int], Dict[str, Any]],
    num_teams: int,
    num_rounds: int,
    selected_pid: Optional[str],
    title_line: str,
) -> None:
    """
    Key behavior:
    - NO JS navigation (which looked like a hard refresh)
    - Use plain href + target="_self" to keep same tab
    """
    html = []
    html.append('<div class="adp-board-wrap">')
    html.append(f'<div class="muted" style="margin: 0 0 8px 6px;">{title_line}</div>')
    html.append('<table class="adp-board">')

    html.append("<tr>")
    html.append('<th class="roundhdr">Round</th>')
    for t in range(1, num_teams + 1):
        html.append(f"<th>Team {t}</th>")
    html.append("</tr>")

    for r in range(1, num_rounds + 1):
        html.append("<tr>")
        html.append(f'<td class="roundlbl">Round {r}</td>')

        for t in range(1, num_teams + 1):
            cell = mapping.get((r, t))
            if not cell:
                html.append("<td></td>")
                continue

            pid = safe_str(cell.get("player_id", ""))
            name = safe_str(cell.get("full_name", ""))
            pos = normalize_pos(cell.get("position", "UNK"))
            team = safe_str(cell.get("team", ""))
            bg = POSITION_COLORS.get(pos, POSITION_COLORS["UNK"])

            is_rp = bool(cell.get("is_rookie_pick", False)) or (pos == "RDP") or pid.startswith("ROOKIE_PICK_")
            selected_class = " selected" if (selected_pid and str(selected_pid) == pid) else ""
            href = f"?pid={pid}"

            if is_rp:
                top = '<div class="rp-top">🔴</div>'
            else:
                img = sleeper_headshot_url(pid)
                top = f'<div class="cell-top"><img src="{img}" loading="lazy" /></div>'

            bottom = (
                f'<div class="cell-bottom" style="background:{bg};">'
                f'  <div class="name">{name}</div>'
                f'  <div class="meta">'
                f'    <span class="pill">{POSITION_TEXT.get(pos, pos)}</span>'
                f'    <span class="pill">{team if team else ""}</span>'
                f'  </div>'
                f'</div>'
            )

            html.append("<td>")
            # target="_self" keeps navigation in the same tab
            html.append(f'<a class="pickcell{selected_class}" href="{href}" target="_self">{top}{bottom}</a>')
            html.append("</td>")

        html.append("</tr>")

    html.append("</table></div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)


# ----------------------------
# Popup with graphs embedded
# ----------------------------
@st.dialog("Player Quick View", width="large")
def show_player_dialog(
    sel_row: pd.Series,
    ppg_season: int,
    p_sub: pd.DataFrame,
    trend: pd.DataFrame,
) -> None:
    sel_name = safe_str(sel_row.get("full_name", "Unknown"))
    sel_team = safe_str(sel_row.get("team", ""))
    sel_pos = normalize_pos(sel_row.get("position", "UNK"))
    sel_pid = safe_str(sel_row.get("player_id", ""))

    sel_is_rp = bool(sel_row.get("is_rookie_pick", False)) or sel_pos == "RDP" or sel_pid.startswith("ROOKIE_PICK_")

    top = st.columns([1.15, 2.85], gap="large")
    with top[0]:
        if not sel_is_rp and sel_pid:
            st.image(sleeper_headshot_url(sel_pid), width=190)
        else:
            st.markdown("### 🔴")
            st.caption("Rookie pick placeholder")

    with top[1]:
        st.markdown(f"### {sel_name}")
        st.caption(f"{sel_team} • {POSITION_TEXT.get(sel_pos, sel_pos)}")

        adp_val = sel_row.get("adp", np.nan)
        ppg_val = sel_row.get("ppg", np.nan)
        gp_val = sel_row.get("games_played", np.nan)
        fp_val = sel_row.get("fantasy_pts", np.nan)
        drafts_val = sel_row.get("drafts", np.nan)
        pos_rank_val = sel_row.get("pos_rank", np.nan)
        min_pick_val = sel_row.get("min_pick", np.nan)
        max_pick_val = sel_row.get("max_pick", np.nan)

        r1 = st.columns(4)
        r1[0].metric("ADP", f"{float(adp_val):.2f}" if pd.notna(adp_val) else "—")
        r1[1].metric(f"PPG ({ppg_season})", f"{float(ppg_val):.2f}" if pd.notna(ppg_val) else "—")
        r1[2].metric("Drafts", f"{int(drafts_val):,}" if pd.notna(drafts_val) else "—")
        r1[3].metric("Pos Rank", f"{int(pos_rank_val):,}" if pd.notna(pos_rank_val) else "—")

        r2 = st.columns(4)
        r2[0].metric("Min", f"{float(min_pick_val):.0f}" if pd.notna(min_pick_val) else "—")
        r2[1].metric("Max", f"{float(max_pick_val):.0f}" if pd.notna(max_pick_val) else "—")
        r2[2].metric("Games", f"{int(gp_val):,}" if pd.notna(gp_val) else "—")
        r2[3].metric("Fantasy Pts", f"{float(fp_val):.1f}" if pd.notna(fp_val) else "—")

    st.divider()

    # --- Graphs live WITH the player card (inside popup) ---
    g1, g2 = st.columns([1.2, 1.3], gap="large")

    with g1:
        st.markdown("#### Draft position distribution")
        if p_sub is None or len(p_sub) == 0:
            st.info("No pick rows found for this entity in the filtered draft set.")
        else:
            fig = plt.figure()
            plt.hist(p_sub["pick_no_calc"].values, bins=30)
            plt.xlabel("Overall pick")
            plt.ylabel("Count")
            st.pyplot(fig, clear_figure=True)

    with g2:
        st.markdown("#### ADP trend by month (last 5 months present)")
        if trend is None or trend.empty:
            st.info("No monthly trend available.")
        else:
            fig2 = plt.figure()
            plt.plot(trend["start_month"].astype(str), trend["adp"].astype(float), marker="o")
            plt.gca().invert_yaxis()
            plt.xlabel("Draft month")
            plt.ylabel("ADP (lower is earlier)")
            st.pyplot(fig2, clear_figure=True)
            st.dataframe(trend, width="stretch", height=180)

    st.divider()
    if st.button("Close", type="primary"):
        # Critical: close means "don't reopen on the next rerun"
        st.session_state["dialog_open"] = False
        st.session_state["last_pid_seen"] = None
        clear_selected_pid()
        st.rerun()


# ============================================================
# UI
# ============================================================
st.title("Sleeper Dynasty ADP Board")

project_root_auto, raw_dir_auto, snapshots_dir_auto = pick_best_data_dir()

# dialog state
st.session_state.setdefault("dialog_open", False)
st.session_state.setdefault("last_pid_seen", None)

with st.sidebar:
    st.header("Options Menu")

    st.subheader("Data / Paths")
    st.caption(f"Auto project root:\n{project_root_auto}")
    raw_dir = st.text_input(
        "RAW dir",
        value=raw_dir_auto,
        help="Folder containing drafts/, picks/, leagues/",
    ).strip()

    snapshots_dir = st.text_input(
        "Snapshots dir (optional)",
        value=snapshots_dir_auto,
    ).strip()

    write_snaps = st.checkbox("Write snapshots for selected season", value=False)

    st.divider()
    st.subheader("Board")
    season = st.number_input("Season", min_value=2015, max_value=2030, value=2026, step=1)
    board_kind = st.selectbox("Board type", ["Startup (snake)", "Rookie (linear)"], index=0)

    num_teams = st.number_input("League size", min_value=4, max_value=32, value=12, step=1)
    if board_kind.startswith("Startup"):
        num_rounds = st.number_input("Rounds", min_value=1, max_value=60, value=35, step=1)
    else:
        num_rounds = st.number_input("Rounds", min_value=1, max_value=20, value=5, step=1)

    st.divider()
    st.subheader("Startup rookies / rookie picks")
    startup_inclusion_mode = st.selectbox(
        "Startup inclusion mode",
        options=[
            "Include rookies (players)",
            "Include rookie picks (K placeholders)",
            "Exclude rookies and rookie picks",
        ],
        index=0,
    )
    kicker_placeholder_rounds = st.slider("K placeholder rounds", 1, 10, 4, 1)

    st.divider()
    st.subheader("Filters")
    filter_draft_status = st.multiselect(
        "Draft status",
        ["complete", "pre_draft", "drafting", "paused"],
        default=["complete"],
    )
    filter_draft_type = st.multiselect(
        "Draft type",
        ["snake", "linear", "auction"],
        default=["snake", "linear"],
    )
    filter_scoring = st.multiselect(
        "Scoring type (md_scoring_type)",
        ["dynasty_2qb", "dynasty_ppr", "dynasty_half_ppr", "dynasty_std", "2qb", "ppr", "half_ppr", "std", "idp", "idp_1qb"],
        default=["dynasty_2qb"],
    )
    te_premium_only = st.checkbox("TE Premium only (bonus_rec_te > 0)", value=False)

    min_rounds = st.number_input("Min st_rounds (0 disables)", 0, 80, 0, 1)
    max_rounds = st.number_input("Max st_rounds (0 disables)", 0, 80, 0, 1)
    min_rounds_val = None if min_rounds == 0 else int(min_rounds)
    max_rounds_val = None if max_rounds == 0 else int(max_rounds)

    st.divider()
    st.subheader("Player pool quality")
    min_drafts_per_month = st.slider("Min drafts per month", 0, 50, 10, 1)
    ppg_season = st.number_input("PPG season", 2015, 2030, 2025, 1)

# Load data
try:
    drafts, picks, leagues = load_raw_season(raw_dir, int(season))
except Exception as e:
    st.error(f"Failed to load RAW season files.\n\n{e}")
    st.stop()

if write_snaps:
    try:
        write_snapshots_for_season(snapshots_dir, int(season), drafts, picks, leagues)
        st.success("Snapshots written.")
    except Exception as e:
        st.warning(f"Could not write snapshots:\n{e}")

# Date controls
drafts_valid_dt = drafts[drafts["start_dt"].notna()].copy()
if len(drafts_valid_dt) > 0:
    dt_min = drafts_valid_dt["start_dt"].min()
    dt_max = drafts_valid_dt["start_dt"].max()
else:
    dt_min = pd.Timestamp("2000-01-01", tz="UTC")
    dt_max = pd.Timestamp("2000-01-02", tz="UTC")

c1, c2, c3 = st.columns([1.2, 1.2, 3.6])
with c1:
    date_from = st.date_input("Drafts from", value=dt_min.date())
with c2:
    date_to = st.date_input("Drafts to", value=dt_max.date())
with c3:
    st.caption("Filters use draft start_time (`start_dt`). Board cells are clickable.")

date_min = pd.Timestamp(datetime.combine(date_from, datetime.min.time()), tz="UTC")
date_max = pd.Timestamp(datetime.combine(date_to, datetime.max.time()), tz="UTC")

drafts_f = filter_drafts(
    drafts=drafts,
    leagues=leagues,
    season=int(season),
    draft_status=filter_draft_status,
    draft_type=filter_draft_type,
    scoring_types=filter_scoring,
    league_sizes=[int(num_teams)],
    min_rounds=min_rounds_val,
    max_rounds=max_rounds_val,
    date_min=date_min,
    date_max=date_max,
    te_premium_only=te_premium_only,
)

months_in_scope = sorted([m for m in drafts_f.get("start_month", pd.Series([], dtype="string")).dropna().unique().tolist()])
num_months_in_scope = len(months_in_scope)
required_player_drafts = int(min_drafts_per_month) * max(num_months_in_scope, 1)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Drafts", f"{len(drafts_f):,}")
m2.metric("Picks (rows)", f"{len(picks):,}")
m3.metric("Months in date range", f"{num_months_in_scope:,}")
m4.metric("Drafts missing start_dt", f"{int(drafts['start_dt'].isna().sum()):,}")

st.caption(
    f"Player pool requirement: {min_drafts_per_month} drafts/month × {num_months_in_scope or 1} months "
    f"= **{required_player_drafts} drafts** minimum per entity"
)

if len(drafts_f) == 0:
    st.warning("No drafts match your filters. Relax filters and try again.")
    # also make sure the dialog doesn’t “stick”
    st.session_state["dialog_open"] = False
    st.session_state["last_pid_seen"] = None
    clear_selected_pid()
    st.stop()

players_df = load_players_df()

with st.spinner("Loading PPG…"):
    try:
        ppg_df = load_ppg(int(ppg_season))
    except Exception as e:
        ppg_df = pd.DataFrame(columns=["player_id", "ppg", "games_played", "fantasy_pts"])
        st.warning(f"Could not load PPG for {ppg_season}: {e}")

extra_meta = pd.DataFrame()
picks_for_pool = picks.copy()
include_positions = ["QB", "RB", "WR", "TE"]

if board_kind.startswith("Startup"):
    if startup_inclusion_mode == "Include rookie picks (K placeholders)":
        rp_picks, rp_meta = build_rookie_pick_placeholders(
            picks=picks,
            drafts_filtered=drafts_f,
            players_df=players_df,
            early_rounds=int(kicker_placeholder_rounds),
        )
        if not rp_picks.empty:
            common_cols = [c for c in picks_for_pool.columns if c in rp_picks.columns]
            picks_for_pool = pd.concat([picks_for_pool, rp_picks[common_cols]], ignore_index=True)
            extra_meta = rp_meta.copy()
        include_positions = ["QB", "RB", "WR", "TE", "RDP"]

pool = compute_player_pick_stats(
    picks=picks_for_pool,
    players_df=players_df,
    drafts_filtered=drafts_f,
    ppg_df=ppg_df,
    include_positions=include_positions,
    extra_meta=extra_meta,
)

if board_kind.startswith("Startup") and startup_inclusion_mode == "Exclude rookies and rookie picks":
    if "years_exp" in pool.columns:
        pool = pool[~pd.to_numeric(pool["years_exp"], errors="coerce").isin([0, 1])].copy()

pool = add_pos_rank(pool)
pool = pool[pool["drafts"] >= required_player_drafts].copy()
pool = pool.sort_values("adp").reset_index(drop=True)

if pool.empty:
    st.warning(
        "No entities meet the minimum drafts requirement for the selected date range.\n\n"
        f"Try lowering 'Min drafts per month' (currently {min_drafts_per_month}) or widening the date range."
    )
    st.session_state["dialog_open"] = False
    st.session_state["last_pid_seen"] = None
    clear_selected_pid()
    st.stop()

pool_for_board = pool
if board_kind.startswith("Rookie"):
    if "years_exp" in pool.columns:
        pool_for_board = pool[pd.to_numeric(pool["years_exp"], errors="coerce").isin([0, 1])].copy()
    if pool_for_board.empty:
        st.warning("No rookie-eligible players meet the minimum drafts requirement.")
        st.session_state["dialog_open"] = False
        st.session_state["last_pid_seen"] = None
        clear_selected_pid()
        st.stop()

if board_kind.startswith("Startup"):
    mapping = build_board_map_snake(pool_for_board, int(num_teams), int(num_rounds))
else:
    mapping = build_board_map_linear(pool_for_board, int(num_teams), int(num_rounds))

selected_pid = get_query_pid()

# ----------------------------
# Dialog open/close logic (prevents "sticky reopen" after closing)
# ----------------------------
# Open only when pid changes (i.e., you clicked a new player)
if selected_pid and selected_pid != st.session_state.get("last_pid_seen"):
    st.session_state["dialog_open"] = True
    st.session_state["last_pid_seen"] = selected_pid

# If pid removed, don't keep dialog open
if not selected_pid:
    st.session_state["dialog_open"] = False
    st.session_state["last_pid_seen"] = None

st.subheader("ADP Board")
title_line = f"{board_kind} • Season {season} • {num_teams} teams × {num_rounds} rounds"
render_adp_board_html(mapping, int(num_teams), int(num_rounds), selected_pid, title_line)

st.caption("Click any square to open a player popup (with charts).")

# ----------------------------
# Popup (only when dialog_open is True)
# ----------------------------
if selected_pid and st.session_state.get("dialog_open", False):
    sel = pool[pool["player_id"].astype(str) == str(selected_pid)].head(1)
    if sel.empty:
        st.warning("Selected entity is not in the current pool (filters may have changed). Click another square.")
        st.session_state["dialog_open"] = False
        st.session_state["last_pid_seen"] = None
        clear_selected_pid()
    else:
        sel_row = sel.iloc[0]

        # Build p_sub + trend here and pass into dialog so graphs live with the player card
        p_sub = picks_for_pool.copy()
        p_sub["draft_id"] = p_sub["draft_id"].astype(str)
        p_sub["player_id"] = p_sub["player_id"].astype(str)
        p_sub = p_sub[
            (p_sub["player_id"] == str(selected_pid))
            & (p_sub["draft_id"].isin(set(drafts_f["draft_id"].astype(str))))
        ].copy()

        if "pick_no_calc" not in p_sub.columns:
            p_sub["pick_no_calc"] = infer_pick_no(p_sub)
        p_sub["pick_no_calc"] = pd.to_numeric(p_sub["pick_no_calc"], errors="coerce")
        p_sub = p_sub[p_sub["pick_no_calc"].notna()].copy()

        trend = player_monthly_trend(picks_for_pool, drafts_f, str(selected_pid), last_n_months=5)

        show_player_dialog(sel_row, int(ppg_season), p_sub, trend)

st.caption("Tip: drafts missing start_dt are excluded by date filters (invalid start_time).")
