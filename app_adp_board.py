# app_adp_board.py
# ============================================================
# Sleeper Dynasty ADP Board + Auction Price Board — CLICKABLE TILES (NO URL CHANGES)
#
# Key UX guarantees:
# - Clicking a tile does NOT change URL, does NOT refresh filters, and does NOT recompute the board. 
# - Board/pool recompute ONLY when filters change (filter_sig changes).
# - Closing the dialog truly closes it; it will NOT re-open just because you change a filter.
#
# Boards supported:
# 1) Startup ADP (snake)
# 2) Rookie ADP (linear)
# 3) Auction Price (snake)
#
# Auction notes:
# - Uses picks.md_amount (from Sleeper auction picks metadata.amount) as price
# - Aggregates by player_id over filtered auction drafts:
#     avg_price, median, min, max, sales, drafts
# - Positional ranks are contiguous AFTER final filters, ranked by:
#     ADP boards: adp ascending (lower is earlier)
#     Auction board: avg_price descending (higher is earlier/left)
#
# Tile right pill:
# - ADP boards: shows slot pick label (1.01, 2.05, 10.11, etc.) based on board slot
#              Startup = snake-wrapped; Rookie = linear
# - Auction board: shows avg auction value ($)
#
# Player dialog graphs:
# - ADP boards: pick distribution (overall pick) + monthly ADP trend
# - Auction board: price distribution ($) + monthly avg price trend
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

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"coroutine 'expire_cache' was never awaited",
)

st.set_page_config(page_title="Sleeper ADP / Auction Board", layout="wide")

POSITION_COLORS = {
    "QB": "#a855f7",
    "RB": "#22c55e",
    "WR": "#3b82f6",
    "TE": "#f59e0b",
    "K":  "#9ca3af",
    "RDP": "#ef4444",
    "UNK": "#64748b",
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
  .block-container { padding-top: 1.2rem; padding-bottom: 3rem; }
  h1, h2, h3 { letter-spacing: -0.02em; }

  .muted { color: rgba(255,255,255,0.62); font-size: 12px; }

  .hdr-cell {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.78);
    text-align: center;
    white-space: nowrap;
  }
  .hdr-round { text-align: left; padding-left: 6px; }

  .roundlbl {
    font-size: 12px;
    font-weight: 900;
    color: rgba(255,255,255,0.75);
    padding: 10px 10px;
    border-radius: 12px;
    background: rgba(2,6,23,0.40);
    height: 128px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .dirpill {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    font-size: 11px;
    font-weight: 900;
    color: rgba(255,255,255,0.80);
  }

  .pickcard {
    width: 142px;
    height: 128px;
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 6px 18px rgba(0,0,0,0.22);
    background: rgba(255,255,255,0.02);
    transition: transform 120ms ease, box-shadow 120ms ease, border 120ms ease;
  }
  .pickcard:hover {
    transform: translateY(-2px);
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 10px 28px rgba(0,0,0,0.30);
  }

  .card-top { height: 72px; width: 100%; position: relative; background: rgba(255,255,255,0.04); }
  .card-top img {
    width: 100%; height: 100%;
    object-fit: cover; object-position: 50% 18%;
    display: block;
    filter: contrast(1.02) saturate(1.05);
  }
  .card-top:after {
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(to bottom, rgba(0,0,0,0.05), rgba(0,0,0,0.40));
    pointer-events: none;
  }

  .rp-top {
    height: 72px; width: 100%;
    background: radial-gradient(circle at 30% 25%, rgba(255,255,255,0.18), rgba(255,255,255,0.04));
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    color: rgba(255,255,255,0.92);
  }

  .card-bottom {
    height: 56px; width: 100%;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 4px;
    font-weight: 800;
  }

  .name {
    font-size: 12px;
    line-height: 1.05;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: rgba(255,255,255,0.95);
    text-shadow: 0 1px 10px rgba(0,0,0,0.45);
  }
  .meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    font-weight: 900;
    color: rgba(255,255,255,0.92);
    text-shadow: 0 1px 10px rgba(0,0,0,0.45);
    gap: 6px;
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
    white-space: nowrap;
  }

  /* Invisible overlay buttons (secondary only) */
  button[data-testid="stBaseButton-secondary"] {
    width: 142px !important;
    height: 128px !important;
    opacity: 0 !important;
    position: relative !important;
    top: -128px !important;
    margin-bottom: -128px !important;
    border-radius: 18px !important;
    padding: 0 !important;
  }
  div[data-testid="stButton"] { margin-top: 0.1rem; margin-bottom: 0.0rem; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

BASE = "https://api.sleeper.app/v1"
session = requests.Session()
session.headers.update({"User-Agent": "Sleeper-Dynasty-ADP/1.0"})


# -----------------------------
# HTTP
# -----------------------------
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


# -----------------------------
# Small utils
# -----------------------------
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


def compute_filter_sig(**kwargs) -> str:
    def norm(v):
        if isinstance(v, (list, tuple, set)):
            return tuple(sorted([str(x) for x in v]))
        return str(v)
    items = [(k, norm(v)) for k, v in sorted(kwargs.items())]
    return "|".join([f"{k}={v}" for k, v in items])


def format_pick_label_linear(round_no: int, team_no: int) -> str:
    # Linear: team column equals pick-in-round
    return f"{int(round_no)}.{int(team_no):02d}"


def format_pick_label_snake(round_no: int, team_no: int, num_teams: int) -> str:
    # Snake: even rounds reverse the pick-in-round order
    if int(round_no) % 2 == 1:
        pick_in_round = int(team_no)
    else:
        pick_in_round = int(num_teams) - int(team_no) + 1
    return f"{int(round_no)}.{int(pick_in_round):02d}"


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


def infer_pick_no(picks: pd.DataFrame) -> pd.Series:
    for c in ["pick_no", "overall_pick", "pick", "draft_pick", "pick_number"]:
        if c in picks.columns:
            s = pd.to_numeric(picks[c], errors="coerce")
            if s.notna().any():
                return s

    # fallback: compute from round/draft_slot (assumes 12 teams unless st_teams in picks, which we don't have here)
    if "round" in picks.columns and "draft_slot" in picks.columns:
        r = pd.to_numeric(picks["round"], errors="coerce")
        ds = pd.to_numeric(picks["draft_slot"], errors="coerce")
        return (r - 1) * 12 + ds

    return pd.Series(np.nan, index=picks.index)


def snake_picks(num_teams: int, num_rounds: int) -> List[Dict[str, int]]:
    out = []
    for r in range(1, num_rounds + 1):
        for pick_in_round in range(1, num_teams + 1):
            team = pick_in_round if (r % 2 == 1) else (num_teams - pick_in_round + 1)
            out.append({"round": r, "pick_in_round": pick_in_round, "team": team})
    return out


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


# ----------------------------
# Rookie logic (season-aware)
# ----------------------------
def is_rookie_for_season_row(row: pd.Series, season: int) -> bool:
    ry = row.get("rookie_year", np.nan)
    if pd.notna(ry):
        return int(ry) == int(season)
    ye = row.get("years_exp", np.nan)
    return pd.notna(ye) and int(ye) == 0


def filter_rookies_by_season(df: pd.DataFrame, season: int, keep_rookies: bool) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df.apply(lambda r: is_rookie_for_season_row(r, season), axis=1)
    return df[mask].copy() if keep_rookies else df[~mask].copy()


# ----------------------------
# Rookie Pick placeholders
# ----------------------------
def build_rookie_pick_placeholders(
    picks: pd.DataFrame,
    drafts_filtered: pd.DataFrame,
    players_df: pd.DataFrame,
    early_rounds: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if picks.empty or drafts_filtered.empty or "st_teams" not in drafts_filtered.columns:
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

    st_teams = pd.to_numeric(p["st_teams"], errors="coerce").fillna(12)
    p["_round_calc"] = (np.floor((p["pick_no_calc"] - 1) / st_teams) + 1)
    p["_round_calc"] = pd.to_numeric(p["_round_calc"], errors="coerce")

    early_k = p[(p["position"] == "K") & (p["_round_calc"] <= int(early_rounds))].copy()
    if early_k.empty:
        return pd.DataFrame(), pd.DataFrame()

    qualifying_draft_ids = set(early_k["draft_id"].unique())
    pk = p[(p["draft_id"].isin(qualifying_draft_ids)) & (p["position"] == "K")].copy()
    if pk.empty:
        return pd.DataFrame(), pd.DataFrame()

    pk = pk.sort_values(["draft_id", "pick_no_calc"]).copy()
    pk["_k_seq"] = pk.groupby("draft_id").cumcount() + 1

    st_teams2 = pd.to_numeric(pk["st_teams"], errors="coerce").fillna(12).astype(int)
    seq0 = pk["_k_seq"] - 1
    rp_round = (seq0 // st_teams2) + 1
    rp_pir = (seq0 % st_teams2) + 1

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
    rp_meta["rookie_year"] = np.nan
    rp_meta["is_rookie_pick"] = True

    keep_cols = [c for c in picks.columns if c in picks_rp.columns]
    for c in ["pick_no_calc", "_rp_label", "_rp_id", "_round_calc", "st_teams"]:
        if c in picks_rp.columns and c not in keep_cols:
            keep_cols.append(c)

    return picks_rp[keep_cols].copy(), rp_meta


# ----------------------------
# ADP pool
# ----------------------------
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

    for c in ["rookie_year", "years_exp", "full_name", "team", "position"]:
        if c not in pl.columns:
            pl[c] = np.nan if c in ["rookie_year", "years_exp"] else ""
    pl["position"] = pl["position"].map(normalize_pos)

    df = p.merge(pl, on="player_id", how="left")
    df["is_rookie_pick"] = pd.Series(False, index=df.index, dtype="boolean")

    if extra_meta is not None and not extra_meta.empty:
        em = extra_meta.copy()
        em["player_id"] = em["player_id"].astype(str)
        for c in ["full_name", "team", "position", "years_exp", "rookie_year", "is_rookie_pick"]:
            if c not in em.columns:
                em[c] = np.nan

        df = df.merge(
            em[["player_id", "full_name", "team", "position", "years_exp", "rookie_year", "is_rookie_pick"]],
            on="player_id",
            how="left",
            suffixes=("", "_extra"),
        )

        for col in ["full_name", "team", "position", "years_exp", "rookie_year"]:
            extra_col = f"{col}_extra"
            if extra_col in df.columns:
                df[col] = df[col].where(df[col].notna(), df[extra_col])
                df = df.drop(columns=[extra_col])

        if "is_rookie_pick_extra" in df.columns:
            df["is_rookie_pick"] = (
                df["is_rookie_pick_extra"].astype("boolean").fillna(False)
                | df["is_rookie_pick"].astype("boolean").fillna(False)
            )
            df = df.drop(columns=["is_rookie_pick_extra"])

    if include_positions is None:
        include_positions = ["QB", "RB", "WR", "TE"]

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
            rookie_year=("rookie_year", "first"),
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
    out["rookie_year"] = pd.to_numeric(out["rookie_year"], errors="coerce")

    out = out[out["drafts"] > 0].copy()
    out = out.sort_values("adp").reset_index(drop=True)
    return out


# ----------------------------
# Auction pool
# ----------------------------
def compute_player_auction_stats(
    picks: pd.DataFrame,
    players_df: pd.DataFrame,
    drafts_filtered: pd.DataFrame,
    ppg_df: Optional[pd.DataFrame] = None,
    include_positions: Optional[List[str]] = None,
) -> pd.DataFrame:
    p = picks.copy()
    if "draft_id" not in p.columns or "player_id" not in p.columns:
        raise RuntimeError("picks parquet must include at least: draft_id, player_id")

    p["draft_id"] = p["draft_id"].astype(str)
    p["player_id"] = p["player_id"].astype(str)

    keep_draft_ids = set(drafts_filtered["draft_id"].astype(str).unique())
    p = p[p["draft_id"].isin(keep_draft_ids)].copy()

    if "md_amount" not in p.columns:
        return pd.DataFrame()

    p["amount"] = pd.to_numeric(p["md_amount"], errors="coerce")
    p = p[p["amount"].notna()].copy()

    pl = players_df.copy()
    pl["player_id"] = pl["player_id"].astype(str)
    for c in ["rookie_year", "years_exp", "full_name", "team", "position"]:
        if c not in pl.columns:
            pl[c] = np.nan if c in ["rookie_year", "years_exp"] else ""
    pl["position"] = pl["position"].map(normalize_pos)

    df = p.merge(pl, on="player_id", how="left")

    if include_positions is None:
        include_positions = ["QB", "RB", "WR", "TE"]
    df["position"] = df["position"].map(normalize_pos)
    allowed = [normalize_pos(x) for x in include_positions]
    df = df[df["position"].isin(allowed)].copy()

    out = (
        df.groupby("player_id", as_index=False)
        .agg(
            sales=("amount", "size"),
            drafts=("draft_id", pd.Series.nunique),
            avg_price=("amount", "mean"),
            med_price=("amount", "median"),
            min_price=("amount", "min"),
            max_price=("amount", "max"),
            full_name=("full_name", "first"),
            team=("team", "first"),
            position=("position", "first"),
            years_exp=("years_exp", "first"),
            rookie_year=("rookie_year", "first"),
        )
    )

    for c in ["avg_price", "med_price", "min_price", "max_price"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    out["sales"] = pd.to_numeric(out["sales"], errors="coerce").fillna(0).astype(int)
    out["drafts"] = pd.to_numeric(out["drafts"], errors="coerce").fillna(0).astype(int)
    out["rookie_year"] = pd.to_numeric(out["rookie_year"], errors="coerce")

    if ppg_df is not None and not ppg_df.empty:
        out = out.merge(ppg_df[["player_id", "ppg", "games_played", "fantasy_pts"]], on="player_id", how="left")

    out = out[out["drafts"] > 0].copy()
    out = out.sort_values(["avg_price", "drafts"], ascending=[False, False]).reset_index(drop=True)
    return out


# ----------------------------
# Board maps
# ----------------------------
def build_board_map_snake_by_col(
    pool: pd.DataFrame,
    num_teams: int,
    num_rounds: int,
    sort_col: str,
    asc: bool,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    need = num_teams * num_rounds
    ranked = pool.sort_values(sort_col, ascending=asc).head(need).reset_index(drop=True)

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


def build_board_map_linear_by_col(
    pool: pd.DataFrame,
    num_teams: int,
    num_rounds: int,
    sort_col: str,
    asc: bool,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    need = num_teams * num_rounds
    ranked = pool.sort_values(sort_col, ascending=asc).head(need).reset_index(drop=True)

    mapping: Dict[Tuple[int, int], Dict[str, Any]] = {}
    idx = 0
    for r in range(1, num_rounds + 1):
        for t in range(1, num_teams + 1):
            if idx >= len(ranked):
                break
            mapping[(r, t)] = ranked.iloc[idx].to_dict()
            idx += 1
    return mapping


# ----------------------------
# Trends
# ----------------------------
def player_monthly_trend_adp(
    picks: pd.DataFrame,
    drafts_filtered: pd.DataFrame,
    player_id: str,
    last_n_months: int = 5
) -> pd.DataFrame:
    if "draft_id" not in picks.columns or "player_id" not in picks.columns:
        return pd.DataFrame(columns=["start_month", "adp", "picks"])

    d = drafts_filtered[["draft_id", "start_month"]].copy()
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


def player_monthly_trend_price(
    picks: pd.DataFrame,
    drafts_filtered: pd.DataFrame,
    player_id: str,
    last_n_months: int = 5
) -> pd.DataFrame:
    if "draft_id" not in picks.columns or "player_id" not in picks.columns or "md_amount" not in picks.columns:
        return pd.DataFrame(columns=["start_month", "avg_price", "sales"])

    d = drafts_filtered[["draft_id", "start_month"]].copy()
    d["draft_id"] = d["draft_id"].astype(str)

    p = picks.copy()
    p["draft_id"] = p["draft_id"].astype(str)
    p["player_id"] = p["player_id"].astype(str)

    p = p[p["player_id"] == str(player_id)].copy()
    p = p.merge(d, on="draft_id", how="inner")

    p["amount"] = pd.to_numeric(p["md_amount"], errors="coerce")
    p = p[p["amount"].notna()].copy()
    p = p[p["start_month"].notna()].copy()

    months = sorted([m for m in p["start_month"].unique() if m and m != "nan"])
    if not months:
        return pd.DataFrame(columns=["start_month", "avg_price", "sales"])
    keep = months[-last_n_months:]

    agg = (
        p[p["start_month"].isin(keep)]
        .groupby("start_month", as_index=False)
        .agg(avg_price=("amount", "mean"), sales=("amount", "size"))
    )
    agg["avg_price"] = pd.to_numeric(agg["avg_price"], errors="coerce").round(2)
    agg = agg.sort_values("start_month")
    return agg


# ----------------------------
# Dialog state
# ----------------------------
def select_player(pid: str) -> None:
    st.session_state["selected_pid"] = str(pid)
    st.session_state["dialog_open"] = True


def close_player_dialog() -> None:
    st.session_state["dialog_open"] = False
    st.session_state["selected_pid"] = None


@st.dialog("Player Quick View", width="large")
def show_player_dialog(
    mode: str,
    sel_row: pd.Series,
    ppg_season: int,
    picks_subset: pd.DataFrame,
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

        pos_rank_val = sel_row.get("pos_rank", np.nan)
        drafts_val = sel_row.get("drafts", np.nan)
        ppg_val = sel_row.get("ppg", np.nan)

        if mode == "auction":
            avg_price = sel_row.get("avg_price", np.nan)
            med_price = sel_row.get("med_price", np.nan)
            min_price = sel_row.get("min_price", np.nan)
            max_price = sel_row.get("max_price", np.nan)
            sales = sel_row.get("sales", np.nan)

            r1 = st.columns(4)
            r1[0].metric("Avg $", f"{float(avg_price):.2f}" if pd.notna(avg_price) else "—")
            r1[1].metric("Median $", f"{float(med_price):.2f}" if pd.notna(med_price) else "—")
            r1[2].metric("Sales", f"{int(sales):,}" if pd.notna(sales) else "—")
            r1[3].metric("Pos Rank", f"{int(pos_rank_val):,}" if pd.notna(pos_rank_val) else "—")

            r2 = st.columns(3)
            r2[0].metric("Min $", f"{float(min_price):.0f}" if pd.notna(min_price) else "—")
            r2[1].metric("Max $", f"{float(max_price):.0f}" if pd.notna(max_price) else "—")
            r2[2].metric(f"PPG ({ppg_season})", f"{float(ppg_val):.2f}" if pd.notna(ppg_val) else "—")

            r3 = st.columns(2)
            r3[0].metric("Drafts", f"{int(drafts_val):,}" if pd.notna(drafts_val) else "—")
            r3[1].metric("Team", sel_team if sel_team else "—")

        else:
            adp_val = sel_row.get("adp", np.nan)
            min_pick_val = sel_row.get("min_pick", np.nan)
            max_pick_val = sel_row.get("max_pick", np.nan)

            r1 = st.columns(4)
            r1[0].metric("ADP", f"{float(adp_val):.2f}" if pd.notna(adp_val) else "—")
            r1[1].metric(f"PPG ({ppg_season})", f"{float(ppg_val):.2f}" if pd.notna(ppg_val) else "—")
            r1[2].metric("Drafts", f"{int(drafts_val):,}" if pd.notna(drafts_val) else "—")
            r1[3].metric("Pos Rank", f"{int(pos_rank_val):,}" if pd.notna(pos_rank_val) else "—")

            r2 = st.columns(2)
            r2[0].metric("Min Pick", f"{float(min_pick_val):.0f}" if pd.notna(min_pick_val) else "—")
            r2[1].metric("Max Pick", f"{float(max_pick_val):.0f}" if pd.notna(max_pick_val) else "—")

    st.divider()

    g1, g2 = st.columns([1.2, 1.3], gap="large")

    with g1:
        if mode == "auction":
            st.markdown("#### Auction price distribution")
            if picks_subset is None or len(picks_subset) == 0:
                st.info("No auction purchases found for this player in the filtered draft set.")
            else:
                fig = plt.figure()
                plt.hist(picks_subset["amount"].values, bins=30)
                plt.xlabel("Auction price ($)")
                plt.ylabel("Count")
                st.pyplot(fig, clear_figure=True)
        else:
            st.markdown("#### Draft position distribution")
            if picks_subset is None or len(picks_subset) == 0:
                st.info("No pick rows found for this entity in the filtered draft set.")
            else:
                fig = plt.figure()
                plt.hist(picks_subset["pick_no_calc"].values, bins=30)
                plt.xlabel("Overall pick")
                plt.ylabel("Count")
                st.pyplot(fig, clear_figure=True)

    with g2:
        if mode == "auction":
            st.markdown("#### Avg auction price trend by month (last 5 months present)")
            if trend is None or trend.empty:
                st.info("No monthly price trend available.")
            else:
                fig2 = plt.figure()
                plt.plot(trend["start_month"].astype(str), trend["avg_price"].astype(float), marker="o")
                plt.xlabel("Draft month")
                plt.ylabel("Avg price ($)")
                st.pyplot(fig2, clear_figure=True)
                st.dataframe(trend, width="stretch", height=180)
        else:
            st.markdown("#### ADP trend by month (last 5 months present)")
            if trend is None or trend.empty:
                st.info("No monthly ADP trend available.")
            else:
                fig2 = plt.figure()
                plt.plot(trend["start_month"].astype(str), trend["adp"].astype(float), marker="o")
                plt.gca().invert_yaxis()
                plt.xlabel("Draft month")
                plt.ylabel("ADP (lower is earlier)")
                st.pyplot(fig2, clear_figure=True)
                st.dataframe(trend, width="stretch", height=180)

    st.divider()

    # IMPORTANT: imperative close + rerun to actually dismiss the dialog
    if st.button("Close", type="primary", key="dlg_close_btn"):
        close_player_dialog()
        st.rerun()


def render_board_clickable_tiles(
    mapping: Dict[Tuple[int, int], Dict[str, Any]],
    num_teams: int,
    num_rounds: int,
    title_line: str,
    *,
    mode: str,
    is_snake_board: bool,
) -> None:
    st.markdown(f'<div class="muted" style="margin: 0 0 8px 2px;">{title_line}</div>', unsafe_allow_html=True)

    hdr_cols = st.columns([120] + [142] * int(num_teams), gap="small")
    with hdr_cols[0]:
        st.markdown('<div class="hdr-cell hdr-round">Round</div>', unsafe_allow_html=True)
    for t in range(1, num_teams + 1):
        with hdr_cols[t]:
            st.markdown(f'<div class="hdr-cell">Team {t}</div>', unsafe_allow_html=True)

    for r in range(1, num_rounds + 1):
        cols = st.columns([120] + [142] * int(num_teams), gap="small")

        direction = "→" if (r % 2 == 1) else "←"
        with cols[0]:
            st.markdown(
                f"<div class='roundlbl'>Round {r} <span class='dirpill'>{direction}</span></div>",
                unsafe_allow_html=True,
            )

        for t in range(1, num_teams + 1):
            cell = mapping.get((r, t))
            with cols[t]:
                if not cell:
                    st.write("")
                    continue

                pid = safe_str(cell.get("player_id", ""))
                name = safe_str(cell.get("full_name", ""))
                pos = normalize_pos(cell.get("position", "UNK"))
                team = safe_str(cell.get("team", ""))
                bg = POSITION_COLORS.get(pos, POSITION_COLORS["UNK"])

                pos_rank = cell.get("pos_rank", None)
                if pd.notna(pos_rank) and str(pos_rank) != "":
                    rank_label = f"{POSITION_TEXT.get(pos, pos)}#{int(pos_rank)}"
                else:
                    rank_label = POSITION_TEXT.get(pos, pos)

                # Right pill: ADP boards show slot label (snake-aware); Auction shows $
                if mode == "auction":
                    val = cell.get("avg_price", np.nan)
                    right_pill = f"${float(val):.0f}" if pd.notna(val) else (team if team else "—")
                else:
                    if is_snake_board:
                        right_pill = format_pick_label_snake(r, t, int(num_teams))
                    else:
                        right_pill = format_pick_label_linear(r, t)

                is_rp = bool(cell.get("is_rookie_pick", False)) or (pos == "RDP") or pid.startswith("ROOKIE_PICK_")

                if is_rp:
                    top_html = "<div class='rp-top'>🔴</div>"
                else:
                    img = sleeper_headshot_url(pid)
                    top_html = f"<div class='card-top'><img src='{img}' loading='lazy' /></div>"

                bottom_html = (
                    f"<div class='card-bottom' style='background:{bg};'>"
                    f"  <div class='name'>{name}</div>"
                    f"  <div class='meta'>"
                    f"    <span class='pill'>{rank_label}</span>"
                    f"    <span class='pill'>{right_pill}</span>"
                    f"  </div>"
                    f"</div>"
                )

                st.markdown(f"<div class='pickcard'>{top_html}{bottom_html}</div>", unsafe_allow_html=True)

                st.button(
                    " ",
                    key=f"cellbtn_{mode}_{r}_{t}_{pid}",
                    type="secondary",
                    on_click=select_player,
                    args=(pid,),
                    help="Open player quick view",
                )


# ============================================================
# UI
# ============================================================
st.title("Sleeper Dynasty Board (ADP + Auction)")

st.session_state.setdefault("selected_pid", None)
st.session_state.setdefault("dialog_open", False)
st.session_state.setdefault("filter_sig", None)
st.session_state.setdefault("board_cache", {})  # dict[str, dict]

project_root_auto, raw_dir_auto, snapshots_dir_auto = pick_best_data_dir()

with st.sidebar:
    st.header("Options Menu")

    st.subheader("Data / Paths")
    st.caption(f"Auto project root:\n{project_root_auto}")
    raw_dir = st.text_input(
        "RAW dir",
        value=raw_dir_auto,
        help="Folder containing drafts/, picks/, leagues/",
    ).strip()

    st.divider()
    st.subheader("Board")
    season = st.number_input("Season", min_value=2015, max_value=2030, value=2026, step=1)

    board_kind = st.selectbox(
        "Board type",
        [
            "Startup ADP (snake)",
            "Rookie ADP (linear)",
            "Auction Price (snake)",
        ],
        index=0,
    )

    num_teams = st.number_input("League size", min_value=4, max_value=32, value=12, step=1)
    if board_kind.startswith("Rookie"):
        num_rounds = st.number_input("Rounds", min_value=1, max_value=20, value=5, step=1)
    else:
        num_rounds = st.number_input("Rounds", min_value=1, max_value=60, value=25, step=1)

    st.divider()

    # Startup-only options (kept; hidden for auction)
    if board_kind.startswith("Startup"):
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
    else:
        startup_inclusion_mode = "Include rookies (players)"
        kicker_placeholder_rounds = 4

    st.divider()
    st.subheader("Filters")

    filter_draft_status = st.multiselect(
        "Draft status",
        ["complete", "pre_draft", "drafting", "paused"],
        default=["complete"],
    )

    # Draft type defaults depend on board
    if board_kind.startswith("Auction"):
        default_types = ["auction"]
    elif board_kind.startswith("Rookie"):
        default_types = ["linear"]
    else:
        default_types = ["snake"]

    filter_draft_type = st.multiselect(
        "Draft type",
        ["snake", "linear", "auction"],
        default=default_types,
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
    min_drafts_per_month = st.slider("Min drafts per month", 0, 50, 5, 1)
    ppg_season = st.number_input("PPG season", 2015, 2030, 2025, 1)


# Load raw season
try:
    drafts, picks, leagues = load_raw_season(raw_dir, int(season))
except Exception as e:
    st.error(f"Failed to load RAW season files.\n\n{e}")
    close_player_dialog()
    st.stop()

# Date range
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
    st.caption("Click tiles to open player info. Board should not recompute on click.")

date_min = pd.Timestamp(datetime.combine(date_from, datetime.min.time()), tz="UTC")
date_max = pd.Timestamp(datetime.combine(date_to, datetime.max.time()), tz="UTC")

# Filter signature (drives cache + ensures dialog doesn’t pop open on filter changes)
filter_sig = compute_filter_sig(
    season=season,
    board_kind=board_kind,
    num_teams=num_teams,
    num_rounds=num_rounds,
    startup_inclusion_mode=startup_inclusion_mode,
    kicker_placeholder_rounds=kicker_placeholder_rounds,
    filter_draft_status=filter_draft_status,
    filter_draft_type=filter_draft_type,
    filter_scoring=filter_scoring,
    te_premium_only=te_premium_only,
    min_rounds=min_rounds,
    max_rounds=max_rounds,
    min_drafts_per_month=min_drafts_per_month,
    ppg_season=ppg_season,
    date_from=str(date_from),
    date_to=str(date_to),
)

prev_sig = st.session_state.get("filter_sig")
if prev_sig is None:
    st.session_state["filter_sig"] = filter_sig
elif prev_sig != filter_sig:
    close_player_dialog()
    st.session_state["filter_sig"] = filter_sig

# ============================================================
# ✅ FAST PATH: reuse computed board/pool if filter_sig unchanged
# ============================================================
cache = st.session_state["board_cache"].get(filter_sig)

if cache is None:
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

    if len(drafts_f) == 0:
        st.warning("No drafts match your filters. Relax filters and try again.")
        close_player_dialog()
        st.stop()

    players_df = load_players_df()

    try:
        ppg_df = load_ppg(int(ppg_season))
    except Exception as e:
        ppg_df = pd.DataFrame(columns=["player_id", "ppg", "games_played", "fantasy_pts"])
        st.warning(f"Could not load PPG for {ppg_season}: {e}")

    mode = "auction" if board_kind.startswith("Auction") else "adp"

    extra_meta = pd.DataFrame()
    picks_for_pool = picks.copy()
    include_positions = ["QB", "RB", "WR", "TE"]

    # Startup: optional rookie picks placeholders (ADP only)
    if (mode == "adp") and board_kind.startswith("Startup") and startup_inclusion_mode == "Include rookie picks (K placeholders)":
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

    # Compute pool
    if mode == "auction":
        pool = compute_player_auction_stats(
            picks=picks_for_pool,
            players_df=players_df,
            drafts_filtered=drafts_f,
            ppg_df=ppg_df,
            include_positions=["QB", "RB", "WR", "TE"],
        )
        if pool.empty:
            st.warning(
                "Auction board is empty.\n\n"
                "Most common causes:\n"
                "1) Your filtered drafts include no auction drafts\n"
                "2) Your picks parquet does not include md_amount (auction price)\n"
                "3) The selected date range excludes all auction drafts"
            )
            close_player_dialog()
            st.stop()
    else:
        pool = compute_player_pick_stats(
            picks=picks_for_pool,
            players_df=players_df,
            drafts_filtered=drafts_f,
            ppg_df=ppg_df,
            include_positions=include_positions,
            extra_meta=extra_meta,
        )

    pool = pool[pool["drafts"] >= required_player_drafts].copy()
    if pool.empty:
        st.warning(
            "No entities meet the minimum drafts requirement for the selected date range.\n\n"
            f"Try lowering 'Min drafts per month' (currently {min_drafts_per_month}) or widening the date range."
        )
        close_player_dialog()
        st.stop()

    pool_for_board = pool.copy()

    # Rookie board = current-season rookies only (ADP only)
    if mode == "adp" and board_kind.startswith("Rookie"):
        pool_for_board = filter_rookies_by_season(pool_for_board, int(season), keep_rookies=True)
        if pool_for_board.empty:
            st.warning("No rookie-eligible players (current class) meet the minimum drafts requirement.")
            close_player_dialog()
            st.stop()

    # Startup: season-aware exclusion of rookies/rookie picks (ADP only)
    if mode == "adp" and board_kind.startswith("Startup"):
        if startup_inclusion_mode == "Exclude rookies and rookie picks":
            pool_for_board = filter_rookies_by_season(pool_for_board, int(season), keep_rookies=False)
            if "is_rookie_pick" in pool_for_board.columns:
                pool_for_board = pool_for_board[~pool_for_board["is_rookie_pick"].astype(bool)].copy()

    # ✅ Contiguous positional ranks AFTER final filters
    pool_for_board["position"] = pool_for_board["position"].map(normalize_pos)

    if mode == "auction":
        pool_for_board["avg_price"] = pd.to_numeric(pool_for_board["avg_price"], errors="coerce")
        pool_for_board = pool_for_board.sort_values(["position", "avg_price"], ascending=[True, False]).reset_index(drop=True)
        pool_for_board["pos_rank"] = pool_for_board.groupby("position").cumcount() + 1
        mapping = build_board_map_snake_by_col(pool_for_board, int(num_teams), int(num_rounds), sort_col="avg_price", asc=False)
    else:
        pool_for_board["adp"] = pd.to_numeric(pool_for_board["adp"], errors="coerce")
        pool_for_board = pool_for_board.sort_values(["position", "adp"], ascending=[True, True]).reset_index(drop=True)
        pool_for_board["pos_rank"] = pool_for_board.groupby("position").cumcount() + 1

        if board_kind.startswith("Startup"):
            mapping = build_board_map_snake_by_col(pool_for_board, int(num_teams), int(num_rounds), sort_col="adp", asc=True)
        else:
            mapping = build_board_map_linear_by_col(pool_for_board, int(num_teams), int(num_rounds), sort_col="adp", asc=True)

    cache = {
        "mode": mode,
        "drafts_f": drafts_f,
        "picks_for_pool": picks_for_pool,
        "pool_for_board": pool_for_board,
        "mapping": mapping,
        "required_player_drafts": required_player_drafts,
        "num_months_in_scope": num_months_in_scope,
    }
    st.session_state["board_cache"][filter_sig] = cache

# Use cached artifacts (no recompute on tile click)
mode = cache["mode"]
drafts_f = cache["drafts_f"]
picks_for_pool = cache["picks_for_pool"]
pool_for_board = cache["pool_for_board"]
mapping = cache["mapping"]
required_player_drafts = cache["required_player_drafts"]
num_months_in_scope = cache["num_months_in_scope"]

# Summary metrics
m1, m2, m3 = st.columns(3)
m1.metric("Drafts", f"{len(drafts_f):,}")
m2.metric("Months in date range", f"{num_months_in_scope:,}")
m3.metric("Min drafts per entity", f"{required_player_drafts:,}")

# Render board
if mode == "auction":
    st.subheader("Auction Price Board")
    title_line = f"{board_kind} • Season {season} • {num_teams} teams × {num_rounds} rounds • ranked by Avg $"
else:
    st.subheader("ADP Board")
    title_line = f"{board_kind} • Season {season} • {num_teams} teams × {num_rounds} rounds • ranked by ADP"

# Startup + Auction are snake; Rookie is linear
is_snake_board = bool(board_kind.startswith("Startup") or board_kind.startswith("Auction"))

render_board_clickable_tiles(
    mapping,
    int(num_teams),
    int(num_rounds),
    title_line,
    mode=mode,
    is_snake_board=is_snake_board,
)

# Dialog (only computes tiny subsets + trend; does not rebuild board)
pid = st.session_state.get("selected_pid", None)
if pid and st.session_state.get("dialog_open", False):
    sel = pool_for_board[pool_for_board["player_id"].astype(str) == str(pid)].head(1)
    if sel.empty:
        st.warning("Selected entity is not in the current pool (filters may have changed). Click another square.")
        close_player_dialog()
    else:
        sel_row = sel.iloc[0]

        if mode == "auction":
            p_sub = picks_for_pool.copy()
            p_sub["draft_id"] = p_sub["draft_id"].astype(str)
            p_sub["player_id"] = p_sub["player_id"].astype(str)
            p_sub = p_sub[
                (p_sub["player_id"] == str(pid))
                & (p_sub["draft_id"].isin(set(drafts_f["draft_id"].astype(str))))
            ].copy()

            if "md_amount" in p_sub.columns:
                p_sub["amount"] = pd.to_numeric(p_sub["md_amount"], errors="coerce")
                p_sub = p_sub[p_sub["amount"].notna()].copy()
            else:
                p_sub = p_sub.iloc[0:0].copy()
                p_sub["amount"] = np.nan

            trend = player_monthly_trend_price(picks_for_pool, drafts_f, str(pid), last_n_months=5)
            show_player_dialog("auction", sel_row, int(ppg_season), p_sub, trend)

        else:
            p_sub = picks_for_pool.copy()
            p_sub["draft_id"] = p_sub["draft_id"].astype(str)
            p_sub["player_id"] = p_sub["player_id"].astype(str)
            p_sub = p_sub[
                (p_sub["player_id"] == str(pid))
                & (p_sub["draft_id"].isin(set(drafts_f["draft_id"].astype(str))))
            ].copy()

            p_sub["pick_no_calc"] = infer_pick_no(p_sub)
            p_sub["pick_no_calc"] = pd.to_numeric(p_sub["pick_no_calc"], errors="coerce")
            p_sub = p_sub[p_sub["pick_no_calc"].notna()].copy()

            trend = player_monthly_trend_adp(picks_for_pool, drafts_f, str(pid), last_n_months=5)
            show_player_dialog("adp", sel_row, int(ppg_season), p_sub, trend)

st.caption("Tip: drafts missing start_dt are excluded by date filters (invalid start_time).")
