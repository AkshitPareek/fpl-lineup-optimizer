"""
Team goal expectations from bookmaker odds.

Each Premier League match on football-data.co.uk carries average market odds for
home/draw/away and over/under 2.5 goals. Those are turned into the goals each
side was expected to score, and every club gets an attack and a defence rating
fitted to its recent matches, so any fixture can be priced. A match whose own
odds are already published is priced from them directly.
"""

import io
import os
import threading
import time
from datetime import datetime, timezone
from math import exp
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
UPCOMING_URL = "https://www.football-data.co.uk/fixtures.csv"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache", "odds")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
CACHE_SECONDS = 6 * 3600
ALIASES = {"Man United": "Man Utd", "Tottenham": "Spurs"}

# Poisson table: probability of home win, away win and over 2.5 goals for each
# pair of expected goals on a 0.02 grid.
_GRID = np.arange(0.15, 4.01, 0.02)
_GOALS = np.arange(11)


def _poisson_pmf(rates: np.ndarray) -> np.ndarray:
    fact = np.cumprod(np.concatenate([[1.0], np.arange(1, len(_GOALS))]))
    return np.exp(-rates[:, None]) * rates[:, None] ** _GOALS / fact


_PMF = _poisson_pmf(_GRID)
_JOINT = _PMF[:, None, :, None] * _PMF[None, :, None, :]  # [home rate, away rate, home goals, away goals]
_HOME_WIN = np.tril(np.ones((len(_GOALS), len(_GOALS))), -1)
_TABLE = np.stack([
    (_JOINT * _HOME_WIN).sum(axis=(2, 3)),
    (_JOINT * _HOME_WIN.T).sum(axis=(2, 3)),
    (_JOINT * (np.add.outer(_GOALS, _GOALS) > 2.5)).sum(axis=(2, 3)),
], axis=-1)


def goal_expectations(home: float, draw: float, away: float,
                      over: Optional[float] = None, under: Optional[float] = None) -> Optional[Tuple[float, float]]:
    """Expected goals for home and away implied by decimal odds, bookmaker margin removed."""
    try:
        implied = np.array([1 / float(home), 1 / float(draw), 1 / float(away)])
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not np.all(np.isfinite(implied)):
        return None
    implied /= implied.sum()
    target = [implied[0], implied[2]]
    weights = [1.0, 1.0]
    try:
        totals = np.array([1 / float(over), 1 / float(under)])
        if np.all(np.isfinite(totals)):
            target.append(totals[0] / totals.sum())
            weights.append(1.0)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    columns = len(target)
    error = ((_TABLE[..., :columns] - np.array(target)) ** 2 * np.array(weights)).sum(axis=-1)
    if columns == 2:  # no totals market: lean toward a typical 2.7 goals
        error += 0.002 * (_GRID[:, None] + _GRID[None, :] - 2.7) ** 2
    i, j = np.unravel_index(np.argmin(error), error.shape)
    return float(_GRID[i]), float(_GRID[j])


def match_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per match with date, team names and market expected goals."""
    rows = []
    for m in raw.itertuples(index=False):
        m = m._asdict()
        rates = goal_expectations(m.get("AvgH"), m.get("AvgD"), m.get("AvgA"), m.get("Avg_2_5"), m.get("Avg_2_5_1"))
        if rates is None or not isinstance(m.get("HomeTeam"), str):
            continue
        rows.append({
            "date": pd.to_datetime(m["Date"], dayfirst=True),
            "home": m["HomeTeam"], "away": m["AwayTeam"],
            "home_xg": rates[0], "away_xg": rates[1],
        })
    return pd.DataFrame(rows)


def _read_odds_csv(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), encoding_errors="replace")
    raw.columns = [c.lstrip("﻿") for c in raw.columns]
    if "Div" in raw.columns:
        raw = raw[raw["Div"] == "E0"]
    # namedtuples cannot carry '>' or '<'
    return raw.rename(columns={"Avg>2.5": "Avg_2_5", "Avg<2.5": "Avg_2_5_1"})


def team_ids(teams_df: pd.DataFrame, names: Iterable[str]) -> Dict[str, int]:
    """Map football-data club names onto FPL team ids ("Hull" -> "Hull City")."""
    fpl = dict(zip(teams_df["name"], teams_df["id"]))
    mapped = {}
    for name in set(names):
        target = ALIASES.get(name, name)
        if target in fpl:
            mapped[name] = int(fpl[target])
            continue
        match = [tid for fname, tid in fpl.items() if fname.startswith(target + " ") or target.startswith(fname + " ")]
        if len(match) == 1:
            mapped[name] = int(match[0])
    return mapped


class TeamRatings:
    """
    Attack and defence per club, fitted to market expected goals:
    log(goals) = base + home edge + own attack - opponent defence.
    """

    HALF_LIFE_DAYS = 120
    RIDGE = 2.0

    def __init__(self, matches: pd.DataFrame, teams_df: pd.DataFrame, as_of: pd.Timestamp,
                 upcoming: Optional[pd.DataFrame] = None):
        ids = team_ids(teams_df, list(matches["home"]) + list(matches["away"]) +
                       (list(upcoming["home"]) + list(upcoming["away"]) if upcoming is not None and len(upcoming) else []))
        self.teams = [int(t) for t in teams_df["id"]]
        index = {t: k for k, t in enumerate(self.teams)}
        past = matches[(matches["date"] < as_of)].copy()
        past["h"] = past["home"].map(ids)
        past["a"] = past["away"].map(ids)
        past = past.dropna(subset=["h", "a"])

        n = len(self.teams)
        rows, targets, weights = [], [], []
        for m in past.itertuples(index=False):
            w = 0.5 ** ((as_of - m.date).days / self.HALF_LIFE_DAYS)
            for scorer, conceder, home, rate in ((m.h, m.a, 1.0, m.home_xg), (m.a, m.h, 0.0, m.away_xg)):
                x = np.zeros(2 + 2 * n)
                x[0], x[1] = 1.0, home
                x[2 + index[int(scorer)]] = 1.0
                x[2 + n + index[int(conceder)]] = -1.0
                rows.append(x)
                targets.append(np.log(rate))
                weights.append(w)

        # Clubs with no recent top-flight matches start level with the weakest three.
        seen = set(past["h"].astype(int)) | set(past["a"].astype(int))
        prior = np.zeros(2 + 2 * n)
        penalty = np.full(2 + 2 * n, self.RIDGE)
        penalty[:2] = 1e-6
        if rows:
            X, y, w = np.array(rows), np.array(targets), np.array(weights)
            first = np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(penalty), X.T @ (w * y))
            strength = {t: first[2 + index[t]] + first[2 + n + index[t]] for t in seen}
            weakest = sorted(strength, key=strength.get)[:3]
            for t in self.teams:
                if t not in seen and weakest:
                    prior[2 + index[t]] = np.mean([first[2 + index[s]] for s in weakest])
                    prior[2 + n + index[t]] = np.mean([first[2 + n + index[s]] for s in weakest])
            theta = np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(penalty), X.T @ (w * y) + penalty * prior)
        else:
            theta = np.zeros(2 + 2 * n)
            theta[0] = np.log(1.35)
            theta[1] = 0.2
        self.base, self.home_edge = float(theta[0]), float(theta[1])
        self.attack = {t: float(theta[2 + index[t]]) for t in self.teams}
        self.defence = {t: float(theta[2 + n + index[t]]) for t in self.teams}
        self.matches_used = len(past)

        # Published odds for upcoming matches override the ratings for that pairing.
        self.fixture_odds: Dict[Tuple[int, int], Tuple[float, float]] = {}
        if upcoming is not None and len(upcoming):
            for m in upcoming.itertuples(index=False):
                h, a = ids.get(m.home), ids.get(m.away)
                if h is not None and a is not None:
                    self.fixture_odds[(h, a)] = (m.home_xg, m.away_xg)

    @property
    def average_goals(self) -> float:
        """Goals per side in a neutral match between average clubs."""
        return exp(self.base + self.home_edge / 2)

    def typical_goals(self, team: int) -> Tuple[float, float]:
        """Goals the team scores and concedes against an average club at a neutral ground."""
        edge = self.home_edge / 2
        return (exp(self.base + edge + self.attack.get(team, 0.0)),
                exp(self.base + edge - self.defence.get(team, 0.0)))

    def expected_goals(self, team: int, opponent: int, is_home: bool) -> Tuple[float, float]:
        """Goals the team is expected to score and to concede in this fixture."""
        key = (team, opponent) if is_home else (opponent, team)
        if key in self.fixture_odds:
            home_xg, away_xg = self.fixture_odds[key]
            return (home_xg, away_xg) if is_home else (away_xg, home_xg)
        edge = self.home_edge if is_home else 0.0
        other = 0.0 if is_home else self.home_edge
        scored = exp(self.base + edge + self.attack.get(team, 0.0) - self.defence.get(opponent, 0.0))
        conceded = exp(self.base + other + self.attack.get(opponent, 0.0) - self.defence.get(team, 0.0))
        return scored, conceded


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


_lock = threading.Lock()
_memo: Dict[str, Tuple[float, object]] = {}


def _fetch(url: str, cache_name: str, max_age: float) -> Optional[str]:
    path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        text = response.content.decode("utf-8-sig", errors="replace")
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except requests.RequestException as exc:
        print(f"Odds download failed for {url}: {exc}")
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        return None


def live_ratings(teams_df: pd.DataFrame, fixtures: List[Dict]) -> Optional[TeamRatings]:
    """Ratings for the season being played, or None when the odds cannot be fetched."""
    kickoffs = sorted(f["kickoff_time"] for f in fixtures or [] if f.get("kickoff_time"))
    if not kickoffs:
        return None
    start_year = int(kickoffs[0][:4])
    with _lock:
        cached = _memo.get("live")
        if cached and time.time() - cached[0] < CACHE_SECONDS:
            return cached[1]
        frames = []
        for year, max_age in ((start_year - 1, 30 * 86400), (start_year, CACHE_SECONDS)):
            text = _fetch(DATA_URL.format(code=season_code(year)), f"E0_{season_code(year)}.csv", max_age)
            if text:
                frames.append(match_frame(_read_odds_csv(text)))
        if not frames:
            return None
        upcoming_text = _fetch(UPCOMING_URL, "fixtures.csv", CACHE_SECONDS)
        upcoming = match_frame(_read_odds_csv(upcoming_text)) if upcoming_text else None
        ratings = TeamRatings(pd.concat(frames, ignore_index=True), teams_df,
                              pd.Timestamp(datetime.now(timezone.utc).date()), upcoming)
        _memo["live"] = (time.time(), ratings)
        return ratings


class SeasonOdds:
    """A completed season's odds, able to price each gameweek as it stood before the deadline."""

    def __init__(self, season: str):
        start = int(season[:4])
        self.frames = []
        for year in (start - 1, start):
            code = season_code(year)
            text = _fetch(DATA_URL.format(code=code), f"E0_{code}.csv", 365 * 86400)
            if text is None:
                raise RuntimeError(f"No odds for {year}/{year + 1}")
            self.frames.append(match_frame(_read_odds_csv(text)))
        self.matches = pd.concat(self.frames, ignore_index=True)

    def ratings(self, teams_df: pd.DataFrame, fixtures: List[Dict], gameweek: int) -> TeamRatings:
        """Fitted to matches before the gameweek, plus its own matches' published odds."""
        week = [f for f in fixtures if f.get("event") == gameweek]
        as_of = pd.Timestamp(min(f["kickoff_time"] for f in week)[:10])
        this_season = self.frames[1]
        ids = team_ids(teams_df, list(this_season["home"]) + list(this_season["away"]))
        pairs = {(f["team_h"], f["team_a"]) for f in week}
        mask = [(ids.get(h), ids.get(a)) in pairs for h, a in zip(this_season["home"], this_season["away"])]
        return TeamRatings(self.matches, teams_df, as_of, this_season[mask])
