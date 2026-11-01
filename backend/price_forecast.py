"""
Chance that a player's price moves before the next deadline.

FPL moves prices on net transfers. Rises follow the raw number of net
transfers in; falls follow the share of owners selling. Two logistic models
were fit on the 2023/24-2025/26 per-gameweek archives (vaastav): one gameweek's
net transfers and the owners at the start of it, against whether the price had
moved by the next deadline. Holding out one season at a time, rises scored AUC
0.97-0.98 and falls 0.77-0.81; of the calls at 50% or more, 75-84% of rises
and 58-67% of falls happened.

Live, only the transfers made so far this gameweek are known, so early in the
week the chance is an underestimate. That use is not validated.
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# intercept, log(1 + net in / 10k), log(1 + net out / 10k), net / owners
RISE_COEFS = (-6.48, 2.38, 0.89, -0.96)
FALL_COEFS = (-3.51, 0.16, 0.83, -6.16)
ALERT_CHANCE = 0.5


def _logistic(coefs, net: float, owners: float) -> float:
    owners = max(owners, 1000.0)
    x = (1.0,
         math.log1p(max(net, 0.0) / 1e4),
         math.log1p(max(-net, 0.0) / 1e4),
         max(-1.0, min(1.0, net / owners)))
    z = sum(c * v for c, v in zip(coefs, x))
    return 1.0 / (1.0 + math.exp(-z))


def price_change_chance(net_transfers: float, owners_at_last_deadline: float) -> Tuple[float, float]:
    """(chance of a rise, chance of a fall) over a gameweek with these net transfers."""
    return (_logistic(RISE_COEFS, net_transfers, owners_at_last_deadline),
            _logistic(FALL_COEFS, net_transfers, owners_at_last_deadline))


def _num(value) -> float:
    try:
        return 0.0 if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return 0.0


def price_alerts(players_df: pd.DataFrame, total_players: Optional[int],
                 squad_ids: Iterable[int], buy_ids: Iterable[int]) -> List[Dict]:
    """
    Squad players likely to fall and planned buys likely to rise before the deadline.

    Moving before the change keeps 0.1m. A player whose price already moved that
    way this gameweek is left out: a second move in one gameweek is rare (5% of
    rises, under 1% of falls in the archives).
    """
    if not total_players or players_df.empty:
        return []
    rows = players_df.set_index("id")
    alerts = []
    for pid, direction in [(j, "fall") for j in squad_ids] + [(j, "rise") for j in buy_ids]:
        if pid not in rows.index:
            continue
        p = rows.loc[pid]
        net = _num(p.get("transfers_in_event")) - _num(p.get("transfers_out_event"))
        owners_now = _num(p.get("selected_by_percent")) / 100.0 * total_players
        rise, fall = price_change_chance(net, owners_now - net)
        moved = _num(p.get("cost_change_event"))
        chance = rise if direction == "rise" else fall
        if chance < ALERT_CHANCE or (moved > 0 if direction == "rise" else moved < 0):
            continue
        alerts.append({
            "id": int(pid),
            "web_name": p.get("web_name"),
            "direction": direction,
            "chance": round(chance, 2),
            "net_transfers": int(net),
            "now_cost": _num(p.get("now_cost")) / 10.0,
        })
    return sorted(alerts, key=lambda a: -a["chance"])
