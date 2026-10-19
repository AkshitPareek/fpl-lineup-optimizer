"""Expected points from automatic substitutions.

A starter who plays no minutes is replaced by the first bench player, in the order the
solver chooses. Goalkeepers only replace goalkeepers. The chance of no minutes comes from
the flagged chance of playing and how many minutes the player has been getting; it is not
a fitted model. The expected number of outfield blanks fills the bench in order: the first substitute is
used for the first blank, the second for the next, each at most once. A starter flagged
as certain not to play therefore brings the first substitute on for certain.
"""

DOUBT_RECOVERY_PER_GW = 0.5
# Averaging this many minutes a gameweek means the player appears.
APPEARANCE_MINUTES = 60


def _number(player, key):
    value = player.get(key) if hasattr(player, "get") else None
    try:
        if value is None or value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def blank_probability(player, gameweek, current_gw) -> float:
    """Probability a selected player plays zero minutes in this gameweek."""
    chance = _number(player, "chance_of_playing_next_round")
    if chance is None:
        available = 1.0
    else:
        chance /= 100.0
        if chance <= 0:
            return 1.0
        weeks_ahead = max(0, gameweek - current_gw)
        available = 1.0 - (1.0 - chance) * DOUBT_RECOVERY_PER_GW ** weeks_ahead
    recent = _number(player, "recent_4_minutes")
    if recent is not None:
        average = recent / 4.0
    else:
        played = max(1, current_gw - 1)
        minutes = _number(player, "minutes")
        # No minutes history at all: nothing says the player will blank.
        if minutes is None:
            return 1.0 - available
        average = minutes / played
    appears = min(1.0, average / APPEARANCE_MINUTES)
    return 1.0 - appears * available


def slot_weight(mean, slot) -> float:
    """How often the slot-th outfield substitute plays, given `mean` expected blanks."""
    return min(1.0, max(0.0, mean - (slot - 1)))


def autosub_points(starters, bench, xp, blank, position) -> float:
    """
    Expected points the bench adds for this fixed lineup.

    `starters` and `bench` are player ids. The bench is ordered: the goalkeeper first if
    present, then outfield substitutes in the order given. Outfield order should already
    be best-first, which is descending expected points.
    """
    starter_ids = list(starters)
    bench_ids = list(bench)
    gk_starter = next((j for j in starter_ids if position(j) == 1), None)
    gk_bench = next((j for j in bench_ids if position(j) == 1), None)
    added = 0.0
    if gk_starter is not None and gk_bench is not None:
        added += blank(gk_starter) * xp(gk_bench)
    out_starters = [j for j in starter_ids if position(j) != 1]
    out_bench = [j for j in bench_ids if position(j) != 1]
    mean = sum(blank(j) for j in out_starters)
    for slot, player in enumerate(out_bench[:3], start=1):
        added += slot_weight(mean, slot) * xp(player)
    return added
