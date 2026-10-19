"""Price paths from the transfer market, with no fitted model.

A rise takes net transfers of about 7.5% of a player's owners, and a fall about
5%, never fewer than 10,000. Transfers so far this gameweek are projected to the
deadline, then that imbalance halves each week after, because a run of transfers
does not continue at full strength. Prices are in tenths of a million.
"""

from datetime import datetime, timezone

RISE_OWNER_FRACTION = 0.075
FALL_OWNER_FRACTION = 0.05
MIN_PRICE_STEP = 10000
MAX_WEEKLY_CHANGE = 3
PRICE_FLOOR = 38
PRICE_CAP = 160
# How quickly this gameweek's transfer imbalance fades in later weeks.
LATER_WEEK_DECAY = 0.5


def transfer_elapsed(events, now=None) -> float:
    """Fraction of the current gameweek's transfer window that has passed, from 0.05 to 1."""
    now = now or datetime.now(timezone.utc)
    ordered = sorted((e for e in events or [] if e.get("deadline_time")), key=lambda e: e.get("id") or 0)
    nxt = next((e for e in ordered if e.get("is_next")), None)
    if nxt is None:
        return 1.0
    prev = next((e for e in ordered if e.get("id") == nxt["id"] - 1), None)
    if prev is None:
        return 1.0

    def parsed(event):
        text = str(event["deadline_time"]).replace("Z", "+00:00")
        moment = datetime.fromisoformat(text)
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)

    span = (parsed(nxt) - parsed(prev)).total_seconds()
    if span <= 0:
        return 1.0
    return min(1.0, max(0.05, (now - parsed(prev)).total_seconds() / span))


def _number(player, key, default=0.0):
    value = player.get(key) if hasattr(player, "get") else None
    try:
        if value is None or value != value:  # NaN
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_purchase_price(sell_tenths, now_tenths, season_start_tenths=None):
    """
    Purchase price consistent with the selling price.

    A rise keeps half, rounded down, so the purchase price is recoverable. After a fall the
    selling price equals the current price and the purchase price is at least the season start.
    """
    sell, now = int(sell_tenths), int(now_tenths)
    if sell < now:
        # Half the rise, rounded down, so the purchase price sits at or below the selling price.
        matched = [
            purchase for purchase in range(max(0, 2 * sell - now), now + 1)
            if purchase + (now - purchase) // 2 == sell
        ]
        # The higher purchase is the smaller rise, which is the one a later buy would have.
        return max(matched) if matched else sell
    start = int(season_start_tenths) if season_start_tenths else now
    return max(now, start)


def sale_proceeds(purchase, now) -> int:
    """Selling price, in tenths, after buying at `purchase` when the price is now `now`.

    A rise keeps half, rounded down. A fall is lost in full.
    """
    purchase, now = int(purchase), int(now)
    if now <= purchase:
        return now
    return purchase + (now - purchase) // 2


def guaranteed_sale(path, week_index) -> int:
    """
    What a sale in this week raises for a player bought earlier in the plan.

    The plan chooses the buy week. Counting the lowest proceeds of any earlier price
    keeps half of a rise and all of a fall, and the move stays affordable.
    """
    now = int(path[week_index])
    earlier = path[:week_index]
    if not earlier:
        return now
    return min(sale_proceeds(purchase, now) for purchase in earlier)


def _steps(net, owners):
    if net >= 0:
        return int(net // max(MIN_PRICE_STEP, owners * RISE_OWNER_FRACTION))
    return -int((-net) // max(MIN_PRICE_STEP, owners * FALL_OWNER_FRACTION))


def forecast_price_path(player, weeks, total_managers, elapsed=1.0):
    """
    Price in tenths at each of `weeks` deadlines.

    Flat when the transfer columns or the size of the game are missing.
    """
    now = int(_number(player, "now_cost"))
    if weeks <= 0 or total_managers <= 0:
        return [now] * max(0, weeks)
    net = _number(player, "transfers_in_event") - _number(player, "transfers_out_event")
    projected = net / min(1.0, max(0.05, elapsed))
    owners = max(1.0, _number(player, "selected_by_percent") / 100.0 * total_managers)
    already = int(_number(player, "cost_change_event"))
    price = now
    path = []
    for week in range(weeks):
        change = _steps(projected, owners)
        if week == 0:
            change -= already
        change = max(-MAX_WEEKLY_CHANGE, min(MAX_WEEKLY_CHANGE, change))
        price = max(PRICE_FLOOR, min(PRICE_CAP, price + change))
        path.append(price)
        projected *= LATER_WEEK_DECAY
    return path
