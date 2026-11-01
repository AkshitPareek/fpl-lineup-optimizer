# Telemetry

The API prints one JSON line per request, and one per frontend event, to stdout.
Cloud Run sends stdout to Cloud Logging, which stores each line's fields under
`jsonPayload`. Nothing else is involved: no third-party service, no cookies.

## What is logged

`kind="request"`: one line per API call (CORS preflights are skipped).

| field | meaning |
|---|---|
| `endpoint` | route template, e.g. `/api/manager-chips/{manager_id}` |
| `method`, `status`, `duration_ms` | as served; `severity` is ERROR for 5xx, WARNING for 4xx |
| `commit`, `revision` | `GIT_COMMIT` from the deploy script, Cloud Run's `K_REVISION` |
| `manager` | salted hash of the manager ID (path or JSON body), never the ID |
| `error_type`, `error` | exception class behind a 500, and its message with 5+ digit runs replaced by `#` |

Planner calls (`/api/optimize/compare`, `/api/optimize/multi-period`) add:
`horizon`, `plan_chips`, `strategy`, `robust`, `forced_chip`, `linked`;
`solves` (planner runs, including re-solves and fallbacks), `resolves` (re-solves
around a Free Hit or changed chip weeks), `fallbacks` (plans that dropped chip weeks
after running out of time); `with_hits` / `no_hits` (compare) or `plan`
(multi-period), each `{status, proven_optimal, solve_seconds, chips, hits,
deadline_transfers}`; and `timed_out` (`with_hits` gives a 504, `no_hits` a partial
result). `proven_optimal: false` means the solver stopped at `SOLVER_TIME_LIMIT`.
`solve_seconds` is the planner's last run only; `duration_ms` covers everything.
For the streaming backtest, `duration_ms` ends when the stream starts.

`kind="client"`: frontend events posted to `/api/telemetry` by
`frontend/src/lib/telemetry.js`. Fields: `event`, `session` (random, per tab, in
sessionStorage), `t_ms` (ms since the page loaded), `props`.

| event | props |
|---|---|
| `page_view` | `tab`, `mobile` (width under 768px), `returning` (a manager ID was remembered) |
| `tab_view` | `tab` |
| `manager_entered` | `source`: `stored` or `typed` |
| `manager_loaded` | `ok`, `synced` (false: FPL had no squad for the ID, often a typo), `kind` on failure |
| `plan_requested` | `view` (planner, quick), `horizon`, `strategy`, `robust`, `forced_chip`, `linked`, `auto` (ran on load) |
| `plan_shown` | the above plus `ms` (click to result), `best`, `proven_optimal`, `partial`, `transfers`, `hits`, `chip_advice` (play, save, hold, none) |
| `error_shown` | `where` (planner, quick, dream, backtest, app), `kind` (`http_504`, `network`, `unsolved`...), `message`, `ms` |
| `strategy_switched`, `gameweek_viewed`, `view_toggled` | `to`, `index`, `view` |
| `transfer_why_opened`, `chip_selected`, `chip_suggestion_applied` | `chip` where relevant |
| `dream_team_requested`, `backtest_requested`, `backtest_shown` | `weeks`, `horizon`, `ms` |

The endpoint accepts at most 8 KB and 20 events per post, only the events above,
and only flat string, number, boolean or null props (strings cut to 120 chars,
long digit runs scrubbed). Anything else is dropped or rejected with 400/413.

Privacy: the browser sends nothing under Do Not Track or Global Privacy Control,
sends no manager ID, and omits credentials. Events are batched and held while an
API call is in flight, because the service runs one request per instance on at
most two instances and a telemetry post must not take the slot a plan needs.

## Setting the salt (once)

Without `TELEMETRY_SALT` each instance makes its own salt, so a manager's hash
changes between instances and restarts. Manager IDs are small numbers, so the salt
must stay secret or the hashes can be reversed by trying every ID. Set it once:

```sh
gcloud run services update fpl-optimizer-api --region asia-south1 \
  --update-env-vars TELEMETRY_SALT="$(openssl rand -hex 16)"
```

`deploy_cloud_run.sh` uses `--update-env-vars`, so later deploys keep it.

## Reading it

Summary of the last N days (endpoints with p50/p95 latency and 5xx rate, solver
trouble, horizons, funnel from page view to plan shown, time to plan, errors shown,
mobile share, requests per commit):

```sh
cd backend && python telemetry_report.py --days 7          # --json for machine output
```

Ready queries (add `--project fpl-optimizer-api` if it is not your default):

```sh
SVC='resource.type="cloud_run_revision" AND resource.labels.service_name="fpl-optimizer-api"'

# Every request line from the last day, newest first
gcloud logging read "$SVC AND jsonPayload.kind=\"request\"" --freshness=1d --limit=50 \
  --format='table(timestamp, jsonPayload.endpoint, jsonPayload.status, jsonPayload.duration_ms)'

# Server errors with their exception type
gcloud logging read "$SVC AND jsonPayload.kind=\"request\" AND jsonPayload.status>=500" --freshness=7d \
  --format='table(timestamp, jsonPayload.endpoint, jsonPayload.error_type, jsonPayload.error)'

# Plans the solver did not prove optimal, or that timed out or fell back
gcloud logging read "$SVC AND jsonPayload.horizon>0 AND (jsonPayload.with_hits.proven_optimal=false OR jsonPayload.plan.proven_optimal=false OR jsonPayload.timed_out:* OR jsonPayload.fallbacks>0)" \
  --freshness=7d --format='table(timestamp, jsonPayload.horizon, jsonPayload.duration_ms, jsonPayload.timed_out, jsonPayload.fallbacks)'

# Slow plans (over 30 s)
gcloud logging read "$SVC AND jsonPayload.endpoint=\"/api/optimize/compare\" AND jsonPayload.duration_ms>30000" --freshness=7d \
  --format='table(timestamp, jsonPayload.horizon, jsonPayload.duration_ms, jsonPayload.solves)'

# Errors users saw in the browser
gcloud logging read "$SVC AND jsonPayload.event=\"error_shown\"" --freshness=7d \
  --format='table(timestamp, jsonPayload.props.where, jsonPayload.props.kind, jsonPayload.props.message)'

# Browser time from "Generate plan" to the plan on screen
gcloud logging read "$SVC AND jsonPayload.event=\"plan_shown\"" --freshness=7d \
  --format='table(timestamp, jsonPayload.props.view, jsonPayload.props.horizon, jsonPayload.props.ms, jsonPayload.props.auto)'

# One session's events in order
gcloud logging read "$SVC AND jsonPayload.session=\"SESSION_ID\"" --freshness=7d --order=asc \
  --format='table(timestamp, jsonPayload.event, jsonPayload.props)'

# Requests served by one deploy
gcloud logging read "$SVC AND jsonPayload.commit=\"fd16c2c\"" --freshness=7d --limit=20
```

In the Cloud Console's Logs Explorer the same filters work without the shell
quoting, e.g. `jsonPayload.kind="client" jsonPayload.event="plan_shown"`. A
log-based metric on `jsonPayload.duration_ms` (distribution) gives latency charts.

Cloud Logging keeps `_Default` bucket logs for 30 days, so the report covers at
most the last 30 days.
