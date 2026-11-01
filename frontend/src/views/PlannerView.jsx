import { useEffect, useMemo, useRef, useState } from 'react'
import { api, friendlyError, withRetry } from '../lib/api'
import { track } from '../lib/telemetry'
import { cx } from '../lib/cx'
import { chipKey, chipLabel, CHIP_LABELS, clockTime, countdown, num, signed, timeAgo } from '../lib/format'
import {
  captaincy, fixturesForGameweek, freeTransferLedger, newsChanges, planFor, playerRange,
  recommendedStrategy, strategiesIdentical, weekRange, benchOrder,
} from '../lib/plan'
import {
  Badge, Button, Card, CardHeader, ConfidenceBadge, EmptyState, ErrorNote, Field,
  SectionLabel, Segmented, Select, Spinner, Stepper, Toggle,
} from '../components/ui'
import { IconBan, IconCrown, IconLock, IconRoute, IconShield, IconX } from '../components/icons'
import ManagerFields from '../components/ManagerFields'
import SquadView, { ViewToggle } from '../components/SquadView'
import TransferPanel from '../components/TransferPanel'
import DeadlineCard from '../components/DeadlineCard'
import PlanTable from '../components/PlanTable'
import ChipStrip from '../components/ChipStrip'
import PlanLoading from '../components/PlanLoading'
import { Canvas, Workspace } from './Workspace'

// HiGHS solves up to 15 gameweeks; longer horizons may stop at the time limit.
const MAX_HORIZON = 15
// A horizon that usually solves well inside the host's time limit.
const SHORT_HORIZON = 5
// Past this the request is abandoned; the API gives up on a solve after 75 s.
const PLAN_TIMEOUT_MS = 150000

const LAST_PLAN_KEY = 'fpl-last-plan'
const PREFS_KEY = 'fpl-player-prefs'
// A remembered plan is shown for up to a day; it is labelled with its age.
const LAST_PLAN_MAX_AGE_MS = 24 * 3600 * 1000

const STRATEGIES = [
  { value: 'standard', label: 'Most points', hint: 'Maximise expected points.' },
  { value: 'differential', label: 'Differentials', hint: 'Favours players owned by under 10% of managers, to climb faster (and fall faster).' },
  { value: 'template', label: 'Template', hint: 'Favours players owned by 20% or more, to protect a rank.' },
]

function readJson(key) {
  try {
    return JSON.parse(localStorage.getItem(key) || 'null')
  } catch {
    return null
  }
}

function writeJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Storage full or blocked: the plan just isn't remembered.
  }
}

function ChipPicker({ startGw, horizon, available, recs, selected, onChange }) {
  const endGw = startGw + horizon - 1
  const chips = Array.isArray(available) ? available.map(chipKey) : Object.keys(CHIP_LABELS)
  const rec = selected && recs?.recommendations?.[selected.chip]

  const pick = chip => {
    track('chip_selected', { chip: chip || 'none' })
    if (!chip) return onChange(null)
    const suggested = recs?.recommendations?.[chip]?.recommended_gw
    const gw = suggested >= startGw && suggested <= endGw ? suggested : startGw
    onChange({ chip, gw })
  }

  return (
    <Field
      label="Chip"
      htmlFor="chip-select"
      hint={rec?.reason || (Array.isArray(available)
        ? `Chips left: ${available.map(chipLabel).join(', ') || 'none'}. Leave this on "Let the plan decide" unless you want to force one.`
        : 'Leave this on "Let the plan decide" unless you want to force one.')}
    >
      <div className="flex gap-2">
        <div className="flex-1">
          <Select id="chip-select" value={selected?.chip || ''} onChange={e => pick(e.target.value)}>
            <option value="">Let the plan decide</option>
            {chips.map(c => <option key={c} value={c}>Force {chipLabel(c)}</option>)}
          </Select>
        </div>
        {selected && (
          <div className="w-28">
            <Select value={selected.gw} onChange={e => onChange({ ...selected, gw: Number(e.target.value) })} aria-label="Chip gameweek">
              {Array.from({ length: horizon }, (_, i) => startGw + i).map(gw => <option key={gw} value={gw}>GW{gw}</option>)}
            </Select>
          </div>
        )}
      </div>
    </Field>
  )
}

/** Hit vs free-transfer plans. When no hit is taken they are one plan, so say so once. */
function StrategyCompare({ compare, selected, onSelect }) {
  const c = compare.comparison
  const best = recommendedStrategy(compare)
  if (strategiesIdentical(compare)) {
    return (
      <div className="rounded-xl bg-ink-900/60 p-3.5 ring-1 ring-inset ring-ink-700">
        <div className="text-xs font-semibold text-fg-2">No hits needed</div>
        <div className="mt-0.5 text-[11px] text-fg-3">The best plan uses free transfers only, so there is nothing to compare.</div>
      </div>
    )
  }
  const options = [
    { key: 'with_hits', title: 'With hits', detail: c.with_hits?.hit_cost ? `${num(c.with_hits.total_xp)} pts − ${c.with_hits.hit_cost} in hits` : 'Takes no hits', data: c.with_hits },
    { key: 'no_hits', title: 'No hits', detail: 'Free transfers only', data: c.no_hits },
  ]
  return (
    <div className="grid grid-cols-2 gap-3">
      {options.map(o => {
        const feasible = o.data?.feasible !== false && Boolean(planFor(compare, o.key))
        return (
          <button
            key={o.key}
            type="button"
            onClick={() => onSelect(o.key)}
            disabled={!feasible}
            aria-pressed={selected === o.key}
            className={cx(
              'rounded-xl p-3.5 text-left transition ring-1 ring-inset disabled:opacity-60 focus-visible:outline-2 focus-visible:outline-fpl-green',
              feasible && 'cursor-pointer',
              selected === o.key ? 'bg-fpl-green/[0.07] ring-fpl-green/60' : 'bg-ink-900/60 ring-ink-700 enabled:hover:ring-ink-600',
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold text-fg-2">{o.title}</span>
              {o.key === best && <Badge tone="green">Best</Badge>}
            </div>
            {feasible ? (
              <>
                <div className="mt-1 font-display text-2xl font-semibold tabular">{num(o.data.net_xp)}<span className="ml-1 text-sm text-fg-3 font-sans font-normal">net pts</span></div>
                <div className="text-[11px] text-fg-3 tabular">{o.detail}</div>
              </>
            ) : (
              <>
                <div className="mt-1 font-display text-lg font-semibold text-fg-2">Unavailable</div>
                <div className="text-[11px] text-fg-3">{o.key === 'no_hits' ? 'Needs more moves than your free transfers allow' : "The solver couldn't find a plan"}</div>
              </>
            )}
          </button>
        )
      })}
    </div>
  )
}

const CONFIDENCE_HELP = 'How far the captain is ahead of the next best pick: high 2+ points, medium 1-2, low under 1.'

function confidenceFromMargin(margin) {
  if (margin == null) return null
  return margin >= 2 ? 'high' : margin >= 1 ? 'medium' : 'low'
}

function CaptainCard({ gwPlan, options }) {
  const { captain, vice, next, margin } = captaincy(gwPlan)
  if (!captain) return null
  const range = playerRange(captain, gwPlan.gameweek)
  const fixtures = gwPlan.captain_recommendation?.player_id === captain.id ? gwPlan.captain_recommendation.fixtures : null
  const list = Array.isArray(options) ? options : []
  return (
    <Card className="p-4">
      <CardHeader
        icon={<IconCrown />}
        title="Captain"
        action={<span title={CONFIDENCE_HELP}><ConfidenceBadge level={confidenceFromMargin(margin)} /></span>}
      />
      <div className="mt-3 flex items-baseline justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate font-display text-lg font-semibold">{captain.web_name}</div>
          {fixtures && <div className="text-xs text-fg-3">{fixtures}</div>}
        </div>
        <div className="text-right shrink-0">
          <div className="font-display text-xl font-semibold text-fpl-green tabular">{num(captain.expected_points)}</div>
          <div className="text-[11px] text-fg-3 tabular">{range ? `likely ${num(range.low, 0)}–${num(range.high, 0)}` : 'xP'}</div>
        </div>
      </div>
      {next && margin != null && (
        <p className="mt-2 text-xs text-fg-2 tabular">
          {signed(margin)} pts ahead of {next.web_name} ({num(next.expected_points)})
          <span className="block text-[11px] text-fg-3">{CONFIDENCE_HELP}</span>
        </p>
      )}
      {vice && (
        <p className="mt-1 text-xs text-fg-3">Vice-captain <span className="text-fg-2 font-medium">{vice.web_name}</span> <span className="tabular">{num(vice.expected_points)} xP</span></p>
      )}
      {list.length > 1 && (
        <div className="mt-3 border-t border-ink-700/70 pt-3">
          <SectionLabel className="mb-2">Against the field</SectionLabel>
          <table className="w-full text-xs tabular">
            <thead className="text-[10px] uppercase tracking-wider text-fg-3">
              <tr>
                <th scope="col" className="pb-1 text-left font-semibold">Player</th>
                <th scope="col" className="pb-1 text-right font-semibold">xP</th>
                <th scope="col" className="pb-1 text-right font-semibold" title="Effective ownership: the share of managers who gain when he scores, counting captains twice. An estimate.">EO</th>
                <th scope="col" className="pb-1 text-right font-semibold" title="How far each point he scores moves you against an average rival when you captain him. Low protects a rank, high chases one.">Swing</th>
              </tr>
            </thead>
            <tbody>
              {list.map(o => (
                <tr key={o.id} className="border-t border-ink-700/50">
                  <td className="py-1 text-fg-2">{o.web_name}{o.behind_best > 0 && <span className="text-fg-3"> −{num(o.behind_best)}</span>}</td>
                  <td className="py-1 text-right text-fg">{num(o.expected_points)}</td>
                  <td className="py-1 text-right text-fg-2">{o.effective_ownership != null ? `${num(o.effective_ownership, 0)}%` : '–'}</td>
                  <td className="py-1 text-right text-fg-2">{num(o.swing, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1.5 text-[10px] text-fg-3">Expected points still pick the captain. Swing shows the risk against the field: low protects a rank, high chases one.</p>
        </div>
      )}
    </Card>
  )
}

/** Kept and banned players, removable, with a nudge when they differ from the plan shown. */
function PlayerPrefs({ prefs, onRemove, stale }) {
  if (!prefs.locked.length && !prefs.excluded.length) {
    return <p className="text-[11px] text-fg-3">Tap a player on the pitch to keep him or rule him out.</p>
  }
  const chip = (p, kind) => (
    <li key={`${kind}-${p.id}`}>
      <button
        type="button"
        onClick={() => onRemove(p.id, kind)}
        className={cx(
          'inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium ring-1 ring-inset cursor-pointer focus-visible:outline-2 focus-visible:outline-fpl-green',
          kind === 'locked' ? 'bg-ink-700 text-fg ring-ink-600' : 'bg-bad/10 text-[#ffb3ba] ring-bad/30',
        )}
        aria-label={`${kind === 'locked' ? 'Stop keeping' : 'Allow'} ${p.name}`}
      >
        {kind === 'locked' ? <IconLock size={11} /> : <IconBan size={11} />}
        {p.name}
        <IconX size={11} className="text-fg-3" />
      </button>
    </li>
  )
  return (
    <div>
      <ul className="flex flex-wrap gap-1.5">
        {prefs.locked.map(p => chip(p, 'locked'))}
        {prefs.excluded.map(p => chip(p, 'excluded'))}
      </ul>
      {stale && <p className="mt-1.5 text-[11px] text-warn">Generate the plan again to apply these.</p>}
    </div>
  )
}

function PlanError({ error, horizon, onRetry, onShorter }) {
  const [open, setOpen] = useState(false)
  return (
    <EmptyState title="The plan didn't finish" action={
      <div className="flex flex-wrap justify-center gap-2">
        <Button onClick={onRetry}>Try again</Button>
        {error.shorter && horizon > SHORT_HORIZON && <Button variant="secondary" onClick={onShorter}>Retry with {SHORT_HORIZON} GWs</Button>}
      </div>
    }>
      {error.message}
      {error.detail && (
        <span className="mt-3 block text-[11px]">
          <button type="button" className="text-fg-3 underline cursor-pointer" aria-expanded={open} onClick={() => setOpen(o => !o)}>Details</button>
          {open && <span className="mt-1 block break-words font-mono text-fg-3">{String(error.detail)}</span>}
        </span>
      )}
    </EmptyState>
  )
}

/** What the chip advice tells the manager: play a chip this week, save one for a week, or hold. */
function chipAdviceKind(plan) {
  const first = plan?.gameweek_plans?.[0]
  if (first?.chip_used) return 'play'
  if (plan?.gameweek_plans?.some(p => p.chip_used)) return 'save'
  return plan?.chip_hold?.length ? 'hold' : 'none'
}

const prefIds = list => list.map(p => p.id).sort((a, b) => a - b).join(',')

export default function PlannerView({
  season, managerId, setManagerId, validManagerId, budget, setBudget,
  managerChips, account, bankedTransfers, setBankedTransfers, fplData,
}) {
  // A plan remembered from the last visit, for this manager, if it is recent.
  const [remembered] = useState(() => {
    const saved = readJson(LAST_PLAN_KEY)
    if (!saved?.compare || saved.managerId !== validManagerId) return null
    if (Date.now() - saved.savedAt > LAST_PLAN_MAX_AGE_MS) return null
    return saved
  })

  const [horizon, setHorizon] = useState(remembered?.settings?.horizon || 8)
  const [strategy, setStrategy] = useState(remembered?.settings?.strategy || 'standard')
  const [useRobust, setUseRobust] = useState(Boolean(remembered?.settings?.robust))
  const [ftOverride, setFtOverride] = useState(false)

  const [chipRecs, setChipRecs] = useState(null)
  const [selectedChip, setSelectedChip] = useState(null)
  const [prefs, setPrefs] = useState(() => {
    const saved = readJson(PREFS_KEY)
    return { locked: saved?.locked || [], excluded: saved?.excluded || [] }
  })

  const [running, setRunning] = useState(null) // { startedAt, horizon, auto }
  const [error, setError] = useState(null)
  const [compare, setCompare] = useState(remembered?.compare || null)
  const [meta, setMeta] = useState(remembered ? { ...remembered, compare: undefined, restored: true } : null)
  const [hitStrategy, setHitStrategy] = useState(remembered?.hitStrategy || 'with_hits')
  const [gwIndex, setGwIndex] = useState(0)
  const [view, setView] = useState('pitch')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const abortRef = useRef(null)
  const loading = Boolean(running)

  useEffect(() => {
    if (remembered) track('plan_restored', { age_min: Math.round((Date.now() - remembered.savedAt) / 60000) })
  }, [remembered])

  useEffect(() => { writeJson(PREFS_KEY, prefs) }, [prefs])

  const maxBanked = season?.max_banked_transfers || 5
  const ftSynced = Boolean(managerChips?.synced && Number.isInteger(managerChips.banked_transfers))
  const startGw = season?.active_gameweek || chipRecs?.current_gw || compare?.current_gw || 1

  useEffect(() => {
    api.post('/api/chip-recommendations', {
      manager_id: validManagerId ? parseInt(validManagerId) : null,
      current_squad: [],
      chips_used: [],
      horizon,
    })
      .then(res => setChipRecs(res.data))
      .catch(err => console.error('Error fetching chip recommendations', err))
  }, [validManagerId, horizon])

  // Drop a selected chip the manager has already played.
  useEffect(() => {
    if (selectedChip && managerChips?.available && !managerChips.available.map(chipKey).includes(selectedChip.chip)) {
      setSelectedChip(null)
    }
  }, [managerChips, selectedChip])

  const generate = async ({ auto = false, horizonOverride } = {}) => {
    const gws = horizonOverride || horizon
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    const started = performance.now()
    setRunning({ startedAt: started, horizon: gws, auto })
    setError(null)
    // On a phone the settings fold away while planning, so the progress is in view.
    setSettingsOpen(false)
    if (!auto && window.matchMedia?.('(max-width: 1023px)').matches) window.scrollTo({ top: 0, behavior: 'smooth' })
    setGwIndex(0)
    const settings = {
      horizon: gws, strategy, robust: useRobust, chip: selectedChip, ft: parseInt(bankedTransfers),
      locked: prefs.locked, excluded: prefs.excluded,
    }
    const asked = {
      view: 'planner', horizon: gws, strategy, robust: useRobust, forced_chip: selectedChip?.chip || null,
      linked: Boolean(validManagerId), auto, kept: prefs.locked.length, banned: prefs.excluded.length,
    }
    track('plan_requested', asked)
    try {
      const body = {
        budget: parseFloat(budget),
        gameweeks: parseInt(gws),
        manager_id: validManagerId ? parseInt(validManagerId) : null,
        banked_transfers: parseInt(bankedTransfers),
        robust: useRobust,
        uncertainty_budget: useRobust ? 0.3 : 0,
        strategy,
        chip_to_use: selectedChip?.gw ? [selectedChip.chip, selectedChip.gw] : null,
        excluded_players: prefs.excluded.map(p => p.id),
        locked_players: prefs.locked.map(p => p.id),
      }
      const res = await withRetry(() => api.post('/api/optimize/compare', body, { signal: controller.signal, timeout: PLAN_TIMEOUT_MS }))
      const { with_hits: withHits, no_hits: noHits, note } = res.data.comparison
      if (withHits?.feasible === false && noHits?.feasible === false) {
        track('error_shown', { where: 'planner', kind: 'unsolved' })
        const kept = prefs.locked.length ? ' Kept players can make a plan impossible (budget, three per club, or positions); try keeping fewer.' : ''
        setError({ kind: 'unsolved', message: `${note || 'No plan could be solved.'}${kept}` })
        setCompare(null)
        return
      }
      const best = recommendedStrategy(res.data)
      const shown = planFor(res.data, best)
      track('plan_shown', {
        ...asked,
        ms: Math.round(performance.now() - started),
        best,
        proven_optimal: shown?.proven_optimal ?? null,
        partial: Boolean(res.data.partial_result),
        transfers: shown?.gameweek_plans?.[0]?.transfers?.in?.length ?? 0,
        hits: res.data.comparison?.[best]?.total_hits ?? 0,
        chip_advice: chipAdviceKind(shown),
      })
      const savedAt = Date.now()
      setCompare(res.data)
      setHitStrategy(best)
      setMeta({ managerId: validManagerId, savedAt, settings, restored: false })
      writeJson(LAST_PLAN_KEY, { managerId: validManagerId, savedAt, settings, hitStrategy: best, compare: res.data })
    } catch (err) {
      const friendly = friendlyError(err)
      if (friendly.kind === 'cancelled') {
        track('plan_cancelled', { horizon: gws, ms: Math.round(performance.now() - started), auto })
        return
      }
      console.error(err)
      track('error_shown', { where: 'planner', kind: friendly.kind, message: String(friendly.detail || '').slice(0, 120), ms: Math.round(performance.now() - started) })
      setError(friendly)
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
        setRunning(null)
      }
    }
  }

  const cancel = () => abortRef.current?.abort()
  useEffect(() => () => abortRef.current?.abort(), [])

  // Plan once on arrival when a confirmed manager has no recent plan to show.
  const generateRef = useRef(generate)
  generateRef.current = generate
  const autoKey = useRef('')

  // A plan shown for another manager is not this one's; switching back plans again.
  const planForOther = meta && meta.managerId !== validManagerId
  useEffect(() => {
    if (planForOther) {
      setCompare(null)
      setMeta(null)
      autoKey.current = ''
    }
  }, [planForOther])
  useEffect(() => {
    if (!validManagerId || !managerChips || compare) return
    const key = `${validManagerId}:${managerChips.bank}:${managerChips.banked_transfers}`
    if (autoKey.current === key) return
    autoKey.current = key
    generateRef.current({ auto: true })
  }, [validManagerId, managerChips, compare])

  const setPref = (player, kind, source) => {
    const other = kind === 'locked' ? 'excluded' : 'locked'
    setPrefs(prev => {
      const has = prev[kind].some(p => p.id === player.id)
      const entry = { id: player.id, name: player.web_name || player.name }
      return {
        [kind]: has ? prev[kind].filter(p => p.id !== player.id) : [...prev[kind], entry],
        [other]: prev[other].filter(p => p.id !== player.id),
      }
    })
    track('player_pref_set', { kind: kind === 'locked' ? 'keep' : 'ban', source })
  }
  const removePref = (id, kind) => setPrefs(prev => ({ ...prev, [kind]: prev[kind].filter(p => p.id !== id) }))
  const keptIds = useMemo(() => new Set(prefs.locked.map(p => p.id)), [prefs.locked])
  const prefsStale = Boolean(compare && meta?.settings && (
    prefIds(prefs.locked) !== prefIds(meta.settings.locked || []) || prefIds(prefs.excluded) !== prefIds(meta.settings.excluded || [])
  ))

  const plan = planFor(compare, hitStrategy) || planFor(compare, 'with_hits')
  const deadlinePlan = plan?.gameweek_plans?.[0]
  const onDeadline = gwIndex === 0
  const gwPlan = plan?.gameweek_plans?.[gwIndex]
  const gwSummary = gwPlan && plan.transfer_summary?.find(t => t.gameweek === gwPlan.gameweek)
  const squad = gwPlan ? [...gwPlan.starting_xi, ...gwPlan.bench] : null
  const strategyMeta = STRATEGIES.find(s => s.value === strategy)
  const ftStart = meta?.settings?.ft ?? bankedTransfers
  const ledger = useMemo(() => freeTransferLedger(plan?.gameweek_plans, ftStart, maxBanked), [plan, ftStart, maxBanked])
  const fixtures = useMemo(() => fixturesForGameweek(fplData, gwPlan?.gameweek), [fplData, gwPlan?.gameweek])
  const news = useMemo(
    () => (meta?.restored ? newsChanges(deadlinePlan, fplData?.static?.elements) : []),
    [meta?.restored, deadlinePlan, fplData],
  )
  const stalePlanGw = deadlinePlan && season?.active_gameweek && deadlinePlan.gameweek !== season.active_gameweek
  const managerName = account?.entry?.name
  const deadlineIn = season?.deadline_time && season.active_gameweek === deadlinePlan?.gameweek ? countdown(season.deadline_time) : null

  const playerActions = player => {
    const kept = keptIds.has(player.id)
    const banned = prefs.excluded.some(p => p.id === player.id)
    return (
      <div className="grid grid-cols-2 gap-2">
        <Button variant={kept ? 'primary' : 'secondary'} size="sm" aria-pressed={kept} onClick={() => setPref(player, 'locked', 'sheet')}>
          <IconLock size={14} /> {kept ? 'Kept' : 'Keep'}
        </Button>
        <Button variant="secondary" size="sm" aria-pressed={banned} className={banned ? 'ring-1 ring-bad/50 text-[#ffb3ba]' : ''} onClick={() => setPref(player, 'excluded', 'sheet')}>
          <IconBan size={14} /> {banned ? 'Ruled out' : 'Never pick'}
        </Button>
        <p className="col-span-2 text-[11px] text-fg-3">Keep holds him in the squad every week of the plan. Never pick sells him and won't buy him. Generate again to apply.</p>
      </div>
    )
  }
  const transferAction = (p, side) => {
    const kind = side === 'out' ? 'locked' : 'excluded'
    const on = prefs[kind].some(x => x.id === p.id)
    return (
      <button
        type="button"
        onClick={() => setPref({ id: p.id, web_name: p.name }, kind, 'transfer')}
        aria-pressed={on}
        className={cx(
          'inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium ring-1 ring-inset cursor-pointer focus-visible:outline-2 focus-visible:outline-fpl-green',
          on ? 'bg-ink-600 text-fg ring-ink-600' : 'text-fg-3 ring-ink-700 hover:text-fg-2',
        )}
        title={side === 'out' ? `Keep ${p.name} instead of selling him` : `Never buy ${p.name}`}
      >
        {side === 'out' ? <><IconLock size={11} /> {on ? 'Kept' : 'Keep'}</> : <><IconBan size={11} /> {on ? 'Ruled out' : 'Never'}</>}
      </button>
    )
  }

  const generateLabel = account?.status === 'not_found'
    ? 'Check your manager ID'
    : (validManagerId && account?.status === 'loading') ? 'Loading your squad…' : compare ? 'Generate again' : 'Generate plan'

  const settings = (
    <>
      <ManagerFields
        {...{ managerId, setManagerId, validManagerId, budget, setBudget, account }}
        bankLocked={Boolean(managerChips?.synced && typeof managerChips.bank === 'number')}
      />

      <Field label="Weeks to plan ahead" htmlFor="horizon" aside={<span className="text-xs font-semibold text-fg tabular">{horizon} GWs</span>}>
        <input id="horizon" type="range" min="3" max={MAX_HORIZON} value={horizon} onChange={e => setHorizon(Number(e.target.value))} className="w-full" />
        <div className="mt-1 flex justify-between text-[10px] text-fg-3 tabular">
          <span>3</span><span>{Math.round((3 + MAX_HORIZON) / 2)}</span><span>{MAX_HORIZON}</span>
        </div>
        <p className="mt-1.5 text-[11px] text-fg-3">
          {horizon > 8 ? 'Longer plans take longer and may stop at the time limit.' : 'Planning ahead stops the plan selling a player before his good fixtures.'}
        </p>
      </Field>

      <Field
        label="Free transfers"
        hint={ftSynced && !ftOverride
          ? `From your FPL history${managerChips.pending_transfers > 0 ? `, after the ${managerChips.pending_transfers} already made` : ''}.`
          : 'How many free transfers you can use this week.'}
        aside={ftSynced && (
          <button
            type="button"
            onClick={() => {
              if (ftOverride) setBankedTransfers(managerChips.banked_transfers)
              setFtOverride(o => !o)
            }}
            className="text-[11px] text-fpl-green hover:underline cursor-pointer"
          >
            {ftOverride ? `Use FPL's (${managerChips.banked_transfers})` : 'Change'}
          </button>
        )}
      >
        {ftSynced && !ftOverride ? (
          <p className="text-sm font-semibold tabular">{bankedTransfers}</p>
        ) : (
          <Stepper value={bankedTransfers} onChange={setBankedTransfers} min={0} max={maxBanked} />
        )}
      </Field>

      <ChipPicker
        startGw={startGw}
        horizon={horizon}
        available={managerChips?.synced ? managerChips.available : undefined}
        recs={chipRecs}
        selected={selectedChip}
        onChange={setSelectedChip}
      />

      <Field label="Kept and ruled-out players">
        <PlayerPrefs prefs={prefs} onRemove={removePref} stale={prefsStale} />
      </Field>

      <details className="group rounded-xl bg-ink-900/60 ring-1 ring-inset ring-ink-700/70" open={strategy !== 'standard' || useRobust}>
        <summary className="flex cursor-pointer list-none items-center justify-between px-3 py-2.5 text-xs font-medium text-fg-2 select-none">
          Advanced
          <span className="text-fg-3 transition group-open:rotate-90" aria-hidden="true">›</span>
        </summary>
        <div className="space-y-4 px-3 pb-3">
          <Field label="Aim" hint={strategyMeta?.hint}>
            <Segmented className="w-full" size="sm" value={strategy} onChange={setStrategy} options={STRATEGIES} />
          </Field>
          <Toggle checked={useRobust} onChange={setUseRobust} label="Avoid rotation risks" description="Prefer a nailed player over the same points from rotation or an injury doubt." />
        </div>
      </details>

      <Button
        size="lg"
        className="w-full"
        onClick={() => generate()}
        disabled={loading || account?.status === 'not_found' || (Boolean(validManagerId) && account?.status === 'loading')}
      >
        {loading ? <><Spinner /> Planning…</> : generateLabel}
      </Button>
      {error && error.kind !== 'cancelled' && <ErrorNote>{error.message}</ErrorNote>}
    </>
  )

  const summary = (
    <Card className="flex items-center justify-between gap-3 px-4 py-3">
      <div className="min-w-0 text-sm">
        <div className="truncate font-semibold text-fg">{managerName || (validManagerId ? `Manager ${validManagerId}` : 'No manager linked')}</div>
        <div className="truncate text-[11px] text-fg-3">
          {[`${meta?.settings?.horizon || horizon} GWs`, STRATEGIES.find(s => s.value === (meta?.settings?.strategy || strategy))?.label, (meta?.settings?.robust ?? useRobust) && 'avoiding rotation'].filter(Boolean).join(' · ')}
        </div>
      </div>
      <Button variant="secondary" size="sm" onClick={() => setSettingsOpen(true)} aria-expanded={settingsOpen}>Edit</Button>
    </Card>
  )
  const collapsed = Boolean(compare) && !settingsOpen

  if (loading || !plan) {
    let body
    if (loading) {
      body = (
        <PlanLoading
          horizon={running.horizon}
          startedAt={running.startedAt}
          onCancel={cancel}
          auto={running.auto}
          who={managerName}
        />
      )
    } else if (error && error.kind !== 'cancelled') {
      body = (
        <PlanError
          error={error}
          horizon={horizon}
          onRetry={() => generate()}
          onShorter={() => { track('plan_retry_shorter', { from: horizon }); setHorizon(SHORT_HORIZON); generate({ horizonOverride: SHORT_HORIZON }) }}
        />
      )
    } else if (account?.status === 'not_found') {
      body = (
        <EmptyState icon={<IconRoute size={24} />} title="That ID isn't an FPL team">
          Check the number after /entry/ in the address of your Points page on fantasy.premierleague.com, or plan without an ID for a squad from scratch.
        </EmptyState>
      )
    } else {
      body = (
        <EmptyState
          icon={<IconRoute size={24} />}
          title="Your team for the next deadline"
          action={!validManagerId && <Button variant="secondary" onClick={() => generate()}>Plan a squad from scratch</Button>}
        >
          Enter your manager ID. The plan loads your squad, bank and free transfers, then sets the transfers, captain, XI and bench for the next deadline.
        </EmptyState>
      )
    }
    return (
      <Workspace settings={settings} collapsed={loading && !settingsOpen} summary={summary}>
        <Canvas>{body}</Canvas>
      </Workspace>
    )
  }

  return (
    <Workspace settings={settings} collapsed={collapsed} summary={summary}>
      <div className="space-y-4">
        {meta && (
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl bg-ink-900/60 px-3.5 py-2 text-xs text-fg-3 ring-1 ring-inset ring-ink-700/70">
            <span>
              {meta.restored ? 'Remembered plan' : 'Planned'} {meta.restored ? timeAgo(meta.savedAt) : `at ${clockTime(meta.savedAt)}`}
              {season?.data_as_of && !meta.restored && <> · FPL data from {clockTime(season.data_as_of * 1000)}</>}
              {managerName && <> · {managerName}</>}
            </span>
            {(meta.restored || prefsStale) && (
              <Button variant="ghost" size="sm" onClick={() => generate()}>Re-plan</Button>
            )}
          </div>
        )}
        {(stalePlanGw || news.length > 0) && (
          <div role="status" className="flex flex-wrap items-center justify-between gap-2 rounded-xl bg-warn/10 px-3.5 py-2.5 text-xs text-warn ring-1 ring-inset ring-warn/25">
            <span>
              {stalePlanGw
                ? `This plan was for GW${deadlinePlan.gameweek}, whose deadline has passed.`
                : `News changed since this plan: ${news.slice(0, 3).map(p => p.web_name).join(', ')}${news.length > 3 ? ` and ${news.length - 3} more` : ''}.`}
            </span>
            <Button size="sm" onClick={() => generate()}>Re-plan</Button>
          </div>
        )}

        {deadlinePlan && (
          <DeadlineCard
            plan={plan}
            gwPlan={deadlinePlan}
            ledger={ledger[0]}
            managerName={managerName}
            priceAlerts={plan.price_alerts}
            deadlineIn={deadlineIn}
          />
        )}

        <Card className="p-4 sm:p-5">
          <div className="flex flex-col xl:flex-row xl:items-center gap-4 xl:gap-6">
            <div className="shrink-0">
              <SectionLabel>Projected over {plan.gameweek_plans.length} GWs</SectionLabel>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="font-display text-3xl font-semibold text-fg tabular">{num(plan.total_expected_points, 0)}</span>
                <span className="text-sm text-fg-3">pts after hits</span>
              </div>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {meta?.settings?.strategy && meta.settings.strategy !== 'standard' && <Badge>{STRATEGIES.find(s => s.value === meta.settings.strategy)?.label}</Badge>}
                {meta?.settings?.robust && <Badge tone="violet"><IconShield size={11} /> Avoiding rotation</Badge>}
                {plan.proven_optimal === false && (
                  <Badge tone="warn" title="The solver hit its time limit. A shorter horizon solves fully.">Best found in time limit</Badge>
                )}
                {meta?.settings?.chip && <Badge tone="pink">{chipLabel(meta.settings.chip.chip)} GW{meta.settings.chip.gw}</Badge>}
              </div>
            </div>
            <div className="flex-1">
              <StrategyCompare compare={compare} selected={hitStrategy} onSelect={k => { track('strategy_switched', { to: k }); setHitStrategy(k); setGwIndex(0) }} />
            </div>
          </div>
        </Card>

        <div className="grid grid-cols-1 2xl:grid-cols-[minmax(0,1fr)_380px] gap-4 items-start">
          <PlanTable plan={plan} ledger={ledger} selectedIndex={gwIndex} onSelect={idx => { track('gameweek_viewed', { index: idx, from: 'table' }); setGwIndex(idx) }} />
          <ChipStrip plan={plan} usedThisHalf={managerChips?.used_this_half} splitGw={season?.chip_half_split_gw} forcedChip={meta?.settings?.chip} />
        </div>

        {/* Gameweek timeline */}
        <div className="flex gap-2 overflow-x-auto scrollbar-thin pb-1" role="tablist" aria-label="Gameweek">
          {plan.gameweek_plans.map((p, idx) => {
            const nTransfers = p.transfers?.in?.length || 0
            const range = weekRange(p)
            return (
              <button
                key={p.gameweek}
                type="button"
                role="tab"
                aria-selected={gwIndex === idx}
                onClick={() => { track('gameweek_viewed', { index: idx }); setGwIndex(idx) }}
                className={cx(
                  'min-w-[112px] flex-1 rounded-xl px-3.5 py-2.5 text-left transition cursor-pointer ring-1 ring-inset focus-visible:outline-2 focus-visible:outline-fpl-green',
                  gwIndex === idx ? 'bg-ink-800 ring-fpl-pink/70 shadow-[inset_0_-2px_0_0] shadow-fpl-pink' : 'bg-ink-850/80 ring-ink-700 hover:ring-ink-600',
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] font-semibold text-fg-3">{idx === 0 ? `GW${p.gameweek} · next` : `GW${p.gameweek}`}</span>
                  {p.chip_used && <Badge tone="pink">{chipLabel(p.chip_used).split(' ').map(w => w[0]).join('')}</Badge>}
                </div>
                <div className="mt-0.5 font-display text-lg font-semibold tabular">{num(p.expected_points)}<span className="ml-1 text-xs font-sans font-normal text-fg-3">xP</span></div>
                <div className="text-[11px] text-fg-3 tabular">
                  {range ? `${num(range.low, 0)}–${num(range.high, 0)} · ` : ''}
                  {nTransfers ? `${nTransfers} move${nTransfers > 1 ? 's' : ''}` : 'Roll'}
                  {p.transfers?.hits > 0 && <span className="text-warn"> · −{p.transfers.hits * 4}</span>}
                </div>
              </button>
            )
          })}
        </div>

        {gwPlan && (
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px] gap-4 items-start">
            <Card className="p-4 sm:p-5">
              <div className="mb-4 flex items-center justify-between gap-3">
                <div>
                  <SectionLabel>{onDeadline ? 'Set this team' : 'Draft'}</SectionLabel>
                  <h2 className="font-display text-lg font-semibold">Gameweek {gwPlan.gameweek}</h2>
                  <p className="text-xs text-fg-3">
                    {onDeadline
                      ? 'Tap a player for details, or to keep or rule him out.'
                      : `A draft for GW${gwPlan.gameweek}; it will change as news comes in. The team to set is GW${deadlinePlan?.gameweek}.`}
                  </p>
                  <p className="text-xs text-fg-3 tabular mt-1">
                    {num(gwPlan.expected_points)} xP from the starting XI, captain doubled
                    {weekRange(gwPlan) && <> · likely {num(weekRange(gwPlan).low, 0)}–{num(weekRange(gwPlan).high, 0)}</>}
                  </p>
                </div>
                <ViewToggle value={view} onChange={v => { track('view_toggled', { view: v }); setView(v) }} />
              </div>
              {!onDeadline && (
                <button type="button" onClick={() => setGwIndex(0)} className="mb-4 text-xs font-semibold text-fpl-green hover:underline cursor-pointer">
                  Back to GW{deadlinePlan?.gameweek}
                </button>
              )}
              <SquadView squad={squad} view={view} gameweek={gwPlan.gameweek} fixtures={fixtures} playerActions={playerActions} kept={keptIds} />
              {benchOrder(gwPlan.bench).length > 0 && (
                <p className="mt-3 text-xs text-fg-3">
                  Bench order: {benchOrder(gwPlan.bench).map((p, i) => `${i === 0 ? 'GK' : i} ${p.web_name}`).join(' · ')}
                </p>
              )}
            </Card>

            <div className="space-y-4">
              <TransferPanel transfers={gwPlan.transfers} summary={gwSummary ?? gwPlan.transfers} explainOpen={onDeadline} playerAction={transferAction} />
              <CaptainCard gwPlan={gwPlan} options={onDeadline ? plan.captain_options : null} />
            </div>
          </div>
        )}
      </div>
    </Workspace>
  )
}
