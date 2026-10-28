import { useEffect, useRef, useState } from 'react'
import { api, errorMessage } from '../lib/api'
import { cx } from '../lib/cx'
import { chipLabel, CHIP_LABELS, num, price, signed } from '../lib/format'
import {
  Badge, Button, Card, CardHeader, ConfidenceBadge, EmptyState, ErrorNote, Field,
  LoadingState, SectionLabel, Segmented, Select, Spinner, Stepper, TextInput, Toggle,
} from '../components/ui'
import { IconChip, IconCrown, IconRoute, IconShield } from '../components/icons'
import SquadView, { ViewToggle } from '../components/SquadView'
import TransferPanel from '../components/TransferPanel'
import { Canvas, Workspace } from './Workspace'

// HiGHS solves up to 15 gameweeks; longer horizons may stop at the time limit.
const MAX_HORIZON = 15

const STRATEGIES = [
  { value: 'standard', label: 'Standard', hint: 'Maximise expected points.' },
  { value: 'differential', label: 'Differential', hint: 'Boost low-ownership players (<10%).' },
  { value: 'template', label: 'Template', hint: 'Favour popular, proven picks.' },
]

export function ManagerFields({ managerId, setManagerId, validManagerId, budget, setBudget }) {
  const invalid = managerId.trim() && !/^\d+$/.test(managerId.trim())
  return (
    <>
      <Field
        label="FPL manager ID"
        htmlFor="manager-id"
        hint={invalid ? 'IDs are numbers only — find yours in the URL of your Points page.' : 'Optional. Plans around your current squad.'}
        aside={validManagerId && <Badge tone="green">Linked</Badge>}
      >
        <TextInput id="manager-id" inputMode="numeric" value={managerId} onChange={e => setManagerId(e.target.value)} placeholder="e.g. 123456" />
      </Field>
      <Field label={validManagerId ? 'Money in the bank' : 'Budget'} hint={validManagerId ? 'Your remaining ITB.' : null}>
        <Stepper value={budget} onChange={setBudget} min={0} max={200} step={0.5} suffix="£m" />
      </Field>
    </>
  )
}

function ChipPicker({ season, horizon, available, recs, selected, onChange }) {
  const startGw = season?.active_gameweek || 1
  const endGw = startGw + horizon - 1
  const chips = available || Object.keys(CHIP_LABELS)
  const rec = selected && recs?.recommendations?.[selected.chip]

  const pick = chip => {
    if (!chip) return onChange(null)
    const suggested = recs?.recommendations?.[chip]?.recommended_gw
    const gw = suggested >= startGw && suggested <= endGw ? suggested : startGw
    onChange({ chip, gw })
  }

  return (
    <Field
      label="Chip"
      aside={recs?.best_chip && selected?.chip !== recs.best_chip.chip && (
        <button
          type="button"
          onClick={() => onChange({ chip: recs.best_chip.chip, gw: recs.best_chip.gw ?? recs.best_chip.recommended_gameweek })}
          className="text-[11px] text-fpl-green hover:underline cursor-pointer"
        >
          Apply suggested: {chipLabel(recs.best_chip.chip)}
        </button>
      )}
      hint={rec?.reason || 'Chips change the solve, so they are only used when you pick one.'}
    >
      <div className="flex gap-2">
        <div className="flex-1">
          <Select value={selected?.chip || ''} onChange={e => pick(e.target.value)}>
            <option value="">No chip</option>
            {chips.map(c => <option key={c} value={c}>{chipLabel(c)}</option>)}
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

function StrategyCompare({ compare, selected, onSelect }) {
  const c = compare.comparison
  const best = recommendedStrategy(compare)
  const options = [
    { key: 'with_hits', title: 'With hits', detail: `${num(c.with_hits.total_xp)} xP − ${c.with_hits.hit_cost} in hits`, data: c.with_hits },
    { key: 'no_hits', title: 'No hits', detail: 'Free transfers only', data: c.no_hits },
  ]
  return (
    <div className="grid grid-cols-2 gap-3">
      {options.map(o => {
        const feasible = o.data.feasible !== false && compare[o.key]?.gameweek_plans?.length > 0
        return (
        <button
          key={o.key}
          onClick={() => onSelect(o.key)}
          disabled={!feasible}
          className={cx(
            'rounded-xl p-3.5 text-left transition ring-1 ring-inset disabled:opacity-60',
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

function recommendedStrategy(compare) {
  const noHits = compare.comparison.no_hits
  const feasible = noHits?.feasible !== false && compare.no_hits?.gameweek_plans?.length > 0
  return feasible && noHits?.recommended ? 'no_hits' : 'with_hits'
}

function CaptainCard({ rec, vice }) {
  if (!rec) return null
  return (
    <Card className="p-4">
      <CardHeader icon={<IconCrown />} title="Captain" action={<ConfidenceBadge level={rec.confidence} />} />
      <div className="mt-3 flex items-baseline justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate font-display text-lg font-semibold">{rec.player_name}</div>
          {rec.fixtures && <div className="text-xs text-fg-3">{rec.fixtures}</div>}
        </div>
        <div className="text-right shrink-0">
          <div className="font-display text-xl font-semibold text-fpl-green tabular">{num(rec.expected_points)}</div>
          <div className="text-[11px] text-fg-3">xP {rec.has_dgw && <Badge tone="violet" className="ml-1">DGW</Badge>}</div>
        </div>
      </div>
      {vice && (
        <p className="mt-2 text-xs text-fg-3">Vice-captain <span className="text-fg-2 font-medium">{vice.web_name}</span> <span className="tabular">{num(vice.expected_points)} xP</span></p>
      )}
      {rec.alternatives?.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-ink-700/70 pt-3">
          <span className="text-[11px] text-fg-3 mr-1">Alternatives</span>
          {rec.alternatives.slice(0, 3).map(a => (
            <span key={a.player_name} className="rounded-md bg-ink-900 px-2 py-0.5 text-[11px] text-fg-2 ring-1 ring-inset ring-ink-700 tabular">
              {a.player_name} <span className="text-fg-3">{num(a.expected_points)}</span>
            </span>
          ))}
        </div>
      )}
    </Card>
  )
}

function ChipAdviceCard({ suggestion, hold, remaining, deadlineGw }) {
  const saved = (remaining || []).map(chipLabel).filter(Boolean)
  if (suggestion) {
    const thisWeek = suggestion.recommended_gameweek === deadlineGw
    return (
      <Card className="p-4">
        <CardHeader icon={<IconChip />} title={thisWeek ? 'Play this chip this week' : `Save ${chipLabel(suggestion.chip)} for GW${suggestion.recommended_gameweek}`} />
        <div className="mt-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-semibold">{chipLabel(suggestion.chip)}</span>
            <span className="text-fg-3">{thisWeek ? 'this week' : 'in'}</span>
            <span className="font-semibold tabular">GW{suggestion.recommended_gameweek}</span>
            <ConfidenceBadge level={suggestion.confidence} />
            {suggestion.is_dgw && <Badge tone="violet">DGW</Badge>}
          </div>
          <p className="mt-1 text-xs text-fg-3">{suggestion.reason}</p>
          {suggestion.margin_over_next_week != null && (
            <p className="mt-1 text-xs text-fg-3 tabular">{signed(suggestion.margin_over_next_week)} pts ahead of the next week in this plan</p>
          )}
        </div>
      </Card>
    )
  }
  if (!saved.length && !hold?.length) return null
  return (
    <Card className="p-4">
      <CardHeader icon={<IconChip />} title="Hold your chips" subtitle="No week in this plan is clearly the one to use them." />
      {saved.length > 0 && <p className="mt-3 text-sm text-fg-2">{saved.join(' · ')}</p>}
      {hold?.length > 0 && (
        <ul className="mt-2 space-y-1 text-xs text-fg-3">
          {hold.map(h => <li key={h.chip}>{h.reason}</li>)}
        </ul>
      )}
    </Card>
  )
}

function benchOrder(bench) {
  const gk = (bench || []).filter(p => p.element_type === 1)
  const outfield = (bench || []).filter(p => p.element_type !== 1)
  return [...gk, ...outfield]
}

export default function PlannerView({
  season, managerId, setManagerId, validManagerId, budget, setBudget,
  managerChips, bankedTransfers, setBankedTransfers,
}) {
  const [horizon, setHorizon] = useState(8)
  const [strategy, setStrategy] = useState('standard')
  const [useRobust, setUseRobust] = useState(false)

  const [chipRecs, setChipRecs] = useState(null)
  const [selectedChip, setSelectedChip] = useState(null)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [compare, setCompare] = useState(null)
  const [hitStrategy, setHitStrategy] = useState('with_hits')
  const [gwIndex, setGwIndex] = useState(0)
  const [view, setView] = useState('pitch')

  const maxBanked = season?.max_banked_transfers || 5

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
    if (selectedChip && managerChips?.available && !managerChips.available.includes(selectedChip.chip)) {
      setSelectedChip(null)
    }
  }, [managerChips, selectedChip])

  const generate = async () => {
    setLoading(true)
    setError(null)
    setCompare(null)
    setGwIndex(0)
    try {
      const res = await api.post('/api/optimize/compare', {
        budget: parseFloat(budget),
        gameweeks: parseInt(horizon),
        manager_id: validManagerId ? parseInt(validManagerId) : null,
        banked_transfers: parseInt(bankedTransfers),
        robust: useRobust,
        uncertainty_budget: useRobust ? 0.3 : 0,
        strategy,
        chip_to_use: selectedChip?.gw ? [selectedChip.chip, selectedChip.gw] : null,
      })
      const { with_hits: withHits, no_hits: noHits, note } = res.data.comparison
      if (withHits?.feasible === false && noHits?.feasible === false) {
        setError(note || 'Neither plan could be solved.')
        return
      }
      setCompare(res.data)
      setHitStrategy(recommendedStrategy(res.data))
    } catch (err) {
      console.error(err)
      setError(errorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  const generateRef = useRef(generate)
  generateRef.current = generate
  const autoKey = useRef('')
  useEffect(() => {
    if (!validManagerId || !managerChips || managerChips.error) return
    const key = `${validManagerId}:${managerChips.bank}:${managerChips.banked_transfers}`
    if (autoKey.current === key) return
    autoKey.current = key
    generateRef.current()
  }, [validManagerId, managerChips])

  const plan = compare?.[hitStrategy]
  const deadlinePlan = plan?.gameweek_plans?.[0]
  const onDeadline = gwIndex === 0
  const gwPlan = plan?.gameweek_plans?.[gwIndex]
  const gwSummary = gwPlan && plan.transfer_summary?.find(t => t.gameweek === gwPlan.gameweek)
  const squad = gwPlan ? [...gwPlan.starting_xi, ...gwPlan.bench] : null
  const strategyMeta = STRATEGIES.find(s => s.value === strategy)

  const settings = (
    <>
      <ManagerFields {...{ managerId, setManagerId, validManagerId, budget, setBudget }} />

      <Field label="Planning horizon" aside={<span className="text-xs font-semibold text-fg tabular">{horizon} GWs</span>}>
        <input type="range" min="3" max={MAX_HORIZON} value={horizon} onChange={e => setHorizon(Number(e.target.value))} className="w-full" aria-label="Planning horizon" />
        <div className="mt-1 flex justify-between text-[10px] text-fg-3 tabular">
          <span>3</span><span>{Math.round((3 + MAX_HORIZON) / 2)}</span><span>{MAX_HORIZON}</span>
        </div>
        {horizon > 8 && <p className="mt-1.5 text-[11px] text-fg-3">Long horizons take longer and may stop at the solver's time limit.</p>}
      </Field>

      <Field
        label="Free transfers banked"
        hint={managerChips && Number.isInteger(managerChips.banked_transfers)
          ? `From your FPL history${managerChips.pending_transfers > 0 ? `, after the ${managerChips.pending_transfers} already made` : ''}. Chips left: ${(managerChips.available || []).map(chipLabel).join(', ') || 'none'}.`
          : null}
      >
        <Stepper value={bankedTransfers} onChange={setBankedTransfers} min={0} max={maxBanked} />
      </Field>

      <Field label="Strategy" hint={strategyMeta?.hint}>
        <Segmented className="w-full" size="sm" value={strategy} onChange={setStrategy} options={STRATEGIES} />
      </Field>

      <ChipPicker
        season={season}
        horizon={horizon}
        available={managerChips?.available}
        recs={chipRecs}
        selected={selectedChip}
        onChange={setSelectedChip}
      />

      <div className="rounded-xl bg-ink-900/60 p-3 ring-1 ring-inset ring-ink-700/70">
        <Toggle checked={useRobust} onChange={setUseRobust} label="Robust optimisation" description="Hedge against projection uncertainty." />
      </div>

      <Button size="lg" className="w-full" onClick={generate} disabled={loading || (Boolean(validManagerId) && !managerChips)}>
        {loading ? <><Spinner /> Solving…</> : (validManagerId && !managerChips) ? 'Loading your squad…' : 'Generate plan'}
      </Button>
      <ErrorNote>{error}</ErrorNote>
    </>
  )

  if (loading || !plan) {
    return (
      <Workspace settings={settings}>
        <Canvas>
          {loading ? (
            <LoadingState
              title="Solving multi-period plan"
              detail={`Comparing hit vs. no-hit strategies across ${horizon} gameweeks with the MILP solver. This can take up to a minute.`}
            />
          ) : (
            <EmptyState icon={<IconRoute size={24} />} title="Your team for the next deadline">
              Enter your manager ID. The plan loads your squad, bank and free transfers, then sets the XI, captain, bench and transfers for the next deadline.
            </EmptyState>
          )}
        </Canvas>
      </Workspace>
    )
  }

  return (
    <Workspace settings={settings}>
      <div className="space-y-4">
        <Card className="p-4 sm:p-5">
          <div className="flex flex-col xl:flex-row xl:items-center gap-4 xl:gap-6">
            <div className="shrink-0">
              <SectionLabel>Projected over horizon</SectionLabel>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="font-display text-4xl font-semibold text-fpl-green tabular">{num(plan.total_expected_points, 0)}</span>
                <span className="text-sm text-fg-3">xP · {plan.gameweek_plans.length} GWs</span>
              </div>
              <div className="mt-1 flex flex-wrap gap-1.5">
                <Badge>{compare.strategy}</Badge>
                {useRobust && <Badge tone="violet"><IconShield size={11} /> Robust</Badge>}
                {plan.proven_optimal === false && (
                  <Badge tone="warn" title="The solver hit its time limit. A shorter horizon solves fully.">Best found in time limit</Badge>
                )}
                {selectedChip && <Badge tone="pink">{chipLabel(selectedChip.chip)} GW{selectedChip.gw}</Badge>}
              </div>
            </div>
            <div className="flex-1">
              <StrategyCompare compare={compare} selected={hitStrategy} onSelect={k => { setHitStrategy(k); setGwIndex(0) }} />
            </div>
          </div>
        </Card>

        {/* Gameweek timeline */}
        <div className="flex gap-2 overflow-x-auto scrollbar-thin pb-1" role="tablist" aria-label="Gameweek">
          {plan.gameweek_plans.map((p, idx) => {
            const nTransfers = p.transfers?.in?.length || 0
            return (
              <button
                key={p.gameweek}
                role="tab"
                aria-selected={gwIndex === idx}
                onClick={() => setGwIndex(idx)}
                className={cx(
                  'min-w-[112px] flex-1 rounded-xl px-3.5 py-2.5 text-left transition cursor-pointer ring-1 ring-inset',
                  gwIndex === idx ? 'bg-ink-800 ring-fpl-pink/70 shadow-[inset_0_-2px_0_0] shadow-fpl-pink' : 'bg-ink-850/80 ring-ink-700 hover:ring-ink-600',
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] font-semibold text-fg-3">{idx === 0 ? 'This week' : `GW${p.gameweek}`}</span>
                  {p.chip_used && <Badge tone="pink">{chipLabel(p.chip_used).split(' ').map(w => w[0]).join('')}</Badge>}
                </div>
                <div className="mt-0.5 font-display text-lg font-semibold tabular">{num(p.expected_points)}<span className="ml-1 text-xs font-sans font-normal text-fg-3">xP</span></div>
                <div className="text-[11px] text-fg-3">
                  {nTransfers ? `${nTransfers} transfer${nTransfers > 1 ? 's' : ''}` : 'Roll'}
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
                      ? 'Make the transfers, set this XI, and save the chip unless the plan says to play it. Later weeks are a draft and update when you plan again.'
                      : `A draft for GW${gwPlan.gameweek}. The team to set is GW${deadlinePlan?.gameweek}.`}
                  </p>
                  <p className="text-xs text-fg-3 tabular mt-1">
                    {num(gwPlan.expected_points)} xP from the starting XI, captain doubled
                    {typeof gwPlan.bank === 'number' && <> · {price(gwPlan.bank)} in the bank after transfers</>}
                  </p>
                </div>
                <ViewToggle value={view} onChange={setView} />
              </div>
              {!onDeadline && (
                <button type="button" onClick={() => setGwIndex(0)} className="mb-4 text-xs font-semibold text-fpl-green hover:underline cursor-pointer">
                  Back to GW{deadlinePlan?.gameweek}
                </button>
              )}
              <SquadView squad={squad} view={view} />
              {onDeadline && benchOrder(gwPlan.bench).length > 0 && (
                <p className="mt-3 text-xs text-fg-3">
                  Bench, first substitute first: {benchOrder(gwPlan.bench).map(p => p.web_name).join(' · ')}
                </p>
              )}
            </Card>

            <div className="space-y-4">
              <TransferPanel transfers={gwPlan.transfers} summary={gwSummary ?? gwPlan.transfers} explainOpen={onDeadline} />
              <CaptainCard rec={gwPlan.captain_recommendation} vice={squad?.find(p => p.is_vice_captain)} />
              {onDeadline && (
                <ChipAdviceCard suggestion={plan.chip_suggestion} hold={plan.chip_hold} remaining={plan.chips_remaining} deadlineGw={deadlinePlan?.gameweek} />
              )}
            </div>
          </div>
        )}
      </div>
    </Workspace>
  )
}
