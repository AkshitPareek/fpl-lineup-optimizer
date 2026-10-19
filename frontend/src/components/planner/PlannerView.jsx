import Pitch from '../Pitch'
import {
  chipName,
  chipShort,
  formatMoney,
  formatPoints,
  planStats,
  plannedPrice,
  plansAreIdentical,
  playerLabel,
  POSITIONS,
  waitMessage,
  weekIndexFor,
} from '../../plannerFormat'

function Stat({ label, value, hint }) {
  return (
    <div className="rounded-lg bg-gray-900/50 border border-gray-700 px-3 py-3">
      <p className="text-[11px] uppercase tracking-wider text-gray-400">{label}</p>
      <p className="text-lg font-bold text-white mt-1">{value}</p>
      {hint && <p className="text-xs text-gray-400 mt-1">{hint}</p>}
    </div>
  )
}

function ProblemPanel({ problem }) {
  if (!problem) return null
  return (
    <div className="rounded-xl border border-red-500/40 bg-red-950/30 p-5">
      <h3 className="text-lg font-bold text-white">{problem.title}</h3>
      <p className="text-sm text-red-100 mt-2">{problem.detail}</p>
      {problem.hints?.length > 0 && (
        <ul className="mt-3 space-y-1 text-sm text-gray-200 list-disc pl-5">
          {problem.hints.map((hint) => <li key={hint}>{hint}</li>)}
        </ul>
      )}
    </div>
  )
}

function PlannerProgress({ gameweeks, elapsed, hasPrevious }) {
  return (
    <div className="rounded-xl border border-fpl-cyan/40 bg-gray-900 p-4" role="status" aria-live="polite">
      <div className="flex items-start gap-3">
        <div className="mt-1 h-5 w-5 shrink-0 rounded-full border-2 border-fpl-cyan border-t-transparent animate-spin" />
        <div>
          <p className="font-bold text-white">Planning {gameweeks} gameweeks</p>
          <p className="text-sm text-gray-200 mt-1">{waitMessage(gameweeks)}</p>
          <p className="text-xs text-gray-400 mt-2">
            {hasPrevious
              ? 'The plan below is the previous result. It stays until this solve finishes.'
              : 'The page fills in when the solver finishes. A slow solve is still working.'}
          </p>
          <p className="text-xs text-gray-500 mt-1">{elapsed}s elapsed</p>
        </div>
      </div>
      <div className="mt-3 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div className="h-full w-1/3 bg-gradient-to-r from-fpl-green to-fpl-cyan planner-indeterminate" />
      </div>
    </div>
  )
}

function EmptyPlan() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center text-center px-4 py-16">
      <p className="text-5xl mb-4" aria-hidden="true">⚽️</p>
      <h3 className="text-xl font-bold text-white">Plan the weeks ahead</h3>
      <p className="text-sm text-gray-300 mt-2 max-w-md">
        Generate a plan to see expected points, chips played and kept, hits, and each week’s transfers at the prices the plan uses.
      </p>
      <ul className="mt-4 text-sm text-gray-400 space-y-2 text-left max-w-md">
        <li>Add a manager ID to start from your squad and bank. Leave it blank to build a new squad from the budget.</li>
        <li>{waitMessage(8)} {waitMessage(15)}</li>
      </ul>
    </div>
  )
}

function PlanSummary({ plan, onSelectWeek }) {
  const stats = planStats(plan)
  const played = plan.chip_plan || []
  const kept = plan.chips_held || []
  const gap = typeof plan.optimality_gap_points === 'number' ? plan.optimality_gap_points : null
  const showGap = plan.proven_optimal === false || (gap != null && gap > 0)
  const range = stats.firstGw && stats.lastGw ? `GW${stats.firstGw}–${stats.lastGw}` : `${stats.weeks} gameweeks`

  return (
    <section className="rounded-xl border border-gray-700 bg-gray-900/40 p-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-[11px] uppercase tracking-wider text-gray-400">Plan total · {range}</p>
          <p className="text-4xl font-bold text-fpl-green leading-none mt-1">
            {formatPoints(stats.total)}
            <span className="text-base font-semibold text-gray-400 ml-2">pts</span>
          </p>
          <p className="text-sm text-gray-300 mt-2">
            {formatPoints(stats.average)} per week, after hits
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 min-w-[220px] flex-1 sm:max-w-sm">
          <Stat
            label="Hits"
            value={stats.hits.count === 0 ? 'None' : `−${stats.hits.points}`}
            hint={stats.hits.count === 0 ? 'Free transfers only' : `${stats.hits.count} hit${stats.hits.count === 1 ? '' : 's'} across the plan`}
          />
          <Stat
            label="After the plan"
            value={Number.isInteger(plan.banked_transfers_after) ? `${plan.banked_transfers_after} FT` : '—'}
            hint="Free transfers banked"
          />
        </div>
      </div>

      {showGap && (
        <p className="mt-3 text-sm text-yellow-300">
          {gap != null
            ? `Best plan found in the time limit, within ${formatPoints(gap)} pts of the best possible.`
            : 'Best plan found in the time limit. A shorter horizon solves fully.'}
        </p>
      )}

      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="rounded-lg border border-fpl-cyan/30 bg-fpl-cyan/5 p-3">
          <p className="text-[11px] uppercase tracking-wider text-fpl-cyan font-bold">Chips played</p>
          {played.length === 0 ? (
            <p className="text-sm text-gray-300 mt-2">None in this plan.</p>
          ) : (
            <ul className="mt-2 space-y-2">
              {played.map((chip) => {
                const index = weekIndexFor(plan, chip.gameweek)
                const gain = formatPoints(chip.expected_gain)
                return (
                  <li key={`${chip.chip}-${chip.gameweek}`}>
                    <button
                      type="button"
                      disabled={index == null}
                      onClick={() => index != null && onSelectWeek(index)}
                      className="text-left w-full rounded-md px-2 py-1 hover:bg-white/5 disabled:hover:bg-transparent"
                    >
                      <span className="text-sm font-semibold text-white">
                        {chipName(chip.chip)} · GW{chip.gameweek}
                        {gain ? ` · +${gain}` : ''}
                      </span>
                      {chip.reason && <span className="block text-xs text-gray-400">{chip.reason}</span>}
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
        <div className="rounded-lg border border-gray-700 bg-gray-900/40 p-3">
          <p className="text-[11px] uppercase tracking-wider text-gray-400 font-bold">Chips kept</p>
          {kept.length === 0 ? (
            <p className="text-sm text-gray-300 mt-2">None kept past this plan.</p>
          ) : (
            <ul className="mt-2 space-y-2">
              {kept.map((chip) => (
                <li key={chip.chip} className="text-sm text-gray-200">
                  <span className="font-semibold text-white">{chipName(chip.chip)}</span>
                  {chip.best_gameweek ? ` · worth more around GW${chip.best_gameweek}` : ''}
                  {formatPoints(chip.expected_gain) ? ` (~${formatPoints(chip.expected_gain)} pts)` : ''}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  )
}

function HitComparison({ compareData, selectedKey, onSelect }) {
  const comparison = compareData?.comparison
  if (!comparison) return null
  const note = comparison.note || ''
  const identical = plansAreIdentical(compareData)

  if (identical) {
    const points = formatPoints(comparison.with_hits?.net_xp)
    return (
      <section className="rounded-xl border border-gray-700 bg-gray-900/50 p-4">
        <p className="text-sm font-bold text-white">One plan</p>
        <p className="text-sm text-gray-200 mt-1">
          {note || 'No hits required. Both solves produce the same plan.'}
          {points ? ` ${points} pts.` : ''}
        </p>
        <p className="text-xs text-gray-400 mt-2">Shown once, because the hit plan and the free-transfer plan match.</p>
      </section>
    )
  }

  const options = [
    { key: 'with_hits', label: 'With hits', meta: comparison.with_hits },
    { key: 'no_hits', label: 'Free transfers only', meta: comparison.no_hits },
  ]
  const bothFeasible = options.every((option) => option.meta?.feasible === true)

  if (!bothFeasible) {
    return (
      <section className="rounded-xl border border-gray-700 bg-gray-900/50 p-4 space-y-2">
        <p className="text-sm font-bold text-white">Hit comparison</p>
        {options.map((option) => (
          <p key={option.key} className="text-sm text-gray-200">
            <span className="text-gray-400">{option.label}: </span>
            {option.meta?.feasible
              ? `${formatPoints(option.meta.net_xp)} pts${option.meta.hit_cost ? ` after −${option.meta.hit_cost} for hits` : ''}`
              : 'Not solved, and not scored as zero.'}
          </p>
        ))}
        <p className="text-sm text-gray-300">{note || 'One of the two plans could not be solved.'}</p>
      </section>
    )
  }

  const difference = comparison.difference
  let differenceText = null
  if (typeof difference === 'number' && difference > 0) {
    differenceText = `Taking hits scores ${formatPoints(difference)} pts more over the horizon.`
  } else if (typeof difference === 'number' && difference < 0) {
    differenceText = `Skipping hits scores ${formatPoints(Math.abs(difference))} pts more over the horizon.`
  } else if (difference === 0) {
    differenceText = 'Both plans score the same over the horizon. The moves can still differ.'
  }

  return (
    <section className="rounded-xl border border-gray-700 bg-gray-900/50 p-4">
      <div className="flex items-center justify-between gap-3 mb-3">
        <p className="text-sm font-bold text-white">Hit comparison</p>
        {compareData.strategy && <p className="text-xs text-gray-500 capitalize">{compareData.strategy}</p>}
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {options.map((option) => {
          const active = selectedKey === option.key
          const meta = option.meta
          return (
            <button
              key={option.key}
              type="button"
              onClick={() => onSelect(option.key)}
              className={`p-3 rounded-lg text-left transition border ${active ? 'border-fpl-green bg-fpl-green/10' : 'border-gray-700 bg-gray-800 hover:border-gray-500'}`}
            >
              <span className="flex items-center justify-between gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-gray-300">{option.label}</span>
                {meta.recommended && typeof difference === 'number' && difference !== 0 && (
                  <span className="text-[10px] bg-green-900/60 text-green-300 px-1.5 py-0.5 rounded">Higher score</span>
                )}
              </span>
              <span className="block text-xl font-bold text-white mt-1">{formatPoints(meta.net_xp)} pts</span>
              <span className="block text-xs text-gray-400 mt-1">
                {option.key === 'with_hits'
                  ? `${formatPoints(meta.total_xp)} xP, minus ${meta.hit_cost || 0} for hits`
                  : 'No point hits'}
              </span>
            </button>
          )
        })}
      </div>
      {differenceText && <p className="text-sm text-gray-300 mt-3">{differenceText}</p>}
      {note && <p className="text-xs text-gray-400 mt-2">{note}</p>}
    </section>
  )
}

function WeekRail({ plan, selectedIndex, onSelect }) {
  return (
    <div className="-mx-1 px-1 py-2 border-b border-gray-700">
      <div className="flex gap-2 overflow-x-auto pb-1" role="tablist" aria-label="Gameweeks">
        {plan.gameweek_plans.map((week, index) => {
          const moves = Math.max(week.transfers?.in?.length || 0, week.transfers?.out?.length || 0)
          const active = index === selectedIndex
          return (
            <button
              key={week.gameweek}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => onSelect(index)}
              className={`shrink-0 min-w-[88px] px-3 py-2 rounded-lg text-left ${active ? 'bg-fpl-pink text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
            >
              <span className="block text-[11px] uppercase tracking-wide opacity-80">GW {week.gameweek}</span>
              <span className="block text-sm font-bold">{formatPoints(week.expected_points)}</span>
              <span className="block text-[10px] mt-1 truncate">
                {week.chip_used ? chipShort(week.chip_used) : moves ? `${moves} transfer${moves === 1 ? '' : 's'}` : 'Hold'}
                {week.transfers?.hits > 0 ? ` · −${week.transfers.hits * 4}` : ''}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function TransferRow({ kind, player }) {
  const price = plannedPrice(player)
  const points = typeof player === 'object' ? formatPoints(player.expected_points) : null
  const position = typeof player === 'object' ? POSITIONS[player.element_type] : null
  const sell = kind === 'out'
  return (
    <li className={`flex items-center justify-between gap-3 rounded-lg px-3 py-2 ${sell ? 'bg-red-950/30' : 'bg-green-950/30'}`}>
      <div className="min-w-0">
        <p className="text-sm font-semibold text-white truncate">
          <span className={`mr-2 text-[10px] font-bold uppercase ${sell ? 'text-red-300' : 'text-green-300'}`}>{sell ? 'Out' : 'In'}</span>
          {playerLabel(player)}
          {position ? <span className="text-gray-400 font-normal"> · {position}</span> : null}
        </p>
        <p className="text-xs text-gray-400 mt-0.5">
          {price != null ? (sell ? `Sells for ${formatMoney(price)}` : `Plan pays ${formatMoney(price)}`) : 'Price not on this move'}
        </p>
      </div>
      {points && <p className="text-sm font-bold text-fpl-green shrink-0">{points} xP</p>}
    </li>
  )
}

function TransferExplanation({ text }) {
  if (!text) return null
  const lines = text.split('\n')
  return (
    <div className="space-y-2 text-sm">
      {lines.map((line, index) => {
        if (!line.trim() || line.startsWith('##')) return null
        if (line.startsWith('###')) {
          return <p key={index} className="font-bold text-gray-400 border-b border-gray-700 pb-1">{line.replace('### ', '')}</p>
        }
        if (line.includes('**SELL:**')) {
          return <p key={index} className="text-red-200">{line.replace(/\*\*/g, '')}</p>
        }
        if (line.includes('**BUY:**')) {
          return <p key={index} className="text-green-200">{line.replace(/\*\*/g, '')}</p>
        }
        return <p key={index} className="text-gray-400">{line.replace(/\*\*/g, '')}</p>
      })}
    </div>
  )
}

function WeekDetail({ plan, week, viewMode, onViewMode }) {
  const starters = week.starting_xi || []
  const bench = week.bench || []
  const squad = [...starters, ...bench].map((player) => ({
    ...player,
    is_vice_captain: player.id === week.vice_captain_id,
  }))
  const captain = starters.find((player) => player.id === week.captain_id) || starters.find((player) => player.is_captain)
  const vice = starters.find((player) => player.id === week.vice_captain_id && player.id !== captain?.id)
  const recommendation = week.captain_recommendation
  const chip = (plan.chip_plan || []).find((entry) => entry.gameweek === week.gameweek && entry.chip === week.chip_used)
    || (week.chip_used ? { chip: week.chip_used, gameweek: week.gameweek } : null)
  const incoming = week.transfers?.in || []
  const outgoing = week.transfers?.out || []
  const hasMoves = incoming.length > 0 || outgoing.length > 0
  const hits = week.transfers?.hits || 0
  const freeUsed = week.transfers?.ft_used || 0

  return (
    <div className="@container">
    <div className="grid grid-cols-1 @min-[840px]:grid-cols-[minmax(260px,340px)_minmax(0,1fr)] gap-4 items-start">
      <div className="space-y-3">
        <div className="rounded-xl border border-gray-700 bg-gray-900/40 p-4">
          <div className="flex items-baseline justify-between gap-3">
            <h3 className="text-xl font-bold text-white">GW {week.gameweek}</h3>
            <p className="text-fpl-green font-bold">{formatPoints(week.expected_points)} xP</p>
          </div>
          <dl className="mt-3 grid grid-cols-2 gap-2 text-sm">
            <div>
              <dt className="text-[11px] uppercase tracking-wider text-gray-500">Bank</dt>
              <dd className="text-white">{typeof week.bank === 'number' ? `${formatMoney(week.bank)} after transfers` : '—'}</dd>
            </div>
            <div>
              <dt className="text-[11px] uppercase tracking-wider text-gray-500">Free transfers</dt>
              <dd className="text-white">{freeUsed} used{hits > 0 ? ` · ${hits} hit${hits === 1 ? '' : 's'} (−${hits * 4})` : ' · no hit'}</dd>
            </div>
          </dl>
        </div>

        <div className="rounded-xl border border-yellow-500/30 bg-yellow-950/20 p-4">
          <p className="text-[11px] uppercase tracking-wider text-yellow-300 font-bold">Captain</p>
          {captain ? (
            <>
              <p className="text-white font-bold mt-1">
                {captain.web_name || captain.name}
                <span className="text-fpl-green font-semibold ml-2">{formatPoints(captain.expected_points)} xP</span>
              </p>
              <p className="text-xs text-gray-400">{captain.team_name} · {POSITIONS[captain.element_type] || captain.position}</p>
            </>
          ) : (
            <p className="text-sm text-gray-300 mt-1">No captain on this week.</p>
          )}
          {vice && (
            <p className="text-sm text-gray-200 mt-2">
              Vice {vice.web_name || vice.name}
              <span className="text-gray-400"> · {formatPoints(vice.expected_points)} xP</span>
            </p>
          )}
          {recommendation?.player_name && recommendation.player_id === captain?.id && (
            <p className="text-xs text-gray-400 mt-2">
              {recommendation.fixtures}
              {recommendation.has_dgw ? ' · Double gameweek' : ''}
              {recommendation.confidence ? ` · ${recommendation.confidence} confidence` : ''}
            </p>
          )}
          {recommendation?.player_name && recommendation.player_id !== captain?.id && (
            <p className="text-xs text-gray-400 mt-2">
              Separate suggestion: {recommendation.player_name}
              {formatPoints(recommendation.expected_points) ? ` (${formatPoints(recommendation.expected_points)} xP)` : ''}
              {recommendation.fixtures ? ` · ${recommendation.fixtures}` : ''}
            </p>
          )}
        </div>

        {chip && (
          <div className="rounded-xl border border-fpl-cyan/30 bg-fpl-cyan/5 p-4">
            <p className="text-[11px] uppercase tracking-wider text-fpl-cyan font-bold">Chip this week</p>
            <p className="text-white font-bold mt-1">{chipName(chip.chip)}</p>
            {chip.reason && <p className="text-sm text-gray-300 mt-1">{chip.reason}</p>}
            {formatPoints(chip.expected_gain) && (
              <p className="text-sm text-fpl-green mt-1">About +{formatPoints(chip.expected_gain)} pts from the chip</p>
            )}
          </div>
        )}

        <div className="rounded-xl border border-gray-700 bg-gray-900/40 p-4">
          <p className="text-[11px] uppercase tracking-wider text-gray-400 font-bold">Transfers</p>
          {hasMoves ? (
            <ul className="mt-2 space-y-2 max-h-80 overflow-y-auto">
              {outgoing.map((player, index) => <TransferRow key={`out-${player.id || index}`} kind="out" player={player} />)}
              {incoming.map((player, index) => <TransferRow key={`in-${player.id || index}`} kind="in" player={player} />)}
            </ul>
          ) : (
            <p className="text-sm text-gray-300 mt-2">No transfers this week.</p>
          )}
          {week.transfers?.explanation && hasMoves && (
            <details className="mt-3">
              <summary className="text-xs font-bold uppercase tracking-wider text-fpl-cyan cursor-pointer">Why these moves</summary>
              <div className="mt-2">
                <TransferExplanation text={week.transfers.explanation} />
              </div>
            </details>
          )}
          {week.transfers?.alternatives?.length > 0 && (
            <details className="mt-3">
              <summary className="text-xs font-bold uppercase tracking-wider text-gray-400 cursor-pointer">Other players, at current prices</summary>
              <div className="mt-2 space-y-2">
                {week.transfers.alternatives.map((group) => (
                  <div key={group.position_id || group.position}>
                    <p className="text-[11px] text-gray-500">{group.position}</p>
                    <ul className="text-xs text-gray-300">
                      {(group.options || []).slice(0, 3).map((option) => (
                        <li key={option.id} className="flex justify-between gap-2">
                          <span>{option.name}</span>
                          <span className="text-gray-500">{formatPoints(option.expected_points)} xP · {formatMoney(option.price)} now</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>

        <div className="rounded-xl border border-gray-700 bg-gray-900/40 p-4">
          <p className="text-[11px] uppercase tracking-wider text-gray-400 font-bold">Bench</p>
          {bench.length === 0 ? (
            <p className="text-sm text-gray-300 mt-2">No bench this week.</p>
          ) : (
            <ol className="mt-2 space-y-1">
              {bench.map((player, index) => (
                <li key={player.id || index} className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-white">
                    <span className="text-gray-500 mr-2">{index + 1}</span>
                    {player.web_name}
                    <span className="text-gray-400"> · {POSITIONS[player.element_type] || player.position}</span>
                  </span>
                  <span className="text-fpl-green font-semibold">{formatPoints(player.expected_points)}</span>
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>

      <div>
        <div className="flex justify-end gap-2 mb-3">
          <button type="button" onClick={() => onViewMode('pitch')} className={`px-3 py-1 rounded text-xs uppercase font-bold ${viewMode === 'pitch' ? 'bg-fpl-green text-gray-900' : 'bg-gray-700 text-gray-300'}`}>Pitch</button>
          <button type="button" onClick={() => onViewMode('list')} className={`px-3 py-1 rounded text-xs uppercase font-bold ${viewMode === 'list' ? 'bg-fpl-green text-gray-900' : 'bg-gray-700 text-gray-300'}`}>List</button>
        </div>
        {viewMode === 'pitch' ? <Pitch lineup={squad} /> : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {squad.map((player) => (
              <div key={player.id} className={`p-3 rounded flex justify-between items-center ${player.is_starter ? 'bg-gray-700' : 'bg-gray-700/40 border border-dashed border-gray-600'}`}>
                <div>
                  <p className="font-bold text-sm">
                    {player.web_name}
                    {player.is_captain && <span className="ml-2 bg-yellow-400 text-black text-[9px] font-bold px-1 rounded">C</span>}
                    {player.is_vice_captain && <span className="ml-1 bg-gray-200 text-black text-[9px] font-bold px-1 rounded">V</span>}
                    {!player.is_starter && <span className="ml-1 text-[9px] text-gray-300">Bench</span>}
                  </p>
                  <p className="text-xs text-gray-400">{player.team_name} · {POSITIONS[player.element_type] || player.position}</p>
                </div>
                <p className="text-fpl-green font-bold">{formatPoints(player.expected_points)}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
    </div>
  )
}

function PlannerView({
  plan,
  compareData,
  selectedKey,
  onSelectStrategy,
  selectedIndex,
  onSelectWeek,
  viewMode,
  onViewMode,
  loading,
  elapsed,
  gameweeks,
  problem,
}) {
  const week = plan?.gameweek_plans?.[selectedIndex] || plan?.gameweek_plans?.[0]

  return (
    <div className="bg-gray-800 rounded-xl p-4 sm:p-6 shadow-lg min-h-[600px] flex flex-col gap-4">
      {loading && !plan && (
        <div className="flex-1 flex items-center">
          <div className="w-full"><PlannerProgress gameweeks={gameweeks} elapsed={elapsed} hasPrevious={false} /></div>
        </div>
      )}
      {problem && !loading && !plan && <ProblemPanel problem={problem} />}
      {!plan && !loading && !problem && <EmptyPlan />}
      {plan && (
        <>
          <PlanSummary plan={plan} onSelectWeek={onSelectWeek} />
          <HitComparison compareData={compareData} selectedKey={selectedKey} onSelect={onSelectStrategy} />
          <div className="sticky top-0 z-30 space-y-2 bg-gray-800/95 py-1 backdrop-blur-sm">
            {loading && <PlannerProgress gameweeks={gameweeks} elapsed={elapsed} hasPrevious />}
            {problem && !loading && (
              <>
                <ProblemPanel problem={problem} />
                <p className="text-xs text-gray-400">The squad below is the previous plan, not this failed solve.</p>
              </>
            )}
            <WeekRail plan={plan} selectedIndex={plan.gameweek_plans[selectedIndex] ? selectedIndex : 0} onSelect={onSelectWeek} />
          </div>
          {week && <WeekDetail plan={plan} week={week} viewMode={viewMode} onViewMode={onViewMode} />}
        </>
      )}
    </div>
  )
}

export default PlannerView
