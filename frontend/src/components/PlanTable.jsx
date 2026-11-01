import { Badge, Card, CardHeader } from './ui'
import { IconRoute } from './icons'
import { cx } from '../lib/cx'
import { chipLabel, num } from '../lib/format'
import { captaincy, pairTransfers, weekRange } from '../lib/plan'

// Weeks after these are a sketch: the plan is re-made every week with new news.
const FIRM_WEEKS = 3

/** The whole plan, one row per gameweek: moves, free transfers, hits, chip, captain, points. */
export default function PlanTable({ plan, ledger, selectedIndex, onSelect }) {
  const weeks = plan.gameweek_plans || []
  return (
    <Card className="p-4 sm:p-5">
      <CardHeader
        icon={<IconRoute />}
        title="The whole plan"
        subtitle={`Weeks after GW${weeks[Math.min(FIRM_WEEKS, weeks.length) - 1]?.gameweek} are likely to change as news comes in. Tap a week to see its team.`}
      />
      <div className="mt-4 -mx-4 sm:mx-0 overflow-x-auto scrollbar-thin">
        <table className="w-full sm:min-w-[620px] text-sm tabular">
          <thead className="text-[11px] uppercase tracking-wider text-fg-3">
            <tr className="text-left">
              <th scope="col" className="py-2 pl-4 sm:pl-3 pr-2 font-semibold">GW</th>
              <th scope="col" className="py-2 px-2 font-semibold">Moves</th>
              <th scope="col" className="hidden sm:table-cell py-2 px-2 font-semibold" title="Free transfers used of those available, and how many carry to the next week">Free transfers</th>
              <th scope="col" className="hidden sm:table-cell py-2 px-2 font-semibold">Hits</th>
              <th scope="col" className="hidden sm:table-cell py-2 px-2 font-semibold">Chip</th>
              <th scope="col" className="hidden sm:table-cell py-2 px-2 font-semibold">Captain</th>
              <th scope="col" className="py-2 pl-2 pr-4 sm:pr-3 text-right font-semibold">Points</th>
            </tr>
          </thead>
          <tbody>
            {weeks.map((w, i) => {
              const pairs = pairTransfers(w.transfers?.in, w.transfers?.out)
              const ft = ledger?.[i]
              const range = weekRange(w)
              const { captain } = captaincy(w)
              const sketch = i >= FIRM_WEEKS
              return (
                <tr
                  key={w.gameweek}
                  className={cx(
                    'border-t border-ink-700/60 align-top transition',
                    i === selectedIndex ? 'bg-fpl-pink/[0.06] shadow-[inset_3px_0_0] shadow-fpl-pink' : 'hover:bg-ink-800/60',
                    sketch && 'text-fg-3',
                  )}
                >
                  <th scope="row" className="py-2.5 pl-4 sm:pl-3 pr-2 text-left font-medium">
                    <button
                      type="button"
                      onClick={() => onSelect(i)}
                      aria-pressed={i === selectedIndex}
                      className="rounded font-semibold text-fg hover:underline cursor-pointer focus-visible:outline-2 focus-visible:outline-fpl-green"
                    >
                      GW{w.gameweek}
                    </button>
                    {i === 0 && <div className="text-[10px] font-normal text-fpl-green">next deadline</div>}
                    {sketch && i === FIRM_WEEKS && <div className="text-[10px] font-normal text-fg-3">likely to change</div>}
                  </th>
                  <td className="py-2.5 px-2">
                    {pairs.length ? (
                      <ul className="space-y-0.5">
                        {pairs.map(([out, inn], k) => (
                          <li key={k} className="sm:whitespace-nowrap">
                            <span className={sketch ? '' : 'text-fg-2'}>{out?.name || '—'}</span>
                            <span className="text-fg-3"> → </span>
                            <span className={sketch ? '' : 'text-fg'}>{inn?.name || '—'}</span>
                          </li>
                        ))}
                      </ul>
                    ) : <span className="text-fg-3">Roll</span>}
                    {/* The columns hidden on phones, in one line */}
                    <div className="sm:hidden mt-0.5 text-[11px] text-fg-3">
                      {[
                        ft && `FT ${ft.used}/${ft.available} → ${ft.next}`,
                        w.transfers?.hits ? `−${w.transfers.hits * 4} hit` : null,
                        w.chip_used && chipLabel(w.chip_used),
                        captain && `C ${captain.web_name}`,
                      ].filter(Boolean).join(' · ')}
                    </div>
                  </td>
                  <td className="hidden sm:table-cell py-2.5 px-2 whitespace-nowrap">
                    {ft ? <>{ft.used}/{ft.available} <span className="text-fg-3">→ {ft.next}</span></> : '–'}
                  </td>
                  <td className="hidden sm:table-cell py-2.5 px-2">{w.transfers?.hits ? <span className="text-warn">−{w.transfers.hits * 4}</span> : <span className="text-fg-3">–</span>}</td>
                  <td className="hidden sm:table-cell py-2.5 px-2">{w.chip_used ? <Badge tone="pink">{chipLabel(w.chip_used)}</Badge> : <span className="text-fg-3">–</span>}</td>
                  <td className="hidden sm:table-cell py-2.5 px-2 whitespace-nowrap">{captain?.web_name || '–'}</td>
                  <td className="py-2.5 pl-2 pr-4 sm:pr-3 text-right whitespace-nowrap">
                    <span className={cx('font-semibold', !sketch && 'text-fg')}>{num(w.expected_points, 0)}</span>
                    {range && <div className="text-[10px] text-fg-3">{num(range.low, 0)}–{num(range.high, 0)}</div>}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
