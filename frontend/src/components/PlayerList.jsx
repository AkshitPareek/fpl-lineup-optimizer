import { Badge } from './ui'
import { cx } from '../lib/cx'
import { num, positionOf, price } from '../lib/format'

const GROUPS = [
  { type: 1, label: 'Goalkeepers' },
  { type: 2, label: 'Defenders' },
  { type: 3, label: 'Midfielders' },
  { type: 4, label: 'Forwards' },
]

function Row({ player }) {
  const a = player.analytics
  const cost = player.now_cost != null ? (player.now_cost > 20 ? player.now_cost / 10 : player.now_cost) : null
  return (
    <tr className={cx('border-t border-ink-700/60', player.is_starter === false && 'text-fg-3')}>
      <td className="py-2.5 pl-4 pr-2">
        <div className="flex items-center gap-2">
          <span className={cx('font-medium', player.is_starter === false ? 'text-fg-2' : 'text-fg')}>{player.web_name}</span>
          {player.is_captain && <Badge tone="green">C</Badge>}
          {player.is_vice_captain && <Badge>V</Badge>}
          {player.is_starter === false && <Badge>Bench</Badge>}
          {a?.is_differential && <Badge tone="violet">Diff</Badge>}
          {a?.is_template && <Badge tone="pink">Template</Badge>}
        </div>
        <div className="text-[11px] text-fg-3 mt-0.5">{[player.team_name, positionOf(player)].filter(Boolean).join(' · ')}</div>
      </td>
      <td className="py-2.5 px-2 text-right text-fg-2 hidden sm:table-cell">{cost != null ? price(cost) : '–'}</td>
      <td className="py-2.5 px-2 text-right text-fg-2 hidden md:table-cell">{a?.ownership_pct != null ? `${a.ownership_pct}%` : '–'}</td>
      <td className="py-2.5 px-2 text-right text-fg-2 hidden md:table-cell">{a?.ceiling != null ? num(a.ceiling) : '–'}</td>
      <td className="py-2.5 pl-2 pr-4 text-right font-semibold text-fg">{num(player.expected_points)}</td>
    </tr>
  )
}

export default function PlayerList({ squad }) {
  if (!squad?.length) return null
  return (
    <div className="overflow-hidden rounded-2xl ring-1 ring-inset ring-ink-700">
      <table className="w-full text-sm tabular">
        <thead className="bg-ink-900 text-[11px] uppercase tracking-wider text-fg-3">
          <tr>
            <th className="py-2.5 pl-4 pr-2 text-left font-semibold">Player</th>
            <th className="py-2.5 px-2 text-right font-semibold hidden sm:table-cell">Price</th>
            <th className="py-2.5 px-2 text-right font-semibold hidden md:table-cell">Owned</th>
            <th className="py-2.5 px-2 text-right font-semibold hidden md:table-cell">Ceiling</th>
            <th className="py-2.5 pl-2 pr-4 text-right font-semibold">xP</th>
          </tr>
        </thead>
        {GROUPS.map(g => {
          const players = squad
            .filter(p => p.element_type === g.type)
            .sort((a, b) => (b.is_starter !== false) - (a.is_starter !== false) || (b.expected_points || 0) - (a.expected_points || 0))
          if (!players.length) return null
          return (
            <tbody key={g.type}>
              <tr className="bg-ink-900/50">
                <td colSpan={5} className="py-1.5 pl-4 text-[10px] font-semibold uppercase tracking-[0.14em] text-fg-3">{g.label}</td>
              </tr>
              {players.map(p => <Row key={p.id} player={p} />)}
            </tbody>
          )
        })}
      </table>
    </div>
  )
}
