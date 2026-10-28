import { Badge, Card, CardHeader, SectionLabel } from './ui'
import { IconSwap } from './icons'
import { num, POSITION_SHORT, price, signed } from '../lib/format'

const clean = s => s.replace(/\*\*/g, '').trim()

/**
 * Split the explainer's markdown into per-transfer blocks so each block's reasoning
 * can sit under the structured in/out row it describes.
 */
function parseBlocks(explanation) {
  const blocks = []
  let block = null
  let side = null
  for (const raw of (explanation || '').split('\n')) {
    const line = raw.trim()
    if (line.startsWith('### Transfer')) {
      block = { sell: null, buy: null }
      blocks.push(block)
      side = null
    } else if (block && line.includes('**SELL:**')) {
      side = block.sell = { summary: clean(line.replace('**SELL:**', '')), details: [] }
    } else if (block && line.includes('**BUY:**')) {
      side = block.buy = { summary: clean(line.replace('**BUY:**', '')), details: [] }
    } else if (side && line.startsWith('└')) {
      side.details.push(clean(line.replace(/^└\s*/, '')))
    } else if (line.startsWith('---')) {
      block = side = null
    }
  }
  return blocks
}

/** Pair each incoming player with an outgoing one in the same position, as the explainer does. */
function pairTransfers(ins, outs) {
  const remaining = [...outs]
  const pairs = ins.map(inn => {
    const idx = remaining.findIndex(o => o.element_type === inn.element_type)
    const out = idx >= 0 ? remaining.splice(idx, 1)[0] : null
    return [out, inn]
  })
  return [...pairs, ...remaining.map(o => [o, null])]
}

function PlayerChip({ player, direction }) {
  const isIn = direction === 'in'
  return (
    <div className="flex min-w-0 flex-1 items-center gap-3 rounded-xl bg-ink-900/70 px-3 py-2.5 ring-1 ring-inset ring-ink-700/70">
      <span className={`grid h-6 w-6 shrink-0 place-items-center rounded-md text-xs font-bold ${isIn ? 'bg-good/15 text-good' : 'bg-bad/15 text-bad'}`} aria-label={isIn ? 'In' : 'Out'}>
        {isIn ? '↑' : '↓'}
      </span>
      <div className="min-w-0">
        <div className="truncate text-sm font-medium text-fg">{player.name}</div>
        <div className="text-[11px] text-fg-3 tabular">
          {[POSITION_SHORT[player.element_type], player.cost != null && price(player.cost), isIn && player.expected_points != null && `${num(player.expected_points)} xP`].filter(Boolean).join(' · ')}
        </div>
      </div>
    </div>
  )
}

/** Multi-period plan transfers: structured in/out pairs with reasoning and alternatives. */
export default function TransferPanel({ transfers, summary, explainOpen = false }) {
  const ins = transfers?.in || []
  const outs = transfers?.out || []
  const hits = transfers?.hits || 0
  const xpGain = summary?.xp_gain
  const worthIt = summary?.hit_worth_it

  if (!ins.length && !outs.length) {
    return (
      <Card className="p-4">
        <CardHeader icon={<IconSwap />} title="Transfers" subtitle="No transfers this gameweek — roll the free transfer." />
      </Card>
    )
  }

  const blocks = parseBlocks(transfers.explanation)
  const pairs = pairTransfers(ins, outs)

  return (
    <Card className="p-4">
      <CardHeader
        icon={<IconSwap />}
        title={`${ins.length} transfer${ins.length === 1 ? '' : 's'}`}
        subtitle={hits > 0 ? `${transfers.ft_used ?? 0} free · ${hits} hit${hits === 1 ? '' : 's'}` : 'Free transfers only'}
        action={
          <div className="flex flex-wrap justify-end gap-1.5">
            {xpGain != null && <Badge tone={xpGain >= 0 ? 'green' : 'bad'}>{signed(xpGain)} xP</Badge>}
            {hits > 0 && <Badge tone={worthIt ? 'warn' : 'bad'}>−{hits * 4} hit{worthIt ? ' · worth it' : ' · marginal'}</Badge>}
          </div>
        }
      />

      <div className="mt-4 space-y-3">
        {pairs.map(([out, inn], i) => {
          const block = inn && blocks.find(b => b.buy?.summary.startsWith(inn.name))
          return (
            <div key={i}>
              <div className="flex flex-col sm:flex-row sm:items-center gap-2">
                {out ? <PlayerChip player={out} direction="out" /> : <div className="flex-1" />}
                <span className="hidden sm:block text-fg-3" aria-hidden="true">→</span>
                {inn ? <PlayerChip player={inn} direction="in" /> : <div className="flex-1" />}
              </div>
              {block?.buy && (
                <details open={explainOpen} className="group mt-1.5 ml-1 text-xs">
                  <summary className="cursor-pointer list-none text-fg-3 hover:text-fg-2 select-none">
                    <span className="inline-block transition group-open:rotate-90">›</span> Why this move
                  </summary>
                  <div className="mt-1.5 border-l border-ink-700 pl-3">
                    <p className="text-fg-2">{block.buy.summary}</p>
                    <ul className="mt-1 space-y-0.5 text-fg-3">
                      {block.buy.details.map((l, j) => <li key={j}>{l}</li>)}
                    </ul>
                    {block.sell && <p className="mt-1.5 text-fg-3"><span className="text-fg-2">Sell:</span> {block.sell.summary}</p>}
                  </div>
                </details>
              )}
            </div>
          )
        })}
      </div>

      {transfers.alternatives?.length > 0 && (
        <div className="mt-5 border-t border-ink-700/70 pt-4">
          <SectionLabel className="mb-2.5">Alternatives considered</SectionLabel>
          <div className="grid grid-cols-1 gap-2">
            {transfers.alternatives.map(alt => (
              <div key={alt.position_id ?? alt.position} className="rounded-xl bg-ink-900/60 p-3 ring-1 ring-inset ring-ink-700/60">
                <div className="text-[11px] font-semibold text-fg-2 mb-1.5">{alt.position}</div>
                <ol className="space-y-1">
                  {alt.options.slice(0, 3).map((o, i) => (
                    <li key={o.id} className="flex items-center justify-between gap-2 text-xs tabular">
                      <span className={i === 0 ? 'text-fg font-medium' : 'text-fg-2'}>{i + 1}. {o.name}</span>
                      <span className="text-fg-3">{num(o.expected_points)} · {price(o.price)}</span>
                    </li>
                  ))}
                </ol>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}
