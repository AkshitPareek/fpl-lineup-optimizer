import { useState } from 'react'
import { Badge, Button, Card, SectionLabel } from './ui'
import { IconCheck, IconChip, IconCopy, IconCrown, IconSwap, IconTrendDown, IconTrendUp } from './icons'
import { num, price, signed } from '../lib/format'
import { benchOrder, captaincy, chipLine, deadlineText, pairTransfers, weekRange } from '../lib/plan'
import { buyReason, shortReason } from '../lib/explain'
import { track } from '../lib/telemetry'

function Row({ icon, label, children }) {
  return (
    <div className="grid grid-cols-[20px_minmax(0,1fr)] gap-x-3 py-3 first:pt-0 last:pb-0 border-t first:border-t-0 border-ink-700/60">
      <div className="mt-0.5 text-fg-3" aria-hidden="true">{icon}</div>
      <div className="min-w-0">
        <SectionLabel className="mb-1">{label}</SectionLabel>
        {children}
      </div>
    </div>
  )
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      track('plan_copied')
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard blocked (http, old browser): select the text for a manual copy instead.
      window.prompt('Copy the plan:', text)
    }
  }
  return (
    <Button variant="secondary" size="sm" onClick={copy} aria-live="polite">
      {copied ? <><IconCheck size={14} /> Copied</> : <><IconCopy size={14} /> Copy</>}
    </Button>
  )
}

/**
 * The checklist for the next deadline: transfers, captain, XI and bench, chip,
 * and what is left after. Reads only the plan's first week.
 */
export default function DeadlineCard({ plan, gwPlan, ledger, managerName, priceAlerts, deadlineIn }) {
  const t = gwPlan.transfers || {}
  const pairs = pairTransfers(t.in, t.out)
  const hits = t.hits || 0
  const { captain, vice, next, margin } = captaincy(gwPlan)
  const bench = benchOrder(gwPlan.bench)
  const chip = chipLine(gwPlan, plan)
  const range = weekRange(gwPlan)
  const alerts = Array.isArray(priceAlerts) ? priceAlerts : []
  const text = deadlineText({ gwPlan, ledger, chipAdvice: chip.short, managerName })

  return (
    <Card className="p-4 sm:p-5 ring-1 ring-inset ring-fpl-green/20">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <SectionLabel className="text-fpl-green">Before the deadline</SectionLabel>
          <h2 className="font-display text-xl font-semibold">Gameweek {gwPlan.gameweek}{deadlineIn && <span className="ml-2 text-sm font-sans font-normal text-fg-3">· {deadlineIn} left</span>}</h2>
          <p className="text-xs text-fg-3 tabular mt-0.5">
            Projected <span className="font-semibold text-fg">{num(gwPlan.expected_points, 0)}</span> pts
            {range && <> · likely {num(range.low, 0)}–{num(range.high, 0)}</>}
          </p>
        </div>
        <CopyButton text={text} />
      </div>

      <div className="mt-4">
        <Row icon={<IconSwap size={18} />} label="Transfers">
          {pairs.length === 0 ? (
            <p className="text-sm text-fg">None. Roll the free transfer.</p>
          ) : (
            <ul className="space-y-2">
              {pairs.map(([out, inn], i) => {
                const reason = inn && shortReason(buyReason(t.explanation, inn.name)?.summary)
                const gain = out && inn ? (inn.expected_points || 0) - (out.expected_points || 0) : null
                return (
                  <li key={i}>
                    <div className="flex flex-wrap items-baseline gap-x-2 text-sm">
                      <span className="text-bad">{out ? out.name : '—'}</span>
                      <span className="text-fg-3" aria-label="replaced by">→</span>
                      <span className="font-semibold text-good">{inn ? inn.name : '—'}</span>
                      {gain != null && <span className="text-[11px] text-fg-3 tabular">{num(out.expected_points)} → {num(inn.expected_points)} xP this week ({signed(gain)})</span>}
                    </div>
                    {reason && <p className="mt-0.5 text-[11px] leading-snug text-fg-3">{reason}</p>}
                  </li>
                )
              })}
            </ul>
          )}
          <p className="mt-2 text-xs text-fg-2 tabular">
            {hits > 0 ? <span className="text-warn">{hits} hit{hits > 1 ? 's' : ''} (−{hits * 4})</span> : 'No hits'}
            {typeof gwPlan.bank === 'number' && <> · {price(gwPlan.bank)} in the bank after</>}
            {ledger && <> · {ledger.next} free transfer{ledger.next === 1 ? '' : 's'} next week</>}
          </p>
        </Row>

        <Row icon={<IconCrown size={18} />} label="Captain">
          {captain ? (
            <>
              <p className="text-sm">
                <span className="font-semibold text-fg">{captain.web_name}</span>{' '}
                <span className="text-fg-3 tabular">{num(captain.expected_points)} xP</span>
                {next && margin != null && (
                  <span className="text-fg-3 tabular"> · {signed(margin)} over {next.web_name} ({num(next.expected_points)})</span>
                )}
              </p>
              {vice && <p className="text-xs text-fg-3">Vice-captain <span className="text-fg-2">{vice.web_name}</span></p>}
            </>
          ) : <p className="text-sm text-fg-3">—</p>}
        </Row>

        <Row icon={<span className="text-[11px] font-bold">XI</span>} label="Starting XI and bench">
          <p className="text-sm text-fg leading-relaxed">
            {[1, 2, 3, 4].map(type => gwPlan.starting_xi.filter(p => p.element_type === type).map(p => p.web_name).join(', ')).filter(Boolean).join('  ·  ')}
          </p>
          {bench.length > 0 && (
            <ol className="mt-1.5 flex flex-wrap gap-1.5 text-[11px]" aria-label="Bench order">
              {bench.map((p, i) => (
                <li key={p.id} className="rounded-md bg-ink-900 px-2 py-0.5 text-fg-2 ring-1 ring-inset ring-ink-700">
                  <span className="text-fg-3">{i === 0 ? 'GK' : i}.</span> {p.web_name}
                </li>
              ))}
            </ol>
          )}
        </Row>

        <Row icon={<IconChip size={18} />} label="Chip">
          <p className="text-sm text-fg">
            {chip.text}
            {chip.play && <Badge tone="pink" className="ml-2">Play</Badge>}
          </p>
        </Row>

        {alerts.length > 0 && (
          <Row icon={<IconTrendUp size={18} />} label="Price changes to beat">
            <ul className="space-y-1 text-xs">
              {alerts.map(a => (
                <li key={`${a.id}-${a.direction}`} className="flex items-center gap-2 text-fg-2">
                  {a.direction === 'rise' ? <IconTrendUp size={14} className="text-good" /> : <IconTrendDown size={14} className="text-bad" />}
                  <span>
                    <span className="font-medium text-fg">{a.web_name}</span>{' '}
                    {a.direction === 'rise' ? 'likely to rise' : 'likely to fall'}{' '}
                    <span className="tabular">({Math.round((a.chance || 0) * 100)}%)</span>
                    {a.direction === 'rise' ? ': buy before it does.' : ': sell before it does, if you are selling.'}
                  </span>
                </li>
              ))}
            </ul>
          </Row>
        )}
      </div>
    </Card>
  )
}
