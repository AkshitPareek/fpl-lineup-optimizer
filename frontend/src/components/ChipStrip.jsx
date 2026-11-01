import { Badge, Card, CardHeader } from './ui'
import { IconChip } from './icons'
import { chipKey, CHIP_LABELS, chipLabel } from '../lib/format'

/**
 * One row per chip: the week the plan plays it, or why it is held, or that it
 * was already used in this half of the season.
 */
export default function ChipStrip({ plan, usedThisHalf, splitGw, forcedChip }) {
  const remaining = (plan.chips_remaining || []).map(chipKey)
  const used = (usedThisHalf || []).map(chipKey)
  const planned = {}
  for (const w of plan.gameweek_plans || []) {
    if (w.chip_used && !planned[chipKey(w.chip_used)]) planned[chipKey(w.chip_used)] = w.gameweek
  }
  const reasons = {}
  for (const h of plan.chip_hold || []) reasons[chipKey(h.chip)] = h.reason
  if (plan.chip_suggestion) reasons[chipKey(plan.chip_suggestion.chip)] = plan.chip_suggestion.reason

  const chips = Object.keys(CHIP_LABELS).filter(c => remaining.includes(c) || used.includes(c) || planned[c])
  if (!chips.length) return null

  return (
    <Card className="p-4 sm:p-5">
      <CardHeader
        icon={<IconChip />}
        title="Chips"
        subtitle={`Each chip can be played once before the GW${splitGw || 19} deadline, and once more after it.`}
      />
      <ul className="mt-4 divide-y divide-ink-700/60">
        {chips.map(c => {
          const week = planned[c]
          const isUsed = used.includes(c) && !remaining.includes(c)
          return (
            <li key={c} className="flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-4 py-2.5 first:pt-0 last:pb-0">
              <div className="flex w-full sm:w-44 shrink-0 items-center justify-between sm:justify-start gap-2">
                <span className={isUsed ? 'text-sm text-fg-3 line-through' : 'text-sm font-semibold text-fg'}>{chipLabel(c)}</span>
                {isUsed ? <Badge>Used</Badge> : week ? <Badge tone="pink">GW{week}</Badge> : <Badge>Hold</Badge>}
              </div>
              <p className="text-xs leading-snug text-fg-3">
                {isUsed
                  ? 'Already played in this half of the season.'
                  : forcedChip && chipKey(forcedChip.chip) === c
                    ? `You asked for it in GW${forcedChip.gw}.`
                    : reasons[c] || (week ? `Planned for GW${week}.` : 'Not planned in these weeks.')}
              </p>
            </li>
          )
        })}
      </ul>
    </Card>
  )
}
