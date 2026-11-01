import { useEffect, useState } from 'react'
import { Button, ProgressBar, Spinner } from './ui'
import { expectedSeconds } from '../lib/plan'

function stage(elapsed, expected, horizon) {
  const f = elapsed / expected
  if (f < 0.15) return 'Loading your squad and the latest FPL data'
  if (f < 0.4) return `Projecting points for ${horizon} gameweek${horizon === 1 ? '' : 's'}`
  if (f < 0.85) return 'Choosing transfers, captain, bench and chip weeks'
  if (f < 1.2) return 'Checking whether a hit pays'
  return 'Taking longer than usual. If the server was asleep, the first plan can take about a minute.'
}

export default function PlanLoading({ horizon, startedAt, onCancel, who, auto }) {
  const [now, setNow] = useState(() => performance.now())
  useEffect(() => {
    const t = setInterval(() => setNow(performance.now()), 250)
    return () => clearInterval(t)
  }, [])

  const elapsed = Math.max(0, (now - startedAt) / 1000)
  const expected = expectedSeconds(horizon)
  // Fill to 85% by the usual time, then creep, so the bar never claims to be done.
  const progress = elapsed <= expected
    ? (elapsed / expected) * 85
    : 85 + 10 * (1 - Math.exp(-(elapsed - expected) / 30))

  return (
    <div className="flex flex-1 flex-col items-center justify-center px-2 sm:px-6 py-12 sm:py-16">
      <div className="w-full max-w-md">
        <div className="flex items-center justify-between gap-3 text-sm mb-3">
          <span className="flex items-center gap-2 text-fg">
            <Spinner className="text-fpl-green" />
            {auto && who ? `Planning for ${who}` : 'Planning'} · {horizon} GW{horizon === 1 ? '' : 's'}
          </span>
          <span className="text-fg-2 tabular" aria-hidden="true">{Math.floor(elapsed)} s</span>
        </div>
        <ProgressBar value={progress} />
        <p className="mt-3 min-h-[2.5em] text-center text-sm text-fg-2" aria-live="polite">{stage(elapsed, expected, horizon)}</p>
        <p className="mt-1 text-center text-[11px] text-fg-3 tabular">
          Usually about {expected} s for {horizon} weeks{elapsed > expected ? '' : '; up to a minute when the server has been idle'}.
        </p>
        <div className="mt-5 flex justify-center">
          <Button variant="secondary" size="sm" onClick={onCancel}>Cancel</Button>
        </div>
      </div>
    </div>
  )
}
