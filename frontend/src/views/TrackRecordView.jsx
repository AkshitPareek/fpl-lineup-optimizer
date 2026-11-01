import { useEffect, useState } from 'react'
import { api, friendlyError, withRetry } from '../lib/api'
import { errorKind, track } from '../lib/telemetry'
import { chipLabel, num } from '../lib/format'
import { Button, Card, CardHeader, EmptyState, LoadingState, Stat } from '../components/ui'
import { IconTrophy } from '../components/icons'

function sum(rows, key) {
  return rows.reduce((t, r) => t + (Number(r[key]) || 0), 0)
}

/**
 * The live planner's advice, recorded a few hours before each deadline for a
 * tracked team and scored with real points once the gameweek is over.
 */
export default function TrackRecordView() {
  const [state, setState] = useState({ loading: true })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let cancelled = false
    withRetry(() => api.get('/api/track-record'))
      .then(res => { if (!cancelled) setState({ data: res.data }) })
      .catch(err => {
        track('error_shown', { where: 'track_record', kind: errorKind(err) })
        if (!cancelled) setState({ error: friendlyError(err) })
      })
    return () => { cancelled = true }
  }, [attempt])

  if (state.loading) return <Card><LoadingState title="Loading the track record" /></Card>
  if (state.error) {
    const missing = state.error.kind === 'http_404'
    return (
      <Card>
        <EmptyState
          icon={<IconTrophy size={24} />}
          title={missing ? 'Not available yet' : "Couldn't load the track record"}
          action={!missing && <Button variant="secondary" onClick={() => { setState({ loading: true }); setAttempt(n => n + 1) }}>Try again</Button>}
        >
          {missing ? 'The server does not serve the track record yet.' : state.error.message}
        </EmptyState>
      </Card>
    )
  }

  const weeks = (state.data?.teams || []).flatMap(t => t.weeks || []).sort((a, b) => a.gameweek - b.gameweek)
  const scored = weeks.filter(w => w.finished && w.advice != null)
  if (!weeks.length) {
    return (
      <Card>
        <EmptyState icon={<IconTrophy size={24} />} title="Nothing recorded yet">
          The advice is recorded a few hours before each deadline and scored after the gameweek.
        </EmptyState>
      </Card>
    )
  }

  const advice = sum(scored, 'advice')
  const average = sum(scored, 'average')
  const projected = sum(scored, 'projected')
  const withManager = scored.filter(w => w.you != null)

  return (
    <div className="space-y-4">
      {scored.length === 0 && (
        <p className="rounded-xl bg-ink-900/60 px-3.5 py-2.5 text-xs text-fg-3 ring-1 ring-inset ring-ink-700/70">
          Recording started recently. Totals appear once the first recorded gameweek has finished.
        </p>
      )}
      {scored.length > 0 && <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Stat label="Weeks scored" value={scored.length} sub={`${weeks.length - scored.length} waiting for results`} />
        <Stat label="Advice scored" value={scored.length ? advice : '–'} tone="green" sub={scored.length ? `${num(advice / scored.length)} a week` : 'After the first finished week'} />
        <Stat label="vs average manager" value={scored.length ? `${advice - average >= 0 ? '+' : '−'}${Math.abs(advice - average)}` : '–'} sub={scored.length ? `${average} for the average` : null} />
        <Stat label="vs projection" value={scored.length ? `${advice - projected >= 0 ? '+' : '−'}${num(Math.abs(advice - projected), 0)}` : '–'} sub={scored.length ? `${num(projected, 0)} projected` : null} />
      </div>}

      <Card className="p-4 sm:p-5">
        <CardHeader
          icon={<IconTrophy />}
          title="Week by week"
          subtitle="The planner's advice for one tracked team, recorded a few hours before each deadline and scored with real points, autosubs and the vice-captain rule. It shows what following that week's advice would have scored."
        />
        <div className="mt-4 -mx-4 sm:mx-0 overflow-x-auto scrollbar-thin">
          <table className="w-full min-w-[560px] text-sm tabular">
            <thead className="text-[11px] uppercase tracking-wider text-fg-3">
              <tr className="text-left">
                <th scope="col" className="py-2 pl-4 sm:pl-3 pr-2 font-semibold">GW</th>
                <th scope="col" className="py-2 px-2 font-semibold">Advice</th>
                <th scope="col" className="py-2 px-2 font-semibold">Captain</th>
                <th scope="col" className="py-2 px-2 text-right font-semibold">Projected</th>
                <th scope="col" className="py-2 px-2 text-right font-semibold">Scored</th>
                {withManager.length > 0 && <th scope="col" className="py-2 px-2 text-right font-semibold">Manager</th>}
                <th scope="col" className="py-2 pl-2 pr-4 sm:pr-3 text-right font-semibold">Average</th>
              </tr>
            </thead>
            <tbody>
              {weeks.map(w => (
                <tr key={w.gameweek} className="border-t border-ink-700/60 align-top">
                  <th scope="row" className="py-2.5 pl-4 sm:pl-3 pr-2 text-left font-semibold text-fg">GW{w.gameweek}</th>
                  <td className="py-2.5 px-2 text-fg-2">
                    {w.transfers_in?.length
                      ? w.transfers_out.map((o, i) => `${o} → ${w.transfers_in[i] || '—'}`).join(', ')
                      : 'No transfer'}
                    {w.hits > 0 && <span className="text-warn"> · −{w.hits * 4}</span>}
                    {w.chip && <span className="text-fpl-pink"> · {chipLabel(w.chip)}</span>}
                  </td>
                  <td className="py-2.5 px-2 text-fg-2">{w.captain || '–'}{w.captain_points != null && <span className="text-fg-3"> ({w.captain_points})</span>}</td>
                  <td className="py-2.5 px-2 text-right text-fg-2">{num(w.projected, 0)}</td>
                  {w.finished && w.advice != null ? (
                    <>
                      <td className="py-2.5 px-2 text-right font-semibold text-fpl-green">{w.advice}</td>
                      {withManager.length > 0 && <td className="py-2.5 px-2 text-right text-fg-2">{w.you ?? '–'}</td>}
                      <td className="py-2.5 pl-2 pr-4 sm:pr-3 text-right text-fg-2">{w.average}</td>
                    </>
                  ) : (
                    <td colSpan={withManager.length > 0 ? 3 : 2} className="py-2.5 pl-2 pr-4 sm:pr-3 text-right text-fg-3">Scored when the gameweek ends</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
