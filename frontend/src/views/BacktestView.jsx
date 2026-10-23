import { useEffect, useState } from 'react'
import { streamEvents } from '../lib/api'
import { cx } from '../lib/cx'
import { num, signed } from '../lib/format'
import { Button, Card, CardHeader, EmptyState, ErrorNote, Field, LoadingState, Spinner, Stat, Stepper } from '../components/ui'
import { IconHistory } from '../components/icons'
import SquadView from '../components/SquadView'
import BacktestChart from '../components/BacktestChart'
import TransferPanel from '../components/TransferPanel'
import { Canvas, Workspace } from './Workspace'

export default function BacktestView({ season }) {
  const [budget, setBudget] = useState(100)
  const [startGw, setStartGw] = useState(1)
  const [endGw, setEndGw] = useState(5)
  const [endTouched, setEndTouched] = useState(false)
  const [horizon, setHorizon] = useState(5)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(null)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)
  const [selectedGw, setSelectedGw] = useState(null)

  const finished = Number(season?.finished_gameweeks) || 0
  const lastGw = finished > 0 ? finished : 38

  // Default the range to every finished gameweek once season data arrives.
  useEffect(() => {
    if (!endTouched && finished > 0) setEndGw(finished)
  }, [finished, endTouched])

  const run = async () => {
    setRunning(true)
    setError(null)
    setResults(null)
    setSelectedGw(null)
    setProgress({ progress: 0, message: 'Initialising…' })
    let finished = false
    try {
      await streamEvents('/api/backtest/stream', {
        start_gw: parseInt(startGw),
        end_gw: parseInt(endGw),
        initial_budget: parseFloat(budget),
        horizon: parseInt(horizon),
      }, event => {
        if (event.result) {
          finished = true
          setResults(event.result)
          const weeks = event.result.weekly_results
          const withTransfers = weeks.find(w => w.transfers?.in?.length || w.transfers?.out?.length)
          setSelectedGw((withTransfers || weeks[0])?.gameweek ?? null)
        } else {
          setProgress({ progress: event.progress, message: event.message })
        }
      })
      if (!finished) setError('Backtest stopped before it finished. Check the server logs and try a shorter range.')
    } catch (err) {
      setError('Backtest failed: ' + err.message)
    } finally {
      setRunning(false)
      setProgress(null)
    }
  }

  const m = results?.metrics
  const weeks = results?.weekly_results || []
  const selected = weeks.find(w => w.gameweek === selectedGw)
  const invalidRange = Number(startGw) > Number(endGw)

  const settings = (
    <>
      <div className="grid grid-cols-2 gap-3">
        <Field label="From GW">
          <Stepper value={startGw} onChange={setStartGw} min={1} max={lastGw} />
        </Field>
        <Field label="To GW">
          <Stepper value={endGw} onChange={v => { setEndTouched(true); setEndGw(v) }} min={1} max={lastGw} />
        </Field>
      </div>
      {finished > 0 && <p className="-mt-3 text-[11px] text-fg-3">{finished} gameweek{finished === 1 ? '' : 's'} finished so far this season.</p>}
      <Field label="Optimiser lookahead" aside={<span className="text-xs font-semibold text-fg tabular">{horizon} GWs</span>}>
        <input type="range" min="3" max="15" value={horizon} onChange={e => setHorizon(Number(e.target.value))} className="w-full" aria-label="Optimiser lookahead" />
      </Field>
      <Field label="Starting budget">
        <Stepper value={budget} onChange={setBudget} min={80} max={120} step={0.5} suffix="£m" />
      </Field>
      <Button size="lg" className="w-full" onClick={run} disabled={running || invalidRange}>
        {running ? <><Spinner /> Simulating…</> : 'Run simulation'}
      </Button>
      <ErrorNote>{invalidRange ? 'The start gameweek must come before the end.' : error}</ErrorNote>
    </>
  )

  if (progress || !results) {
    return (
      <Workspace settings={settings}>
        <Canvas>
          {progress ? (
            <LoadingState title={progress.message} progress={progress.progress} detail="Re-solving the plan each gameweek with only the information available at the time." />
          ) : (
            <EmptyState icon={<IconHistory size={24} />} title="How would the optimiser have done?">
              Replays past gameweeks with a rolling-horizon plan and scores each decision against what actually happened.
            </EmptyState>
          )}
        </Canvas>
      </Workspace>
    )
  }

  return (
    <Workspace settings={settings}>
      <div className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="Actual points" value={m.total_actual_points} sub="Net of hits" tone="green" />
          <Stat label="Predicted" value={num(m.total_predicted_points, 0)} />
          <Stat
            label="Over-prediction"
            value={signed(m.prediction_error, 0)}
            sub={m.prediction_error > 0 ? 'Model was optimistic' : 'Model was conservative'}
          />
          <Stat
            label="Avg per GW"
            value={num(m.avg_points_per_gw)}
            sub={m.efficiency_pct != null ? `${num(m.efficiency_pct, 0)}% of hindsight best` : null}
          />
        </div>

        <Card className="p-4 sm:p-5">
          <CardHeader title={`Gameweeks ${startGw}–${endGw}`} subtitle="Click a gameweek to see the lineup that was picked." />
          <div className="mt-4">
            <BacktestChart weeks={weeks} selected={selectedGw} onSelect={setSelectedGw} />
          </div>
        </Card>

        <div className="grid grid-cols-1 xl:grid-cols-[340px_minmax(0,1fr)] gap-4 items-start">
          <Card className="overflow-hidden">
            <div className="max-h-[640px] overflow-y-auto scrollbar-thin">
              <table className="w-full text-sm tabular">
                <thead className="sticky top-0 bg-ink-900 text-[11px] uppercase tracking-wider text-fg-3">
                  <tr>
                    <th className="py-2.5 pl-4 text-left font-semibold">GW</th>
                    <th className="py-2.5 px-2 text-right font-semibold">Pts</th>
                    <th className="py-2.5 px-2 text-right font-semibold">xP</th>
                    <th className="py-2.5 px-2 text-right font-semibold">Hits</th>
                    <th className="py-2.5 pr-4 text-right font-semibold" title="Share of the template covered">Templ</th>
                  </tr>
                </thead>
                <tbody>
                  {weeks.map(w => (
                    <tr
                      key={w.gameweek}
                      onClick={() => setSelectedGw(w.gameweek)}
                      className={cx(
                        'cursor-pointer border-t border-ink-700/60 transition',
                        w.gameweek === selectedGw ? 'bg-fpl-green/[0.07] shadow-[inset_3px_0_0] shadow-fpl-green' : 'hover:bg-ink-800',
                      )}
                    >
                      <td className="py-2.5 pl-4 font-medium text-fg">GW{w.gameweek}</td>
                      <td className="py-2.5 px-2 text-right font-semibold text-fg">{w.actual_points}</td>
                      <td className="py-2.5 px-2 text-right text-fg-2">{num(w.predicted_points)}</td>
                      <td className="py-2.5 px-2 text-right text-fg-3">{w.hits_cost ? `−${w.hits_cost}` : '–'}</td>
                      <td className="py-2.5 pr-4 text-right text-fg-3">{w.template_coverage ? `${num(w.template_coverage, 0)}%` : '–'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <div className="space-y-4">
            {selected?.transfers && <TransferPanel transfers={selected.transfers} summary={selected.transfers} />}
            <Card className="p-4 sm:p-5">
              {selected ? (
                <SquadView
                  squad={selected.squad || []}
                  title={
                    <div>
                      <h2 className="font-display text-lg font-semibold">GW{selected.gameweek} lineup</h2>
                      <p className="text-xs text-fg-3 tabular">{selected.actual_points} pts scored · {num(selected.predicted_points)} predicted</p>
                    </div>
                  }
                />
              ) : (
                <EmptyState title="Select a gameweek" />
              )}
            </Card>
          </div>
        </div>
      </div>
    </Workspace>
  )
}
