import { useState } from 'react'
import { api, errorMessage } from '../lib/api'
import { num, price } from '../lib/format'
import { Badge, Button, Card, CardHeader, EmptyState, ErrorNote, Field, LoadingState, Select, Spinner, Stat, Stepper } from '../components/ui'
import { IconBolt, IconSwap } from '../components/icons'
import SquadView from '../components/SquadView'
import { ManagerFields } from './PlannerView'
import { Canvas, Workspace } from './Workspace'

const MODES = {
  standard: { label: 'Best squad (wildcard)', hint: 'Build the best 15 from scratch within budget.' },
  my_squad: { label: 'Best XI from my squad', hint: 'Pick the starting XI and captain from your current 15.', needsManager: true },
  transfers: { label: 'Suggest transfers', hint: 'Recommend transfers from your current squad.', needsManager: true },
}

export default function QuickPickView({ managerId, setManagerId, validManagerId, budget, setBudget }) {
  const [strategy, setStrategy] = useState('standard')
  const [freeTransfers, setFreeTransfers] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const mode = MODES[strategy]
  const blocked = mode.needsManager && !validManagerId

  const run = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.post('/api/optimize', {
        budget: parseFloat(budget),
        gameweeks: 1,
        strategy,
        manager_id: validManagerId ? parseInt(validManagerId) : null,
        free_transfers: parseInt(freeTransfers),
      })
      setResult(res.data)
    } catch (err) {
      console.error(err)
      setError(errorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  const squad = result?.lineup || result?.squad
  const transfers = result?.transfers

  const settings = (
    <>
      <Field label="Mode" hint={blocked ? 'This mode needs your manager ID.' : mode.hint}>
        <Select value={strategy} onChange={e => setStrategy(e.target.value)}>
          {Object.entries(MODES).map(([value, m]) => <option key={value} value={value}>{m.label}</option>)}
        </Select>
      </Field>
      <ManagerFields {...{ managerId, setManagerId, validManagerId, budget, setBudget }} />
      {strategy === 'transfers' && (
        <Field label="Free transfers" hint="Transfers beyond this cost −4 each.">
          <Stepper value={freeTransfers} onChange={setFreeTransfers} min={0} max={5} />
        </Field>
      )}
      <Button size="lg" className="w-full" onClick={run} disabled={loading || blocked}>
        {loading ? <><Spinner /> Optimising…</> : 'Optimise lineup'}
      </Button>
      <ErrorNote>{error}</ErrorNote>
    </>
  )

  return (
    <Workspace settings={settings}>
      {loading || !squad ? (
        <Canvas>
          {loading ? (
            <LoadingState title="Optimising lineup" detail="Picking the best XI and captain for this gameweek." />
          ) : (
            <EmptyState icon={<IconBolt size={24} />} title="One gameweek, one answer">
              The fastest way to a lineup: best squad, best XI and captain for the upcoming gameweek only.
            </EmptyState>
          )}
        </Canvas>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Expected points" value={num(result.total_expected_points)} tone="green" />
            <Stat label="Squad cost" value={price(result.budget_used)} />
            <Stat label="In the bank" value={price(result.bank)} />
            <Stat label="Transfer cost" value={transfers?.cost ? `−${transfers.cost}` : '0'} tone={transfers?.cost ? 'bad' : undefined} />
          </div>

          {transfers && (
            <Card className="p-4">
              <CardHeader icon={<IconSwap />} title="Suggested transfers" subtitle={transfers.in?.length ? null : 'Your squad is already optimal for this gameweek.'} />
              {transfers.in?.length > 0 && (
                <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-wider text-fg-3 mb-1.5">Out</div>
                    <div className="flex flex-wrap gap-1.5">{transfers.out.map(n => <Badge key={n} tone="bad" className="normal-case text-xs">↓ {n}</Badge>)}</div>
                  </div>
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-wider text-fg-3 mb-1.5">In</div>
                    <div className="flex flex-wrap gap-1.5">{transfers.in.map(n => <Badge key={n} tone="green" className="normal-case text-xs">↑ {n}</Badge>)}</div>
                  </div>
                </div>
              )}
            </Card>
          )}

          <Card className="p-4 sm:p-5">
            <SquadView squad={squad} title={<h2 className="font-display text-lg font-semibold">Your lineup</h2>} />
          </Card>
        </div>
      )}
    </Workspace>
  )
}
