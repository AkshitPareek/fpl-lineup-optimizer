import { useState } from 'react'
import { api, errorMessage } from '../lib/api'
import { num, price } from '../lib/format'
import { Badge, Button, Card, EmptyState, ErrorNote, LoadingState, Spinner, Stat } from '../components/ui'
import { IconBolt } from '../components/icons'
import SquadView from '../components/SquadView'
import TransferPanel from '../components/TransferPanel'
import { ManagerFields } from './PlannerView'
import { Canvas, Workspace } from './Workspace'

function recommendedStrategy(compare) {
  const noHits = compare.comparison.no_hits
  const feasible = noHits?.feasible !== false && compare.no_hits?.gameweek_plans?.length > 0
  return feasible && noHits?.recommended ? 'no_hits' : 'with_hits'
}

export default function QuickPickView({
  managerId, setManagerId, validManagerId, budget, setBudget, bankedTransfers,
}) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const run = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.post('/api/optimize/compare', {
        budget: parseFloat(budget),
        gameweeks: 1,
        manager_id: validManagerId ? parseInt(validManagerId) : null,
        banked_transfers: parseInt(bankedTransfers) || 1,
        strategy: 'standard',
      })
      const { with_hits: withHits, no_hits: noHits, note } = res.data.comparison
      if (withHits?.feasible === false && noHits?.feasible === false) {
        setError(note || 'No lineup could be solved.')
        return
      }
      const key = recommendedStrategy(res.data)
      const plan = res.data[key]
      setResult({ plan, gw: plan.gameweek_plans?.[0], hits: key === 'with_hits' })
    } catch (err) {
      console.error(err)
      setError(errorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  const gw = result?.gw
  const squad = gw ? [...gw.starting_xi, ...gw.bench] : null
  const summary = gw && result.plan.transfer_summary?.find(t => t.gameweek === gw.gameweek)

  const settings = (
    <>
      <ManagerFields {...{ managerId, setManagerId, validManagerId, budget, setBudget }} />
      <p className="text-xs text-fg-3">
        Scores the next gameweek with the same projections as the Planner. The Planner is the team to set, because it keeps the weeks after this one.
      </p>
      <Button size="lg" className="w-full" onClick={run} disabled={loading || !validManagerId}>
        {loading ? <><Spinner /> Solving…</> : 'Show this week'}
      </Button>
      {!validManagerId && <p className="text-xs text-fg-3">Enter your manager ID so this uses your squad, bank and free transfers.</p>}
      <ErrorNote>{error}</ErrorNote>
    </>
  )

  return (
    <Workspace settings={settings}>
      {loading || !squad ? (
        <Canvas>
          {loading ? (
            <LoadingState title="Scoring this gameweek" detail="Using your squad and the same projections as the Planner." />
          ) : (
            <EmptyState icon={<IconBolt size={24} />} title="This gameweek, from your squad">
              A one-week look at the same model. For the team to set at the deadline, use the Planner.
            </EmptyState>
          )}
        </Canvas>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <Stat label={`GW${gw.gameweek} xP`} value={num(gw.expected_points)} tone="green" />
            <Stat label="In the bank" value={price(gw.bank)} />
            <Stat label="Hits" value={gw.transfers?.hits ? `−${gw.transfers.hits * 4}` : '0'} tone={gw.transfers?.hits ? 'bad' : undefined} />
          </div>
          <TransferPanel transfers={gw.transfers} summary={summary ?? gw.transfers} explainOpen />
          <Card className="p-4 sm:p-5">
            <div className="mb-4 flex items-center justify-between gap-3">
              <h2 className="font-display text-lg font-semibold">Gameweek {gw.gameweek}</h2>
              {result.hits && <Badge tone="warn">Includes a hit</Badge>}
            </div>
            <SquadView squad={squad} />
          </Card>
        </div>
      )}
    </Workspace>
  )
}
