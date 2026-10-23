import { useState } from 'react'
import { api, errorMessage } from '../lib/api'
import { num, price } from '../lib/format'
import { Button, Card, EmptyState, ErrorNote, LoadingState, Stat } from '../components/ui'
import { IconStar } from '../components/icons'
import SquadView from '../components/SquadView'

export default function DreamTeamView({ teamCodes }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dream, setDream] = useState(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.get('/api/dream-team')
      setDream(res.data)
    } catch (err) {
      console.error(err)
      setError(errorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  // The endpoint returns a bare XI; mark everyone as a starter and resolve shirt codes.
  const squad = dream?.dream_team.map(p => ({
    ...p,
    is_starter: true,
    is_captain: p.id === dream.captain?.id,
    team_code: p.team_code ?? teamCodes[p.team],
  }))

  if (!dream || loading) {
    return (
      <Card className="min-h-[480px] flex flex-col">
        {loading ? (
          <LoadingState title="Finding the best XI" />
        ) : (
          <EmptyState
            icon={<IconStar size={24} />}
            title="The no-limits XI"
            action={
              <div className="space-y-3">
                <Button size="lg" onClick={load}>Show dream team</Button>
                <ErrorNote>{error}</ErrorNote>
              </div>
            }
          >
            No budget, no club limits — the eleven highest-projected players for this gameweek in a valid formation.
          </EmptyState>
        )}
      </Card>
    )
  }

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_320px] gap-4 items-start animate-fade-in">
      <Card className="p-4 sm:p-5">
        <SquadView
          squad={squad}
          title={<h2 className="font-display text-lg font-semibold">Gameweek {dream.gameweek}</h2>}
          actions={<Button variant="ghost" size="sm" onClick={load}>Refresh</Button>}
        />
      </Card>
      <div className="grid grid-cols-2 xl:grid-cols-1 gap-3">
        <Stat label="Expected points" value={num(dream.total_expected_points)} sub="Captain doubled" tone="green" />
        <Stat label="Total cost" value={price(dream.total_cost)} sub={dream.total_cost > 100 ? `${price(dream.total_cost - 100)} over a £100m budget` : 'Within a £100m budget'} />
        {dream.captain && (
          <Stat className="col-span-2 xl:col-span-1" label="Captain" value={dream.captain.web_name} sub={`${num(dream.captain.expected_points)} xP · ${dream.captain.team_name}`} />
        )}
        <ErrorNote>{error}</ErrorNote>
      </div>
    </div>
  )
}
