import { useState, useEffect } from 'react'
import axios from 'axios'
import { Card, CardHeader, LoadingState } from '../ui'
import { IconShield, IconSparkle } from '../icons'

// Floor → ceiling range with the expected value marked; all players share one 0–maxVal scale.
function EVRange({ floor, expected, ceiling, maxVal }) {
    const pct = v => `${Math.max(0, Math.min(100, (v / maxVal) * 100))}%`
    return (
        <div className="mt-2">
            <div className="relative h-2 rounded-full bg-ink-900 ring-1 ring-inset ring-ink-700/60">
                <div
                    className="absolute inset-y-0 rounded-full bg-fpl-green/30"
                    style={{ left: pct(floor), width: `calc(${pct(ceiling)} - ${pct(floor)})` }}
                />
                <div
                    className="absolute -top-1 h-4 w-1 -translate-x-1/2 rounded-full bg-fg ring-2 ring-ink-850"
                    style={{ left: pct(expected) }}
                />
            </div>
            <div className="mt-1.5 flex justify-between text-[11px] text-fg-3 tabular">
                <span>Floor {floor.toFixed(1)}</span>
                <span className="text-fg">xP {expected.toFixed(1)}</span>
                <span>Ceiling {ceiling.toFixed(1)}</span>
            </div>
        </div>
    )
}

function PlayerRanges({ players, metric }) {
    const top = players.slice(0, 5)
    const maxVal = Math.max(10, ...top.map(p => p.ceiling || 0))
    return (
        <div className="mt-4 space-y-3">
            {top.map((player, idx) => (
                <div key={idx} className="rounded-xl bg-ink-900/50 px-3 py-2.5 ring-1 ring-inset ring-ink-700/60">
                    <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-fg">{player.web_name}</span>
                        <span className="text-xs text-fg-2 tabular">{metric(player)}</span>
                    </div>
                    <EVRange floor={player.floor} expected={player.expected_points} ceiling={player.ceiling} maxVal={maxVal} />
                </div>
            ))}
        </div>
    )
}

function EVCharts() {
    const [highCeiling, setHighCeiling] = useState([])
    const [safePlayers, setSafePlayers] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [ceilingRes, safeRes] = await Promise.all([
                    axios.get('/api/ev/high-ceiling'),
                    axios.get('/api/ev/safe-players')
                ])
                setHighCeiling(ceilingRes.data.high_ceiling_players)
                setSafePlayers(safeRes.data.safe_players)
            } catch (err) {
                console.error("Error fetching EV stats", err)
            } finally {
                setLoading(false)
            }
        }
        fetchData()
    }, [])

    if (loading) return <Card className="flex min-h-[320px]"><LoadingState title="Calculating EV distributions" /></Card>

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-start">
            <Card className="p-5">
                <CardHeader icon={<IconSparkle />} title="Explosive potential" subtitle="Highest 90th-percentile outcomes. High risk, high reward." />
                <PlayerRanges players={highCeiling} metric={p => `Upside +${p.upside?.toFixed(1)}`} />
            </Card>
            <Card className="p-5">
                <CardHeader icon={<IconShield />} title="Safe anchors" subtitle="High floor, low variance. Reliable captain options." />
                <PlayerRanges players={safePlayers} metric={p => `Risk ${p.risk_score?.toFixed(2)}`} />
            </Card>
        </div>
    )
}

export default EVCharts
