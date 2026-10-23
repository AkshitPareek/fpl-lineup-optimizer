import { useState, useEffect } from 'react'
import axios from 'axios'
import { Card, CardHeader, LoadingState, SectionLabel } from '../ui'
import { IconPound, IconTrendDown, IconTrendUp } from '../icons'

function MoverList({ players, empty, tone }) {
    if (!players.length) return <p className="text-xs text-fg-3 italic py-2">{empty}</p>
    return (
        <ul className="divide-y divide-ink-700/60">
            {players.slice(0, 5).map((p, i) => (
                <li key={i} className="flex items-center justify-between py-2 text-sm tabular">
                    <span className="text-fg">{p.web_name}</span>
                    <span className="text-right">
                        <span className="block text-xs text-fg-2">£{p.current_price}m</span>
                        <span className={`block text-[11px] ${tone === 'up' ? 'text-good' : 'text-bad'}`}>
                            {tone === 'up' ? '+' : ''}{p.net_transfers.toLocaleString()} net
                        </span>
                    </span>
                </li>
            ))}
        </ul>
    )
}

function LiveTracker() {
    const [summary, setSummary] = useState(null)
    const [risers, setRisers] = useState([])
    const [fallers, setFallers] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [sumRes, riseRes, fallRes] = await Promise.all([
                    axios.get('/api/live/summary'),
                    axios.get('/api/live/price-risers?min_net_transfers=10000'),
                    axios.get('/api/live/price-fallers?min_net_out=10000')
                ])
                setSummary(sumRes.data)
                setRisers(riseRes.data.risers)
                setFallers(fallRes.data.fallers)
            } catch (err) {
                console.error("Error fetching live stats", err)
            } finally {
                setLoading(false)
            }
        }
        fetchData()
    }, [])

    if (loading) return <Card className="flex min-h-[320px]"><LoadingState title="Connecting to the FPL live API" /></Card>

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-start">
            <Card className="lg:col-span-2 p-5">
                <CardHeader
                    title={<span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-fpl-pink animate-pulse" />Gameweek {summary?.gameweek ?? '–'}</span>}
                    subtitle="Top performers and transfer trends"
                />

                <SectionLabel className="mt-5 mb-2.5">Top performers</SectionLabel>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                    {summary?.top_scorers?.slice(0, 6).map((player, idx) => (
                        <div key={idx} className="flex items-center justify-between rounded-xl bg-ink-900/70 px-3 py-2.5 ring-1 ring-inset ring-ink-700/70">
                            <div className="flex items-center gap-2.5 min-w-0">
                                <span className="grid h-6 w-6 shrink-0 place-items-center rounded-md bg-ink-700 text-[11px] font-semibold text-fg-2 tabular">{idx + 1}</span>
                                <div className="min-w-0">
                                    <p className="truncate text-sm font-medium text-fg">{player.web_name}</p>
                                    <p className="text-[11px] text-fg-3">{player.team}</p>
                                </div>
                            </div>
                            <span className="font-display text-lg font-semibold text-fpl-green tabular">{player.event_points}</span>
                        </div>
                    ))}
                </div>

                <SectionLabel className="mt-6 mb-2.5">Transfer trends</SectionLabel>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {[
                        { title: 'Most transferred in', list: summary?.most_transferred_in, key: 'transfers_in', up: true },
                        { title: 'Most transferred out', list: summary?.most_transferred_out, key: 'transfers_out', up: false },
                    ].map(col => (
                        <div key={col.title} className="rounded-xl bg-ink-900/60 p-3 ring-1 ring-inset ring-ink-700/70">
                            <p className="flex items-center gap-1.5 text-xs font-semibold text-fg-2 mb-1.5">
                                {col.up ? <IconTrendUp size={14} className="text-good" /> : <IconTrendDown size={14} className="text-bad" />}
                                {col.title}
                            </p>
                            {col.list?.slice(0, 3).map((p, i) => (
                                <div key={i} className="flex justify-between py-1 text-xs tabular">
                                    <span className="text-fg">{p.web_name}</span>
                                    <span className="text-fg-2">{col.up ? '+' : '−'}{p[col.key].toLocaleString()}</span>
                                </div>
                            ))}
                        </div>
                    ))}
                </div>
            </Card>

            <Card className="p-5">
                <CardHeader icon={<IconPound />} title="Price watch" subtitle="Net transfer pressure since the last change" />
                <SectionLabel className="mt-5 mb-1 flex items-center gap-1.5"><IconTrendUp size={13} className="text-good" />Likely risers</SectionLabel>
                <MoverList players={risers} tone="up" empty="No imminent price rises detected" />
                <SectionLabel className="mt-5 mb-1 flex items-center gap-1.5"><IconTrendDown size={13} className="text-bad" />Likely fallers</SectionLabel>
                <MoverList players={fallers} tone="down" empty="No imminent price falls detected" />
            </Card>
        </div>
    )
}

export default LiveTracker
