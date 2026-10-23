import { useState, useEffect } from 'react'
import axios from 'axios'
import { Card, CardHeader, LoadingState } from '../ui'
import { IconShield, IconTarget } from '../icons'

function OwnershipStats() {
    const [template, setTemplate] = useState([])
    const [differentials, setDifferentials] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [tempRes, diffRes] = await Promise.all([
                    axios.get('/api/ownership/template'),
                    axios.get('/api/ownership/differentials')
                ])
                setTemplate(tempRes.data.template_players)
                setDifferentials(diffRes.data.differentials)
            } catch (err) {
                console.error("Error fetching ownership stats", err)
            } finally {
                setLoading(false)
            }
        }
        fetchData()
    }, [])

    if (loading) return <Card className="flex min-h-[320px]"><LoadingState title="Loading ownership data" /></Card>

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] gap-4 items-start">
            <Card className="overflow-hidden">
                <CardHeader className="p-5 pb-4" icon={<IconShield />} title="Template players" subtitle="The most-owned players — not owning them is a risk." />
                <div className="overflow-x-auto">
                    <table className="w-full text-sm tabular">
                        <thead className="bg-ink-900 text-[11px] uppercase tracking-wider text-fg-3">
                            <tr>
                                <th className="py-2.5 pl-5 text-left font-semibold">Player</th>
                                <th className="py-2.5 px-2 text-right font-semibold">Ownership</th>
                                <th className="py-2.5 px-2 text-right font-semibold">EO (est.)</th>
                                <th className="py-2.5 pr-5 text-right font-semibold">Price</th>
                            </tr>
                        </thead>
                        <tbody>
                            {template.slice(0, 10).map((player, idx) => (
                                <tr key={idx} className="border-t border-ink-700/60 hover:bg-ink-800/60 transition">
                                    <td className="py-2.5 pl-5">
                                        <span className="font-medium text-fg">{player.web_name}</span>
                                        <span className="ml-1.5 text-[11px] text-fg-3">{player.team}</span>
                                    </td>
                                    <td className="py-2.5 px-2">
                                        <div className="flex items-center justify-end gap-2">
                                            <div className="hidden sm:block w-16 h-1.5 rounded-full bg-ink-700">
                                                <div className="h-1.5 rounded-full bg-fpl-green/80" style={{ width: `${Math.min(player.ownership_pct, 100)}%` }} />
                                            </div>
                                            <span className="w-12 text-right font-semibold text-fg">{player.ownership_pct}%</span>
                                        </div>
                                    </td>
                                    <td className="py-2.5 px-2 text-right text-fg-2">{player.effective_ownership?.toFixed(1)}%</td>
                                    <td className="py-2.5 pr-5 text-right text-fg-2">£{player.price}m</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </Card>

            <Card className="p-5">
                <CardHeader icon={<IconTarget />} title="Top differentials" subtitle="Low-owned picks that can swing your rank." />
                <div className="mt-4 space-y-2">
                    {differentials.slice(0, 6).map((player, idx) => (
                        <div key={idx} className="flex items-center justify-between rounded-xl bg-ink-900/60 px-3 py-2.5 ring-1 ring-inset ring-ink-700/70 hover:ring-fpl-pink/50 transition">
                            <div>
                                <p className="text-sm font-medium text-fg">{player.web_name}</p>
                                <p className="text-[11px] text-fg-3">{player.team} · {player.position}</p>
                            </div>
                            <div className="text-right tabular">
                                <p className="font-display text-base font-semibold text-fg">{player.ownership_pct}%</p>
                                <p className="text-[11px] text-fg-3">£{player.price}m</p>
                            </div>
                        </div>
                    ))}
                </div>
            </Card>
        </div>
    )
}

export default OwnershipStats
