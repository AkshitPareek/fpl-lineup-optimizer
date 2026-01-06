import { useState, useEffect } from 'react'
import axios from 'axios'

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

    if (loading) return <div className="text-center p-8 text-gray-500">Connecting to FPL Live API...</div>

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Live GW Summary */}
            <div className="lg:col-span-2 bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xl font-bold text-white flex items-center gap-2">
                        <span className="animate-pulse text-red-500">●</span> Live Gameweek {summary?.gameweek}
                    </h3>
                    <span className="text-xs bg-gray-700 px-2 py-1 rounded text-gray-300">Updated: Just now</span>
                </div>

                {/* Top Scorers */}
                <h4 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-3">Top Performers</h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 mb-8">
                    {summary?.top_scorers?.slice(0, 6).map((player, idx) => (
                        <div key={idx} className="bg-gray-700/50 p-3 rounded-lg flex justify-between items-center border border-gray-700">
                            <div className="flex items-center gap-2">
                                <div className="w-6 h-6 rounded-full bg-gray-600 flex items-center justify-center text-xs font-bold">
                                    {idx + 1}
                                </div>
                                <div>
                                    <p className="font-bold text-sm text-white">{player.web_name}</p>
                                    <p className="text-[10px] text-gray-400">{player.team}</p>
                                </div>
                            </div>
                            <div className="text-fpl-green font-bold text-lg">{player.event_points}</div>
                        </div>
                    ))}
                </div>

                {/* Transfer Trends */}
                <h4 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-3">Transfer Trends</h4>
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-green-900/10 p-3 rounded border border-green-900/30">
                        <p className="text-xs text-green-400 font-bold mb-2">Most Transferred In</p>
                        {summary?.most_transferred_in?.slice(0, 3).map((p, i) => (
                            <div key={i} className="flex justify-between text-xs py-1">
                                <span>{p.web_name}</span>
                                <span className="text-green-400">+{p.transfers_in.toLocaleString()}</span>
                            </div>
                        ))}
                    </div>
                    <div className="bg-red-900/10 p-3 rounded border border-red-900/30">
                        <p className="text-xs text-red-400 font-bold mb-2">Most Transferred Out</p>
                        {summary?.most_transferred_out?.slice(0, 3).map((p, i) => (
                            <div key={i} className="flex justify-between text-xs py-1">
                                <span>{p.web_name}</span>
                                <span className="text-red-400">-{p.transfers_out.toLocaleString()}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Market Watch (Price Changes) */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                    <span>💷</span> Market Watch
                </h3>

                <div className="space-y-6">
                    <div>
                        <h4 className="text-xs font-bold text-green-400 uppercase mb-2 flex items-center gap-1">
                            <span>▲</span> Likely Risers
                        </h4>
                        {risers.length > 0 ? (
                            <div className="space-y-2">
                                {risers.slice(0, 5).map((p, i) => (
                                    <div key={i} className="flex justify-between items-center text-sm bg-gray-700/30 p-2 rounded">
                                        <span>{p.web_name}</span>
                                        <div className="text-right">
                                            <div className="text-xs text-gray-400">£{p.current_price}m</div>
                                            <div className="text-[10px] text-green-400">+{p.net_transfers.toLocaleString()} net</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-xs text-gray-500 italic">No imminent price rises detected</p>
                        )}
                    </div>

                    <div>
                        <h4 className="text-xs font-bold text-red-400 uppercase mb-2 flex items-center gap-1">
                            <span>▼</span> Likely Fallers
                        </h4>
                        {fallers.length > 0 ? (
                            <div className="space-y-2">
                                {fallers.slice(0, 5).map((p, i) => (
                                    <div key={i} className="flex justify-between items-center text-sm bg-gray-700/30 p-2 rounded">
                                        <span>{p.web_name}</span>
                                        <div className="text-right">
                                            <div className="text-xs text-gray-400">£{p.current_price}m</div>
                                            <div className="text-[10px] text-red-400">{p.net_transfers.toLocaleString()} net</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-xs text-gray-500 italic">No imminent price falls detected</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default LiveTracker
