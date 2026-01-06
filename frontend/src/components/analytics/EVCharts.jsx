import { useState, useEffect } from 'react'
import axios from 'axios'

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

    if (loading) return <div className="text-center p-8 text-gray-500">Calculating EV distributions...</div>

    // Helper to render EV range bar
    const EVBar = ({ floor, expected, ceiling, maxVal = 15 }) => {
        const left = (floor / maxVal) * 100
        const width = ((ceiling - floor) / maxVal) * 100
        const expectedPos = ((expected - floor) / (ceiling - floor)) * 100

        return (
            <div className="relative w-full h-8 bg-gray-900 rounded-full mt-2 overflow-hidden border border-gray-700">
                {/* Full Range Bar (Floor to Ceiling) */}
                <div
                    className="absolute h-full bg-gradient-to-r from-gray-600 via-fpl-cyan to-fpl-green opacity-30"
                    style={{ left: `${left}%`, width: `${width}%` }}
                ></div>

                {/* Expected Point Marker */}
                <div
                    className="absolute h-full w-1 bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] z-10"
                    style={{ left: `${(expected / maxVal) * 100}%` }}
                ></div>

                {/* Labels - positioned absolutely based on values */}
                <span className="absolute top-1/2 -translate-y-1/2 text-[9px] text-gray-400" style={{ left: `${left}%`, transform: 'translateX(-110%)' }}>
                    {floor.toFixed(1)}
                </span>
                <span className="absolute top-1/2 -translate-y-1/2 text-[9px] text-white font-bold" style={{ left: `${(expected / maxVal) * 100}%`, transform: 'translateX(4px)' }}>
                    {expected.toFixed(1)}
                </span>
                <span className="absolute top-1/2 -translate-y-1/2 text-[9px] text-fpl-green" style={{ left: `${left + width}%`, transform: 'translateX(4px)' }}>
                    {ceiling.toFixed(1)}
                </span>
            </div>
        )
    }

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* High Ceiling Players */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <span>⚡</span> Explosive Potential (High Ceiling)
                </h3>
                <p className="text-xs text-gray-400 mb-4">Players with the highest 90th percentile outcome. High risk, high reward.</p>

                <div className="space-y-4">
                    {highCeiling.slice(0, 5).map((player, idx) => (
                        <div key={idx} className="bg-gray-700/30 p-3 rounded-lg">
                            <div className="flex justify-between items-center mb-1">
                                <span className="font-bold text-sm text-white">{player.web_name}</span>
                                <span className="text-xs text-fpl-green font-mono">Upside: +{player.upside?.toFixed(1)}</span>
                            </div>
                            <EVBar
                                floor={player.floor}
                                expected={player.expected_points}
                                ceiling={player.ceiling}
                            />
                        </div>
                    ))}
                </div>
            </div>

            {/* Safe Players */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <span>🏰</span> Safe Anchors (High Floor)
                </h3>
                <p className="text-xs text-gray-400 mb-4">Consistent performers with low variance. Good for captain reliability.</p>

                <div className="space-y-4">
                    {safePlayers.slice(0, 5).map((player, idx) => (
                        <div key={idx} className="bg-gray-700/30 p-3 rounded-lg">
                            <div className="flex justify-between items-center mb-1">
                                <span className="font-bold text-sm text-white">{player.web_name}</span>
                                <span className="text-xs text-fpl-cyan font-mono">Risk: {player.risk_score?.toFixed(2)}</span>
                            </div>
                            <EVBar
                                floor={player.floor}
                                expected={player.expected_points}
                                ceiling={player.ceiling}
                            />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

export default EVCharts
