import { useState, useEffect } from 'react'
import axios from 'axios'

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

    if (loading) return <div className="text-center p-8 text-gray-500">Loading ownership data...</div>

    return (
        <div className="space-y-6">
            {/* Template Players */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <span>🛡️</span> Template Players (Highly Owned)
                </h3>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead className="text-gray-400 bg-gray-900/50 uppercase text-xs">
                            <tr>
                                <th className="p-3 text-left">Player</th>
                                <th className="p-3 text-right">Ownership</th>
                                <th className="p-3 text-right">EO (Est.)</th>
                                <th className="p-3 text-right">Price</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-700">
                            {template.slice(0, 10).map((player, idx) => (
                                <tr key={idx} className="hover:bg-gray-700/50 transition">
                                    <td className="p-3 font-bold text-white">
                                        {player.web_name} <span className="text-xs font-normal text-gray-500 ml-1">{player.team}</span>
                                    </td>
                                    <td className="p-3 text-right">
                                        <div className="flex items-center justify-end gap-2">
                                            <span className="text-fpl-cyan font-bold">{player.ownership_pct}%</span>
                                            <div className="w-16 bg-gray-700 rounded-full h-1.5">
                                                <div
                                                    className="bg-fpl-cyan h-1.5 rounded-full"
                                                    style={{ width: `${Math.min(player.ownership_pct, 100)}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="p-3 text-right text-gray-300">{player.effective_ownership?.toFixed(1)}%</td>
                                    <td className="p-3 text-right text-fpl-green font-mono">£{player.price}m</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Differentials */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <span>🚀</span> Top Differentials (Low Owned)
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {differentials.slice(0, 6).map((player, idx) => (
                        <div key={idx} className="bg-gray-700/50 p-4 rounded-lg flex justify-between items-center border border-gray-600 hover:border-fpl-pink transition">
                            <div>
                                <p className="font-bold text-white">{player.web_name}</p>
                                <p className="text-xs text-gray-400">{player.team} • {player.position}</p>
                            </div>
                            <div className="text-right">
                                <p className="text-fpl-pink font-bold text-lg">{player.ownership_pct}%</p>
                                <p className="text-xs text-fpl-green">£{player.price}m</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

export default OwnershipStats
