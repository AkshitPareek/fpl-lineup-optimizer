import { useState, useEffect } from 'react'
import axios from 'axios'
import Pitch from './components/Pitch'

function App() {
  const [data, setData] = useState(null)
  const [lineup, setLineup] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Controls
  const [budget, setBudget] = useState(100.0)
  const [gameweeks, setGameweeks] = useState(1)
  const [strategy, setStrategy] = useState("standard")
  const [managerId, setManagerId] = useState("")
  const [freeTransfers, setFreeTransfers] = useState(1)
  const [viewMode, setViewMode] = useState("pitch") // pitch or list

  useEffect(() => {
    // Fetch initial data just to verify connection
    axios.get('/api/data')
      .then(res => setData(res.data))
      .catch(err => console.error("Error fetching data", err))
  }, [])

  const optimize = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.post('/api/optimize', {
        budget: parseFloat(budget),
        gameweeks: parseInt(gameweeks),
        strategy: strategy,
        manager_id: managerId ? parseInt(managerId) : null,
        free_transfers: parseInt(freeTransfers)
      })
      setLineup(response.data)
    } catch (err) {
      setError("Failed to generate lineup. Please try again.")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-fpl-purple text-white p-8">
      <header className="mb-10 text-center">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-fpl-green to-fpl-cyan">
          FPL Lineup Optimizer
        </h1>
        <p className="text-gray-400 mt-2">AI-Powered Fantasy Premier League Assistant</p>
      </header>

      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Controls Panel */}
        <div className="bg-gray-800 rounded-xl p-6 shadow-lg h-fit">
          <h2 className="text-2xl font-semibold mb-6 text-fpl-cyan">Configuration</h2>

          <div className="space-y-6">
            <div>
              <label className="block text-gray-400 mb-2">Manager ID (Optional)</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={managerId}
                  onChange={(e) => setManagerId(e.target.value)}
                  placeholder="e.g. 123456"
                  className="w-full bg-gray-700 rounded-lg p-3 focus:ring-2 focus:ring-fpl-green outline-none"
                />
                <a href="https://fantasy.premierleague.com/" target="_blank" rel="noreferrer" className="bg-gray-700 p-3 rounded-lg text-gray-400 hover:text-white" title="Find ID in URL after login">
                  ?
                </a>
              </div>
              <p className="text-xs text-gray-500 mt-1">Found in your browser URL: .../entry/<b>12345</b>/history</p>
            </div>

            <div>
              <label className="block text-gray-400 mb-2">Budget (£m)</label>
              <input
                type="number"
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
                className="w-full bg-gray-700 rounded-lg p-3 focus:ring-2 focus:ring-fpl-green outline-none"
                step="0.1"
              />
            </div>

            <div>
              <label className="block text-gray-400 mb-2">Gameweeks to Look Ahead</label>
              <input
                type="range"
                min="1"
                max="5"
                value={gameweeks}
                onChange={(e) => setGameweeks(e.target.value)}
                className="w-full accent-fpl-pink"
              />
              <div className="text-right text-sm text-fpl-pink font-bold">{gameweeks} GWs</div>
            </div>

            <div>
              <label className="block text-gray-400 mb-2">Strategy</label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full bg-gray-700 rounded-lg p-3 outline-none"
              >
                <option value="standard">Standard (Maximize Points)</option>
                <option value="safe">Safe (High Ownership)</option>
                <option value="differential">Differential (Low Ownership)</option>
                <option value="my_squad">Optimize My Squad (Best XI)</option>
                <option value="transfers">Suggest Transfers</option>
              </select>
            </div>

            {strategy === "transfers" && (
              <div>
                <label className="block text-gray-400 mb-2">Free Transfers Available</label>
                <input
                  type="number"
                  min="1"
                  max="5"
                  value={freeTransfers}
                  onChange={(e) => setFreeTransfers(e.target.value)}
                  className="w-full bg-gray-700 rounded-lg p-3 outline-none"
                />
              </div>
            )}

            <div>
              <label className="block text-gray-400 mb-2">Budget (£m)</label>
              <input
                type="number"
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
                className="w-full bg-gray-700 rounded-lg p-3 focus:ring-2 focus:ring-fpl-green outline-none"
                step="0.1"
              />
            </div>

            <button
              onClick={optimize}
              disabled={loading}
              className="w-full bg-gradient-to-r from-fpl-green to-fpl-cyan text-fpl-purple font-bold py-4 rounded-lg hover:opacity-90 transition disabled:opacity-50"
            >
              {loading ? "Optimizing..." : "Generate Dream Team"}
            </button>

            {error && <p className="text-red-500 text-sm text-center">{error}</p>}
          </div>
        </div>

        {/* Pitch / Results Panel */}
        <div className="lg:col-span-2 bg-gray-800 rounded-xl p-6 shadow-lg min-h-[600px] relative overflow-hidden">
          {!lineup ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
              <p>Configure your preferences and hit generate!</p>
            </div>
          ) : (
            <div>
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h3 className="text-xl font-bold">Projected Points: <span className="text-fpl-green">{lineup.total_expected_points.toFixed(1)}</span></h3>
                  <p className="text-sm text-gray-400">Budget Used: £{lineup.budget_used.toFixed(1)}m</p>
                </div>

                <div className="flex gap-4 items-center">
                  <div className="px-3 py-1 bg-gray-700 rounded text-xs uppercase tracking-wider">
                    {lineup.status}
                  </div>

                  {/* View Toggle */}
                  <div className="bg-gray-700 rounded-lg p-1 flex text-sm">
                    <button
                      onClick={() => setViewMode("pitch")}
                      className={`px-3 py-1 rounded ${viewMode === 'pitch' ? 'bg-fpl-green text-gray-900 font-bold shadow' : 'text-gray-400 hover:text-white'}`}
                    >
                      Pitch
                    </button>
                    <button
                      onClick={() => setViewMode("list")}
                      className={`px-3 py-1 rounded ${viewMode === 'list' ? 'bg-fpl-green text-gray-900 font-bold shadow' : 'text-gray-400 hover:text-white'}`}
                    >
                      List
                    </button>
                  </div>
                </div>
              </div>

              {/* Transfer Suggestions Panel */}
              {strategy === "transfers" && lineup.transfers && (
                <div className="mb-6 grid grid-cols-2 gap-4">
                  <div className="bg-red-900/30 p-4 rounded border border-red-500/30">
                    <h4 className="text-red-400 font-bold mb-2">Transfers OUT</h4>
                    <ul className="list-disc list-inside text-sm text-gray-300">
                      {lineup.transfers.out.map(p => <li key={p}>{p}</li>)}
                    </ul>
                  </div>
                  <div className="bg-green-900/30 p-4 rounded border border-green-500/30">
                    <h4 className="text-green-400 font-bold mb-2">Transfers IN</h4>
                    <ul className="list-disc list-inside text-sm text-gray-300">
                      {lineup.transfers.in.map(p => <li key={p}>{p}</li>)}
                    </ul>
                  </div>

                  {/* Financials Row */}
                  <div className="col-span-2 bg-blue-900/20 p-3 rounded border border-blue-500/30 flex justify-between items-center text-sm">
                    <div className="flex gap-4">
                      <span className="text-gray-400">Money In: <span className="text-green-400 font-bold">£{lineup.transfers.financials?.money_in}m</span></span>
                      <span className="text-gray-400">Money Out: <span className="text-red-400 font-bold">£{lineup.transfers.financials?.money_out}m</span></span>
                      <span className="text-gray-400">Net: <span className={`${lineup.transfers.financials?.net >= 0 ? 'text-green-400' : 'text-red-400'} font-bold`}>
                        {lineup.transfers.financials?.net > 0 ? '+' : ''}£{lineup.transfers.financials?.net}m
                      </span></span>
                    </div>
                    <div>
                      <span className="text-gray-300">Bank: <span className="text-white font-bold">£{lineup.bank}m</span></span>
                    </div>
                  </div>

                  <div className="col-span-2 text-center text-sm text-gray-400">
                    Transfer Cost: <span className="text-red-500 font-bold">-{lineup.transfers.cost} pts</span>
                  </div>
                </div>
              )}

              {viewMode === "pitch" ? (
                <Pitch lineup={lineup.lineup} />
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {lineup.lineup.map(player => (
                    <div key={player.id} className={`bg-gray-700 p-3 rounded flex justify-between items-center ${player.is_starter ? '' : 'opacity-60 border border-dashed border-gray-600'}`}>
                      <div>
                        <p className="font-bold flex items-center gap-2">
                          <span className="text-white">{player.web_name}</span>
                          {!player.is_starter && <span className="text-[10px] bg-gray-500 px-1 rounded text-white">BENCH</span>}
                        </p>
                        <p className="text-xs text-gray-400">{player.team_name} | {player.position}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-fpl-green font-bold">{player.expected_points.toFixed(1)} pts</p>
                        <p className="text-xs text-gray-400">£{player.now_cost / 10}m</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
