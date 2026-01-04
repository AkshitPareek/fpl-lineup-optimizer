import { useState, useEffect } from 'react'
import axios from 'axios'
import Pitch from './components/Pitch'

function App() {
  const [data, setData] = useState(null)
  const [lineup, setLineup] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Controls
  const [mode, setMode] = useState("multi") // 'single', 'multi', 'backtest', 'dream'
  const [budget, setBudget] = useState(100.0)
  const [gameweeks, setGameweeks] = useState(5) // Default 5
  const [strategy, setStrategy] = useState("standard")
  const [managerId, setManagerId] = useState("")
  const [freeTransfers, setFreeTransfers] = useState(1)
  const [bankedTransfers, setBankedTransfers] = useState(1)

  // Backtest Controls
  const [startGw, setStartGw] = useState(1)
  const [endGw, setEndGw] = useState(5)
  const [backtestResults, setBacktestResults] = useState(null)
  const [backtestProgress, setBacktestProgress] = useState(null)
  const [selectedBacktestGw, setSelectedBacktestGw] = useState(null) // GW number
  const [dreamTeam, setDreamTeam] = useState(null)

  // Robust Optimization
  const [useRobust, setUseRobust] = useState(false)

  // Chip Activation (for multi-period)
  const [selectedChip, setSelectedChip] = useState(null) // { chip: 'triple_captain', gw: 20 }
  const [chipRecommendations, setChipRecommendations] = useState(null)
  const [managerChips, setManagerChips] = useState(null) // { used: [], available: [] }

  // Strategy Comparison
  const [compareData, setCompareData] = useState(null)
  const [selectedHitStrategy, setSelectedHitStrategy] = useState("with_hits") // 'with_hits' or 'no_hits'

  // Results View
  const [viewMode, setViewMode] = useState("pitch") // pitch of list
  const [selectedGwIndex, setSelectedGwIndex] = useState(0)

  useEffect(() => {
    // Fetch initial data just to verify connection
    axios.get('/api/data')
      .then(res => setData(res.data))
      .catch(err => console.error("Error fetching data", err))

    // Fetch chip recommendations (POST endpoint)
    axios.post('/api/chip-recommendations', { manager_id: null, current_squad: [], chips_used: [] })
      .then(res => {
        setChipRecommendations(res.data.recommendations)
        // Auto-apply best chip if available
        if (res.data.best_chip) {
          setSelectedChip(res.data.best_chip)
        }
      })
      .catch(err => console.error("Error fetching chip recommendations", err))
  }, [])

  // Fetch manager chips when manager ID changes
  useEffect(() => {
    if (managerId) {
      axios.get(`/api/manager-chips/${managerId}`)
        .then(res => setManagerChips(res.data))
        .catch(err => console.error("Error fetching manager chips", err))

      // Also fetch chip recommendations with manager context
      axios.post('/api/chip-recommendations', { manager_id: parseInt(managerId), current_squad: [], chips_used: [] })
        .then(res => {
          setChipRecommendations(res.data.recommendations)
          if (res.data.best_chip) {
            setSelectedChip(res.data.best_chip)
          }
        })
        .catch(err => console.error("Error", err))
    }
  }, [managerId])

  const optimize = async () => {
    setLoading(true)
    setError(null)
    setLineup(null)
    setBacktestResults(null)
    setBacktestProgress(null)
    setSelectedGwIndex(0)
    setSelectedBacktestGw(null)
    setDreamTeam(null)
    setCompareData(null)

    try {
      if (mode === "single") {
        const response = await axios.post('/api/optimize', {
          budget: parseFloat(budget),
          gameweeks: parseInt(gameweeks),
          strategy: strategy,
          manager_id: managerId ? parseInt(managerId) : null,
          free_transfers: parseInt(freeTransfers)
        })
        setLineup(response.data)
        setLoading(false)
      } else if (mode === "multi") {
        // Use compare endpoint to get both with-hits and no-hits strategies
        const response = await axios.post('/api/optimize/compare', {
          budget: parseFloat(budget),
          gameweeks: parseInt(gameweeks),
          manager_id: managerId ? parseInt(managerId) : null,
          banked_transfers: parseInt(bankedTransfers),
          robust: useRobust,
          uncertainty_budget: useRobust ? 0.3 : 0,
          strategy: strategy,
          chip_to_use: selectedChip ? [selectedChip.chip, selectedChip.gw] : null
        })
        setCompareData(response.data)
        // Default to the recommended strategy
        const recommended = response.data.comparison.with_hits.recommended ? 'with_hits' : 'no_hits'
        setSelectedHitStrategy(recommended)
        setLineup(response.data[recommended])
        setLoading(false)
      } else if (mode === "backtest") {
        // Streaming implementation
        setBacktestProgress({ progress: 0, message: "Initializing..." })

        try {
          const response = await fetch('/api/backtest/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              start_gw: parseInt(startGw),
              end_gw: parseInt(endGw),
              initial_budget: parseFloat(budget),
              horizon: parseInt(gameweeks)
            })
          })

          if (!response.ok) throw new Error(response.statusText);

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              const trimLine = line.trim();
              if (trimLine.startsWith('data: ')) {
                const jsonStr = trimLine.substring(6);
                try {
                  const event = JSON.parse(jsonStr);
                  if (event.result) {
                    setBacktestResults(event.result);
                    setBacktestProgress(null);
                    setLoading(false);
                    // Select first GW automatically
                    if (event.result.weekly_results.length > 0) {
                      setSelectedBacktestGw(event.result.weekly_results[0].gameweek);
                    }
                  } else {
                    setBacktestProgress({
                      progress: event.progress,
                      message: event.message
                    });
                  }
                } catch (e) {
                  console.error("JSON parse error", e);
                }
              }
            }
          }
        } catch (err) {
          setError("Backtest failed: " + err.message);
          setLoading(false);
          setBacktestProgress(null);
        }
      } else if (mode === "dream") {
        const response = await axios.get('/api/dream-team')
        setDreamTeam(response.data)
        setLoading(false)
      }
    } catch (err) {
      setError("Failed. " + (axios.isAxiosError(err) ? (err.response?.data?.detail || err.message) : err.message))
      console.error(err)
      setLoading(false)
    }
  }

  // Helper to render current plan/lineup based on mode
  const getCurrentDisplay = () => {
    if (mode === 'backtest') {
      if (!backtestResults || !selectedBacktestGw) return null;
      const result = backtestResults.weekly_results.find(r => r.gameweek === selectedBacktestGw);
      if (!result) return null;

      return {
        points: result.predicted_points,
        actual: result.actual_points,
        squad: result.squad || [],
        gameweek: result.gameweek,
        transfers: null // TODO: pass transfer details if needed
      }
    }

    if (!lineup) return null;

    if (mode === "single") {
      return {
        points: lineup.total_expected_points,
        cost: lineup.budget_used,
        squad: lineup.squad,
        transfers: lineup.transfers
      }
    } else {
      const plan = lineup.gameweek_plans[selectedGwIndex]
      if (!plan) return null;

      const fullSquad = [...plan.starting_xi, ...plan.bench]
      return {
        points: plan.expected_points,
        cost: 0,
        squad: fullSquad,
        transfers: plan.transfers,
        gameweek: plan.gameweek
      }
    }
  }

  const currentData = getCurrentDisplay()

  return (
    <div className="min-h-screen bg-fpl-purple text-white p-8 font-sans">
      <header className="mb-10 text-center">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-fpl-green to-fpl-cyan">
          FPL Lineup Optimizer 2025/26
        </h1>
        <p className="text-gray-400 mt-2">AI-Powered Multi-Period Planner & Backtesting Engine</p>
      </header>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Controls Panel */}
        <div className="lg:col-span-1 bg-gray-800 rounded-xl p-6 shadow-lg h-fit space-y-6">
          <h2 className="text-xl font-semibold text-fpl-cyan border-b border-gray-700 pb-2">Configuration</h2>
          <div className="flex bg-gray-700 rounded-lg p-1">
            <button onClick={() => setMode("multi")} className={`flex-1 py-2 rounded text-sm font-bold transition ${mode === 'multi' ? 'bg-fpl-pink text-white' : 'text-gray-400'}`}>Multi</button>
            <button onClick={() => setMode("single")} className={`flex-1 py-2 rounded text-sm font-bold transition ${mode === 'single' ? 'bg-fpl-green text-gray-900' : 'text-gray-400'}`}>Single</button>
            <button onClick={() => setMode("dream")} className={`flex-1 py-2 rounded text-sm font-bold transition ${mode === 'dream' ? 'bg-fpl-cyan text-gray-900' : 'text-gray-400'}`}>Dream</button>
            <button onClick={() => setMode("backtest")} className={`flex-1 py-2 rounded text-sm font-bold transition ${mode === 'backtest' ? 'bg-yellow-500 text-gray-900' : 'text-gray-400'}`}>Test</button>
          </div>

          {mode === "backtest" ? (
            <div className="space-y-4">
              <div>
                <label className="block text-gray-400 mb-2 text-sm">Gameweek Range</label>
                <div className="flex items-center gap-2">
                  <input type="number" value={startGw} onChange={e => setStartGw(e.target.value)} className="w-full bg-gray-700 p-3 rounded-lg text-center outline-none" placeholder="Start" />
                  <span className="text-gray-500">to</span>
                  <input type="number" value={endGw} onChange={e => setEndGw(e.target.value)} className="w-full bg-gray-700 p-3 rounded-lg text-center outline-none" placeholder="End" />
                </div>
              </div>
              <div>
                <label className="block text-gray-400 mb-2 text-sm">Optimizer Horizon</label>
                <input type="range" min="3" max="5" value={gameweeks} onChange={(e) => setGameweeks(e.target.value)} className="w-full accent-fpl-pink" />
                <div className="text-right text-xs text-gray-400">{gameweeks} GWs lookahead</div>
              </div>
            </div>
          ) : (
            <>
              <div>
                <label className="block text-gray-400 mb-2 text-sm">Manager ID</label>
                <input type="text" value={managerId} onChange={(e) => setManagerId(e.target.value)} placeholder="123456" className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none" />
              </div>
              <div>
                <label className="block text-gray-400 mb-2 text-sm">Planning Horizon (GWs)</label>
                <div className="flex items-center gap-4">
                  <input type="range" min={mode === 'single' ? "1" : "3"} max={mode === 'single' ? "1" : "8"} value={gameweeks} onChange={(e) => setGameweeks(e.target.value)} className="w-full accent-fpl-cyan" />
                  <span className="font-bold text-fpl-cyan w-8">{gameweeks}</span>
                </div>
              </div>
              {mode === "multi" && (
                <div>
                  <label className="block text-gray-400 mb-2 text-sm">Banked Transfers</label>
                  <input type="number" min="0" max="5" value={bankedTransfers} onChange={(e) => setBankedTransfers(e.target.value)} className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none" />
                </div>
              )}
              {mode === "multi" && (
                <div>
                  <label className="block text-gray-400 mb-2 text-sm">Strategy</label>
                  <select value={strategy} onChange={(e) => setStrategy(e.target.value)} className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none">
                    <option value="standard">🎯 Standard - Max Points</option>
                    <option value="differential">🚀 Differential - Low Owned</option>
                    <option value="template">🛡️ Template - Safe Picks</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    {strategy === "differential" ? "Boost low-ownership players (<10%)" :
                      strategy === "template" ? "Favor popular, proven picks" :
                        "Maximize expected points"}
                  </p>
                </div>
              )}
              {mode === "multi" && lineup?.chip_suggestion && (
                <div className="mb-3 p-3 bg-fpl-cyan/10 border border-fpl-cyan/20 rounded-lg">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-lg">💡</span>
                    <span className="text-xs font-bold text-fpl-cyan uppercase tracking-wider">Chip Opportunity</span>
                  </div>
                  <p className="text-sm font-bold text-white">
                    Use {lineup.chip_suggestion.chip.replace('_', ' ').toUpperCase()} in GW{lineup.chip_suggestion.gameweek}
                  </p>
                  <p className="text-[10px] text-gray-400 mt-1">
                    +{lineup.chip_suggestion.estimated_gain} pts • {lineup.chip_suggestion.reason}
                  </p>
                </div>
              )}
              {mode === "single" && (
                <div>
                  <label className="block text-gray-400 mb-2 text-sm">Strategy</label>
                  <select value={strategy} onChange={(e) => setStrategy(e.target.value)} className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none">
                    <option value="standard">Standard (Wildcard)</option>
                    <option value="my_squad">Optimize My Squad</option>
                    <option value="transfers">Suggest Transfers</option>
                  </select>
                </div>
              )}
            </>
          )}

          {mode !== "dream" && (
            <div>
              <label className="block text-gray-400 mb-2 text-sm">
                {managerId ? "Bank Balance (£m)" : "Budget (£m)"}
              </label>
              <input type="number" value={budget} onChange={(e) => setBudget(e.target.value)} className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none" step="0.1" />
              {managerId && (
                <p className="text-xs text-gray-500 mt-1">Your remaining ITB (in the bank)</p>
              )}
            </div>
          )}

          {mode === "dream" && (
            <div className="bg-gradient-to-r from-fpl-cyan/10 to-fpl-green/10 p-4 rounded-lg border border-fpl-cyan/30">
              <p className="text-sm text-gray-300 text-center">🌟 No budget constraints</p>
              <p className="text-xs text-gray-500 text-center mt-1">See the absolute best team possible</p>
            </div>
          )}

          {mode === "multi" && (
            <div className="bg-gray-900/50 p-3 rounded-lg border border-gray-700">
              <div className="flex items-center gap-2 mb-2">
                <input type="checkbox" checked={useRobust} onChange={(e) => setUseRobust(e.target.checked)} className="w-4 h-4 accent-fpl-pink" />
                <label className="text-sm font-semibold text-gray-300">Robust Optimization</label>
              </div>
              <p className="text-xs text-gray-500">Hedges against prediction uncertainty</p>
            </div>
          )}

          <button onClick={optimize} disabled={loading && mode !== 'backtest'} className="w-full bg-gradient-to-r from-fpl-green to-fpl-cyan text-fpl-purple font-bold py-3 rounded-lg hover:opacity-90 transition disabled:opacity-50 mt-4 cursor-pointer">
            {loading && mode !== 'backtest' ? "Crunching Numbers..." :
              mode === 'backtest' && loading ? "Simulation Running..." :
                mode === 'backtest' ? "Run Simulation" :
                  mode === 'dream' ? "Show Dream Team" :
                    "Generate Plan"}
          </button>

          {error && <p className="text-red-400 text-xs text-center mt-2">{error}</p>}
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-3 bg-gray-800 rounded-xl p-6 shadow-lg min-h-[600px] flex flex-col">
          {mode === "backtest" ? (
            backtestProgress ? (
              <div className="flex-1 flex flex-col items-center justify-center space-y-6">
                <div className="w-full max-w-md">
                  <div className="flex justify-between text-sm text-gray-400 mb-2"><span>{backtestProgress.message}</span><span>{backtestProgress.progress}%</span></div>
                  <div className="w-full bg-gray-700 rounded-full h-4 overflow-hidden">
                    <div className="bg-gradient-to-r from-fpl-green to-fpl-cyan h-4 rounded-full transition-all duration-300" style={{ width: `${backtestProgress.progress}%` }}></div>
                  </div>
                  <p className="text-center text-xs text-gray-500 mt-4 animate-pulse">Solving complex optimization (MILP) with rolling horizon...</p>
                </div>
              </div>
            ) : backtestResults ? (
              <div className="space-y-6 animate-fade-in flex flex-col h-full">
                <h3 className="text-2xl font-bold text-white border-b border-gray-700 pb-2">Simulation Report (GW {startGw} - {endGw})</h3>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-gray-700/30 p-4 rounded-lg text-center border border-gray-600">
                    <p className="text-xs text-gray-400 uppercase tracking-wider">Total Actual</p>
                    <p className="text-3xl font-bold text-fpl-green">{backtestResults.metrics.total_actual_points}</p>
                  </div>
                  <div className="bg-gray-700/30 p-4 rounded-lg text-center border border-gray-600">
                    <p className="text-xs text-gray-400 uppercase tracking-wider">Predicted</p>
                    <p className="text-3xl font-bold text-fpl-cyan">{backtestResults.metrics.total_predicted_points.toFixed(0)}</p>
                  </div>
                  <div className="bg-gray-700/30 p-4 rounded-lg text-center border border-gray-600">
                    <p className="text-xs text-gray-400 uppercase tracking-wider">Delta</p>
                    <p className={`text-3xl font-bold ${backtestResults.metrics.prediction_error > 0 ? 'text-red-400' : 'text-green-400'}`}>{backtestResults.metrics.prediction_error > 0 ? '+' : ''}{backtestResults.metrics.prediction_error.toFixed(0)}</p>
                  </div>
                  <div className="bg-gray-700/30 p-4 rounded-lg text-center border border-gray-600">
                    <p className="text-xs text-gray-400 uppercase tracking-wider">Avg Pts/GW</p>
                    <p className="text-3xl font-bold text-white">{backtestResults.metrics.avg_points_per_gw.toFixed(1)}</p>
                  </div>
                </div>

                <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6 overflow-hidden">
                  {/* GW Breakdown Table */}
                  <div className="lg:col-span-1 overflow-y-auto border border-gray-700 rounded-lg">
                    <table className="w-full text-sm text-left text-gray-300">
                      <thead className="text-xs uppercase bg-gray-900 text-gray-400 sticky top-0">
                        <tr>
                          <th className="px-4 py-3">GW</th>
                          <th className="px-4 py-3 text-right">Pts</th>
                          <th className="px-4 py-3 text-right">xP</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-700">
                        {backtestResults.weekly_results.map(res => (
                          <tr
                            key={res.gameweek}
                            onClick={() => setSelectedBacktestGw(res.gameweek)}
                            className={`cursor-pointer hover:bg-gray-700 transition ${selectedBacktestGw === res.gameweek ? 'bg-fpl-green/20 border-l-4 border-fpl-green' : 'bg-gray-800'}`}
                          >
                            <td className="px-4 py-3 font-bold text-white">GW {res.gameweek}</td>
                            <td className="px-4 py-3 text-right text-white font-bold">{res.actual_points}</td>
                            <td className="px-4 py-3 text-right text-gray-400">{res.predicted_points.toFixed(1)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="text-center text-xs text-gray-500 p-2">Click a row to view lineup</p>
                  </div>

                  {/* Lineup Visualization */}
                  <div className="lg:col-span-2 overflow-y-auto bg-gray-900/50 rounded-lg p-2 flex flex-col">
                    {currentData ? (
                      <>
                        <div className="flex justify-between items-center mb-2 px-2">
                          <h4 className="font-bold text-white">GW {currentData.gameweek} Lineup</h4>
                          <div className="space-x-2">
                            <button onClick={() => setViewMode("pitch")} className={`px-2 py-1 rounded text-xs ${viewMode === 'pitch' ? 'bg-fpl-green text-black' : 'bg-gray-700'}`}>Pitch</button>
                            <button onClick={() => setViewMode("list")} className={`px-2 py-1 rounded text-xs ${viewMode === 'list' ? 'bg-fpl-green text-black' : 'bg-gray-700'}`}>List</button>
                          </div>
                        </div>
                        <div className="flex-1 overflow-y-auto">
                          {viewMode === 'pitch' ? (
                            <Pitch lineup={currentData.squad} />
                          ) : (
                            <div className="grid grid-cols-2 gap-2">
                              {currentData.squad.map(p => (
                                <div key={p.id} className="bg-gray-700 p-2 rounded flex justify-between text-xs">
                                  <span>{p.web_name} {p.is_captain ? '(C)' : ''}</span>
                                  <span>{p.expected_points?.toFixed(1)} xP</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </>
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-500">Select a gameweek</div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-gray-500 opacity-50">
                <div className="text-6xl mb-4">🧪</div>
                <p>Simulate performance on historical data</p>
              </div>
            )) : mode === "dream" ? (
              dreamTeam ? (
                <div className="space-y-6 animate-fade-in">
                  <div className="flex justify-between items-center border-b border-gray-700 pb-4">
                    <div>
                      <h3 className="text-2xl font-bold text-white">🌟 Dream Team GW {dreamTeam.gameweek}</h3>
                      <p className="text-gray-400 text-sm mt-1">Best possible XI with no constraints</p>
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-bold text-fpl-cyan">{dreamTeam.total_expected_points} xP</div>
                      <div className="text-sm text-gray-400">Total Cost: £{dreamTeam.total_cost}m</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {dreamTeam.dream_team.map(player => (
                      <div
                        key={player.id}
                        className={`p-4 rounded-lg flex justify-between items-center ${player.id === dreamTeam.captain.id
                          ? 'bg-gradient-to-r from-yellow-500/20 to-yellow-600/10 border border-yellow-500'
                          : 'bg-gray-700'
                          }`}
                      >
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-sm">{player.web_name}</span>
                            {player.id === dreamTeam.captain.id && (
                              <span className="bg-yellow-500 text-black text-[9px] font-bold px-1.5 py-0.5 rounded">C</span>
                            )}
                          </div>
                          <div className="text-xs text-gray-400 mt-1">{player.team_name} • {player.position}</div>
                          <div className="text-xs text-gray-500 mt-1">£{player.now_cost}m</div>
                        </div>
                        <div className="text-right">
                          <div className="text-fpl-green font-bold text-lg">{player.expected_points}</div>
                          <div className="text-xs text-gray-500">xP</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-gray-500 opacity-50">
                  <div className="text-6xl mb-4">🌟</div>
                  <p>Dream Team - Best XI with no budget limits</p>
                  <p className="text-sm mt-2">Click "Show Dream Team" to see the ultimate lineup!</p>
                </div>
              )
            ) : loading ? (
              /* Loading state for non-backtest modes */
              <div className="flex-1 flex flex-col items-center justify-center space-y-6">
                <div className="w-full max-w-md">
                  <div className="flex justify-between text-sm text-gray-400 mb-2">
                    <span>
                      {mode === 'dream' ? 'Finding best players...' :
                        mode === 'multi' ? 'Solving multi-period optimization...' :
                          'Optimizing lineup...'}
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-4 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-fpl-green to-fpl-cyan h-4 rounded-full animate-pulse"
                      style={{ width: '75%' }}
                    ></div>
                  </div>
                  <div className="flex items-center justify-center gap-2 mt-4">
                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-fpl-cyan border-t-transparent"></div>
                    <p className="text-xs text-gray-500">
                      {mode === 'multi' ?
                        `Planning ${gameweeks} gameweeks with MILP solver...` :
                        'Crunching numbers...'}
                    </p>
                  </div>
                </div>
              </div>
            ) : !lineup ? (
              <div className="flex-1 flex flex-col items-center justify-center text-gray-500 opacity-50">
                <div className="text-6xl mb-4">⚽️</div>
                <p>Ready to optimize for 2025/26</p>
              </div>
            ) : (
            <>
              {/* Timeline for Multi-Mode */}
              {mode === "multi" && lineup.gameweek_plans && (
                <div className="flex gap-2 overflow-x-auto pb-4 mb-4 border-b border-gray-700">
                  {lineup.gameweek_plans.map((plan, idx) => (
                    <button
                      key={plan.gameweek}
                      onClick={() => setSelectedGwIndex(idx)}
                      className={`flex-shrink-0 px-4 py-2 rounded-lg text-sm transition ${selectedGwIndex === idx
                        ? 'bg-fpl-pink text-white font-bold'
                        : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                        }`}
                    >
                      <span className="block text-xs opacity-70">GW {plan.gameweek}</span>
                      <span className="block">{plan.expected_points.toFixed(1)} pts</span>
                    </button>
                  ))}
                  <div className="ml-auto flex flex-col items-end justify-center text-right px-4">
                    <span className="text-xs text-gray-400">Total Horizon</span>
                    <span className="text-xl font-bold text-fpl-green">{lineup.total_expected_points.toFixed(0)} pts</span>
                  </div>
                </div>
              )}

              {/* Strategy Comparison Toggle */}
              {mode === "multi" && compareData && (
                <div className="bg-gradient-to-r from-gray-900 to-gray-800 p-4 rounded-xl mb-4 border border-gray-700">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-bold text-gray-300">📊 Strategy Comparison</span>
                    <span className="text-xs text-gray-500">{compareData.strategy} optimization</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {/* With Hits Option */}
                    <button
                      onClick={() => {
                        setSelectedHitStrategy('with_hits')
                        setLineup(compareData.with_hits)
                      }}
                      className={`p-3 rounded-lg transition ${selectedHitStrategy === 'with_hits'
                        ? 'bg-yellow-900/50 border-2 border-yellow-500'
                        : 'bg-gray-800 border border-gray-700 hover:border-gray-600'}`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-bold text-yellow-400">⚡ With Hits</span>
                        {compareData.comparison.with_hits.recommended && (
                          <span className="text-[10px] bg-green-900/50 text-green-400 px-1.5 py-0.5 rounded">Recommended</span>
                        )}
                      </div>
                      <div className="text-lg font-bold text-white">{compareData.comparison.with_hits.net_xp} pts</div>
                      <div className="text-xs text-gray-400">
                        {compareData.comparison.with_hits.total_xp} xP - {compareData.comparison.with_hits.hit_cost} hits
                      </div>
                    </button>

                    {/* No Hits Option */}
                    <button
                      onClick={() => {
                        setSelectedHitStrategy('no_hits')
                        setLineup(compareData.no_hits)
                      }}
                      className={`p-3 rounded-lg transition ${selectedHitStrategy === 'no_hits'
                        ? 'bg-green-900/50 border-2 border-green-500'
                        : 'bg-gray-800 border border-gray-700 hover:border-gray-600'}`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-bold text-green-400">🛡️ No Hits</span>
                        {compareData.comparison.no_hits.recommended && (
                          <span className="text-[10px] bg-green-900/50 text-green-400 px-1.5 py-0.5 rounded">Recommended</span>
                        )}
                      </div>
                      <div className="text-lg font-bold text-white">{compareData.comparison.no_hits.net_xp} pts</div>
                      <div className="text-xs text-gray-400">
                        Free transfers only
                      </div>
                    </button>
                  </div>
                  <div className="mt-2 text-center">
                    <span className={`text-xs ${compareData.comparison.difference > 0 ? 'text-yellow-400' : 'text-green-400'}`}>
                      {compareData.comparison.difference > 0
                        ? `Hits gain +${compareData.comparison.difference} pts`
                        : `No-hit saves ${Math.abs(compareData.comparison.difference)} pts`}
                    </span>
                  </div>
                </div>
              )}

              {/* Current Gw View Header */}
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h3 className="text-2xl font-bold text-white mb-1">
                    GW {currentData.gameweek || 'Selection'}
                    <span className="text-fpl-green ml-2">{currentData.points.toFixed(1)} xP</span>
                  </h3>
                  {currentData.transfers?.explanation ? (
                    <div className="text-sm text-gray-300 max-w-2xl bg-gray-900/50 p-3 rounded border-l-2 border-fpl-cyan mt-2">
                      {/* xP Gain and Hit Analysis */}
                      <div className="flex items-center gap-4 mb-2 pb-2 border-b border-gray-700">
                        {currentData.transfers.xp_gain !== undefined && (
                          <span className={`px-2 py-1 rounded text-xs font-bold ${currentData.transfers.xp_gain >= 0 ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'}`}>
                            {currentData.transfers.xp_gain >= 0 ? '📈' : '📉'} xP Gain: {currentData.transfers.xp_gain > 0 ? '+' : ''}{currentData.transfers.xp_gain?.toFixed(1)}
                          </span>
                        )}
                        {currentData.transfers.hits > 0 && (
                          <span className={`px-2 py-1 rounded text-xs font-bold ${currentData.transfers.hit_worth_it ? 'bg-yellow-900/50 text-yellow-400' : 'bg-red-900/50 text-red-400'}`}>
                            {currentData.transfers.hit_worth_it ? '✅' : '⚠️'} -{currentData.transfers.hits * 4} hit {currentData.transfers.hit_worth_it ? '(Worth it!)' : '(Not recommended)'}
                          </span>
                        )}
                      </div>


                      {/* Structured Explanation Display */}
                      <div className="space-y-3 font-sans text-sm">
                        {currentData.transfers.explanation.split('\n').map((line, i) => {
                          // Skip headers
                          if (line.startsWith('##')) return null;
                          if (line.startsWith('###')) return <div key={i} className="font-bold text-gray-400 mt-2 border-b border-gray-700 pb-1">{line.replace('### ', '')}</div>;

                          // Sell lines
                          if (line.includes('**SELL:**')) {
                            const content = line.replace('**SELL:**', '').trim();
                            return (
                              <div key={i} className="flex items-start gap-2 text-red-300 bg-red-900/10 p-1.5 rounded">
                                <span className="font-bold text-xs bg-red-900/50 text-red-400 px-1.5 rounded">SELL</span>
                                <span>{content}</span>
                              </div>
                            );
                          }

                          // Buy lines 
                          if (line.includes('**BUY:**')) {
                            const content = line.replace('**BUY:**', '').trim();
                            return (
                              <div key={i} className="flex items-start gap-2 text-green-300 bg-green-900/10 p-1.5 rounded">
                                <span className="font-bold text-xs bg-green-900/50 text-green-400 px-1.5 rounded">BUY</span>
                                <span>{content}</span>
                              </div>
                            );
                          }

                          // Detail lines (fixtures, risks, expected)
                          if (line.trim().startsWith('└')) {
                            return <div key={i} className="ml-10 text-xs text-gray-400">{line.trim()}</div>;
                          }

                          // Generic bold lines (like multipliers)
                          if (line.includes('**')) {
                            return <div key={i} className="font-medium text-gray-300">{line.replace(/\*\*/g, '')}</div>;
                          }

                          // Empty strings or other text
                          if (!line.trim()) return null;

                          return <div key={i} className="text-gray-400">{line}</div>;
                        })}
                      </div>

                      {/* Transfer Alternatives */}
                      {currentData.transfers.alternatives?.length > 0 && (
                        <div className="mt-4 pt-3 border-t border-gray-700">
                          <h4 className="text-xs font-bold text-fpl-cyan mb-2 uppercase tracking-wider">🔄 Alternative Options</h4>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                            {currentData.transfers.alternatives.map(alt => (
                              <div key={alt.position_id} className="bg-gray-800/50 rounded p-2">
                                <div className="text-[10px] font-bold text-gray-400 mb-1">{alt.position}</div>
                                <div className="space-y-1">
                                  {alt.options.slice(0, 3).map((option, i) => (
                                    <div key={option.id} className="flex items-center justify-between text-xs">
                                      <span className={`${i === 0 ? 'text-fpl-green font-medium' : 'text-gray-300'}`}>
                                        {i + 1}. {option.name}
                                      </span>
                                      <span className="text-gray-500 text-[10px]">
                                        {option.expected_points.toFixed(1)} | £{option.price.toFixed(1)}m
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No transfers planned.</p>
                  )}
                </div>

                <div className="flex gap-2">
                  <button onClick={() => setViewMode("pitch")} className={`px-3 py-1 rounded text-xs uppercase font-bold ${viewMode === 'pitch' ? 'bg-fpl-green text-gray-900' : 'bg-gray-700 text-gray-400'}`}>Pitch</button>
                  <button onClick={() => setViewMode("list")} className={`px-3 py-1 rounded text-xs uppercase font-bold ${viewMode === 'list' ? 'bg-fpl-green text-gray-900' : 'bg-gray-700 text-gray-400'}`}>List</button>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto">
                {viewMode === "pitch" ? <Pitch lineup={currentData.squad} /> : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {currentData.squad.map(player => (
                      <div key={player.id} className={`p-3 rounded flex justify-between items-center ${player.is_starter ? 'bg-gray-700' : 'bg-gray-700/50 border border-dashed border-gray-600'}`}>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-sm">{player.web_name}</span>
                            {player.is_captain && <span className="bg-yellow-500 text-black text-[9px] font-bold px-1 rounded">C</span>}
                            {player.is_vice_captain && <span className="bg-gray-500 text-white text-[9px] font-bold px-1 rounded">V</span>}
                            {!player.is_starter && <span className="bg-gray-600 text-white text-[9px] font-bold px-1 rounded">BENCH</span>}
                          </div>
                          <div className="text-xs text-gray-400 mt-1">{player.team_name} • {player.position}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-fpl-green font-bold">{player.expected_points?.toFixed(1)}</div>
                          <div className="text-xs text-gray-500">xP</div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div >
  )
}

export default App
