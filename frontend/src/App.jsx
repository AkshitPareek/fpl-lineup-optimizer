import { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import Pitch from './components/Pitch'
import AnalyticsDashboard from './components/analytics/AnalyticsDashboard'
import PlannerView from './components/planner/PlannerView'
import { chipName, chooseStrategy, failureCopy, infeasibleCopy, isUsableMultiPlan } from './plannerFormat'

const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
axios.defaults.baseURL = API_URL

const TOOLS = [
  { id: 'multi', label: 'Planner' },
  { id: 'single', label: 'Single week' },
  { id: 'dream', label: 'Dream team' },
  { id: 'backtest', label: 'Backtest' },
  { id: 'analytics', label: 'Analytics' },
]

function App() {
  const [lineup, setLineup] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [planProblem, setPlanProblem] = useState(null)
  const [elapsed, setElapsed] = useState(0)

  const [mode, setMode] = useState('multi')
  const [budget, setBudget] = useState(100.0)
  const [gameweeks, setGameweeks] = useState(8)
  const [strategy, setStrategy] = useState('standard')
  const [managerId, setManagerId] = useState('')
  const [freeTransfers, setFreeTransfers] = useState(1)
  const [bankedTransfers, setBankedTransfers] = useState(1)

  const [startGw, setStartGw] = useState(1)
  const [endGw, setEndGw] = useState(5)
  const [backtestResults, setBacktestResults] = useState(null)
  const [backtestProgress, setBacktestProgress] = useState(null)
  const [selectedBacktestGw, setSelectedBacktestGw] = useState(null)
  const [dreamTeam, setDreamTeam] = useState(null)

  const [useRobust, setUseRobust] = useState(false)
  const [selectedChip, setSelectedChip] = useState(null)
  const [chipRecommendations, setChipRecommendations] = useState(null)
  const [managerChips, setManagerChips] = useState(null)

  const [compareData, setCompareData] = useState(null)
  const [selectedHitStrategy, setSelectedHitStrategy] = useState('with_hits')

  const [viewMode, setViewMode] = useState('pitch')
  const [selectedGwIndex, setSelectedGwIndex] = useState(0)
  const resultsRef = useRef(null)
  const wasPlanning = useRef(false)

  useEffect(() => {
    axios.post('/api/chip-recommendations', { manager_id: null, current_squad: [], chips_used: [], horizon: 5 })
      .then(res => setChipRecommendations(res.data))
      .catch(err => console.error('Error fetching chip recommendations', err))
  }, [])

  useEffect(() => {
    if (managerId) {
      axios.get(`/api/manager-chips/${managerId}`)
        .then(res => {
          setManagerChips(res.data)
          if (Number.isInteger(res.data.banked_transfers)) {
            setBankedTransfers(res.data.banked_transfers)
          }
          if (typeof res.data.bank === 'number') {
            setBudget(res.data.bank)
          }
        })
        .catch(err => console.error('Error fetching manager chips', err))

      axios.post('/api/chip-recommendations', { manager_id: parseInt(managerId), current_squad: [], chips_used: [], horizon: parseInt(gameweeks) || 5 })
        .then(res => setChipRecommendations(res.data))
        .catch(err => console.error('Error', err))
    }
  }, [managerId, gameweeks])

  useEffect(() => {
    if (!loading || mode !== 'multi') return undefined
    const started = Date.now()
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [loading, mode])

  useEffect(() => {
    if (mode !== 'multi') return
    const started = loading && !wasPlanning.current
    const finished = !loading && wasPlanning.current
    wasPlanning.current = loading
    if ((started || finished) && window.matchMedia('(max-width: 1023px)').matches) {
      resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [loading, mode])

  const applyStrategy = (data, key) => {
    const plan = key === 'no_hits' ? data?.no_hits : data?.with_hits
    if (!isUsableMultiPlan(plan)) return
    setSelectedHitStrategy(key)
    setLineup(plan)
    setSelectedGwIndex(0)
    setPlanProblem(null)
  }

  const optimize = async () => {
    setLoading(true)
    setError(null)
    setElapsed(0)
    if (mode !== 'multi') {
      setLineup(null)
      setCompareData(null)
    }
    setPlanProblem(null)
    setBacktestResults(null)
    setBacktestProgress(null)
    setSelectedBacktestGw(null)
    setDreamTeam(null)

    try {
      if (mode === 'single') {
        const response = await axios.post('/api/optimize', {
          budget: parseFloat(budget),
          gameweeks: 1,
          strategy: strategy,
          manager_id: managerId ? parseInt(managerId) : null,
          free_transfers: parseInt(freeTransfers)
        })
        setLineup(response.data)
        setLoading(false)
      } else if (mode === 'multi') {
        if (managerId && !/^\d+$/.test(String(managerId).trim())) {
          setPlanProblem(failureCopy('Manager ID needs to be the number from your FPL team page.'))
          setLoading(false)
          return
        }
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
        const choice = chooseStrategy(response.data)
        if (choice.plan) {
          setCompareData(response.data)
          setSelectedHitStrategy(choice.key)
          setLineup(choice.plan)
          setSelectedGwIndex(0)
          setPlanProblem(null)
        } else {
          setPlanProblem(infeasibleCopy(response.data.comparison?.note))
        }
        setLoading(false)
      } else if (mode === 'backtest') {
        setBacktestProgress({ progress: 0, message: 'Initializing...' })

        try {
          const response = await fetch(`${API_URL}/api/backtest/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              start_gw: parseInt(startGw),
              end_gw: parseInt(endGw),
              initial_budget: parseFloat(budget),
              horizon: parseInt(gameweeks)
            })
          })

          if (!response.ok) throw new Error(response.statusText)

          const reader = response.body.getReader()
          const decoder = new TextDecoder()
          let buffer = ''

          while (true) {
            const { value, done } = await reader.read()
            if (done) break

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n\n')
            buffer = lines.pop() || ''

            for (const line of lines) {
              const trimLine = line.trim()
              if (trimLine.startsWith('data: ')) {
                const jsonStr = trimLine.substring(6)
                try {
                  const event = JSON.parse(jsonStr)
                  if (event.result) {
                    setBacktestResults(event.result)
                    setBacktestProgress(null)
                    setLoading(false)
                    if (event.result.weekly_results.length > 0) {
                      const withTransfers = event.result.weekly_results.find(
                        (week) => week.transfers?.in?.length || week.transfers?.out?.length
                      )
                      setSelectedBacktestGw((withTransfers || event.result.weekly_results[0]).gameweek)
                    }
                  } else {
                    setBacktestProgress({
                      progress: event.progress,
                      message: event.message
                    })
                  }
                } catch (parseError) {
                  console.error('JSON parse error', parseError)
                }
              }
            }
          }
        } catch (err) {
          setError('Backtest failed: ' + err.message)
          setLoading(false)
          setBacktestProgress(null)
        }
      } else if (mode === 'dream') {
        const response = await axios.get('/api/dream-team')
        setDreamTeam(response.data)
        setLoading(false)
      }
    } catch (err) {
      const detail = axios.isAxiosError(err) ? (err.response?.data?.detail || err.message) : err.message
      if (mode === 'multi') {
        setPlanProblem(failureCopy(detail))
      } else {
        setError('Failed. ' + detail)
      }
      console.error(err)
      setLoading(false)
    }
  }

  const getCurrentDisplay = () => {
    if (mode === 'backtest') {
      if (!backtestResults || !selectedBacktestGw) return null
      const result = backtestResults.weekly_results.find(r => r.gameweek === selectedBacktestGw)
      if (!result) return null
      return {
        points: result.predicted_points,
        actual: result.actual_points,
        squad: result.squad || [],
        gameweek: result.gameweek,
        transfers: result.transfers || null
      }
    }

    if (!lineup || lineup.gameweek_plans) return null

    return {
      points: lineup.total_expected_points,
      cost: lineup.budget_used,
      squad: lineup.lineup || lineup.squad || [],
      transfers: lineup.transfers,
      bank: lineup.bank
    }
  }

  const currentData = getCurrentDisplay()
  const multiPlan = isUsableMultiPlan(lineup) ? lineup : null
  const horizon = Number(gameweeks) || 8

  return (
    <div className="min-h-screen bg-fpl-purple text-white px-4 py-6 sm:p-8 font-sans">
      <header className="mb-6 text-center">
        <h1 className="text-3xl sm:text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-fpl-green to-fpl-cyan">
          FPL Lineup Optimizer
        </h1>
        <p className="text-gray-400 mt-2 text-sm">Plan transfers, chips, and the starting XI across the next gameweeks.</p>
        <nav className="mt-4 flex flex-wrap justify-center gap-2" aria-label="Tools">
          {TOOLS.map((tool) => (
            <button
              key={tool.id}
              type="button"
              onClick={() => setMode(tool.id)}
              className={`px-3 py-1.5 rounded-full text-sm font-semibold transition ${mode === tool.id
                ? 'bg-fpl-pink text-white'
                : 'bg-gray-800 text-gray-300 hover:text-white'}`}
            >
              {tool.label}
            </button>
          ))}
        </nav>
      </header>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 bg-gray-800 rounded-xl p-5 shadow-lg h-fit space-y-5">
          <h2 className="text-lg font-semibold text-fpl-cyan border-b border-gray-700 pb-2">
            {mode === 'multi' ? 'Plan settings' : TOOLS.find((tool) => tool.id === mode)?.label}
          </h2>

          {mode === 'backtest' ? (
            <div className="space-y-4">
              <div>
                <label className="block text-gray-400 mb-2 text-sm">Gameweek range</label>
                <div className="flex items-center gap-2">
                  <input type="number" value={startGw} onChange={e => setStartGw(e.target.value)} className="w-full bg-gray-700 p-3 rounded-lg text-center outline-none" placeholder="Start" />
                  <span className="text-gray-500">to</span>
                  <input type="number" value={endGw} onChange={e => setEndGw(e.target.value)} className="w-full bg-gray-700 p-3 rounded-lg text-center outline-none" placeholder="End" />
                </div>
              </div>
              <div>
                <label className="block text-gray-400 mb-2 text-sm">Optimizer horizon</label>
                <input type="range" min="3" max="15" value={gameweeks} onChange={(e) => setGameweeks(e.target.value)} className="w-full accent-fpl-pink" />
                <div className="text-right text-xs text-gray-400">{gameweeks} GWs lookahead</div>
              </div>
            </div>
          ) : mode !== 'analytics' && mode !== 'dream' ? (
            <>
              <div>
                <label className="block text-gray-400 mb-2 text-sm" htmlFor="manager-id">Manager ID</label>
                <input id="manager-id" type="text" value={managerId} onChange={(e) => setManagerId(e.target.value)} placeholder="Optional, from your team URL" className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none" />
                {managerChips && (
                  <p className="text-xs text-gray-500 mt-1">
                    Chips left: {(managerChips.available || []).join(', ') || 'none'}
                    {Number.isInteger(managerChips.banked_transfers) ? ` · ${managerChips.banked_transfers} free transfer${managerChips.banked_transfers === 1 ? '' : 's'} for the next deadline` : ''}
                    {managerChips.pending_transfers > 0 ? ` (after the ${managerChips.pending_transfers} already made)` : ''}
                  </p>
                )}
                <p className="text-xs text-gray-500 mt-1">Leave this blank to build a new squad from the budget.</p>
              </div>
              {mode === 'multi' && chipRecommendations?.best_chip && (selectedChip || !multiPlan?.chip_plan) && (
                <button
                  type="button"
                  onClick={() => {
                    const chip = chipRecommendations.best_chip
                    const gw = chip.gw ?? chip.recommended_gameweek
                    if (selectedChip?.chip === chip.chip && selectedChip?.gw === gw) {
                      setSelectedChip(null)
                    } else {
                      setSelectedChip({ chip: chip.chip, gw })
                    }
                  }}
                  className={`w-full text-left text-xs rounded-lg p-3 border ${selectedChip ? 'border-fpl-cyan bg-fpl-cyan/10 text-white' : 'border-gray-600 text-gray-300'}`}
                >
                  {selectedChip
                    ? `Using ${chipName(selectedChip.chip)} in GW${selectedChip.gw}. Click to let the plan choose.`
                    : `Suggested chip: ${chipName(chipRecommendations.best_chip.chip)} in GW${chipRecommendations.best_chip.gw ?? chipRecommendations.best_chip.recommended_gameweek}. Click to force it.`}
                </button>
              )}
              {mode === 'multi' && (
                <div>
                  <label className="block text-gray-400 mb-2 text-sm" htmlFor="horizon">Planning horizon</label>
                  <div className="flex items-center gap-4">
                    <input id="horizon" type="range" min="3" max="15" value={gameweeks} onChange={(e) => setGameweeks(Number(e.target.value))} className="w-full accent-fpl-cyan" />
                    <span className="font-bold text-fpl-cyan w-8">{gameweeks}</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">8 weeks is usually a few seconds. 15 weeks can take about a minute.</p>
                </div>
              )}
              {mode === 'multi' && (
                <div>
                  <label className="block text-gray-400 mb-2 text-sm" htmlFor="banked">Free transfers banked</label>
                  <input id="banked" type="number" min="0" max="5" value={bankedTransfers} onChange={(e) => setBankedTransfers(e.target.value)} className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none" />
                </div>
              )}
              {mode === 'multi' && (
                <div>
                  <label className="block text-gray-400 mb-2 text-sm" htmlFor="strategy">Strategy</label>
                  <select id="strategy" value={strategy} onChange={(e) => setStrategy(e.target.value)} className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none">
                    <option value="standard">Standard</option>
                    <option value="differential">Differential</option>
                    <option value="template">Template</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    {strategy === 'differential' ? 'Boost low-ownership players' :
                      strategy === 'template' ? 'Favor popular picks' :
                        'Maximize expected points'}
                  </p>
                </div>
              )}
              {mode === 'single' && strategy === 'transfers' && (
                <div>
                  <label className="block text-gray-400 mb-2 text-sm">Free transfers</label>
                  <input type="number" min="0" max="5" value={freeTransfers} onChange={(e) => setFreeTransfers(e.target.value)} className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none" />
                </div>
              )}
              {mode === 'single' && (
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
          ) : null}

          {mode !== 'dream' && mode !== 'analytics' && (
            <div>
              <label className="block text-gray-400 mb-2 text-sm" htmlFor="budget">
                {managerId ? 'Bank balance (£m)' : 'Budget (£m)'}
              </label>
              <input id="budget" type="number" value={budget} onChange={(e) => setBudget(e.target.value)} className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none" step="0.1" />
              {managerId && mode === 'multi' && (
                <p className="text-xs text-gray-500 mt-1">Money in the bank. The squad value is added from the manager ID.</p>
              )}
            </div>
          )}

          {mode === 'dream' && (
            <div className="bg-gradient-to-r from-fpl-cyan/10 to-fpl-green/10 p-4 rounded-lg border border-fpl-cyan/30">
              <p className="text-sm text-gray-300 text-center">No budget constraints</p>
              <p className="text-xs text-gray-500 text-center mt-1">The best XI the projections can build</p>
            </div>
          )}

          {mode === 'multi' && (
            <div className="bg-gray-900/50 p-3 rounded-lg border border-gray-700">
              <div className="flex items-center gap-2">
                <input id="robust" type="checkbox" checked={useRobust} onChange={(e) => setUseRobust(e.target.checked)} className="w-4 h-4 accent-fpl-pink" />
                <label htmlFor="robust" className="text-sm font-semibold text-gray-300">Robust optimization</label>
              </div>
              <p className="text-xs text-gray-500 mt-1">Hedges against prediction uncertainty</p>
            </div>
          )}

          {mode !== 'analytics' && (
            <button onClick={optimize} disabled={loading && mode !== 'backtest'} className="w-full bg-gradient-to-r from-fpl-green to-fpl-cyan text-fpl-purple font-bold py-3 rounded-lg hover:opacity-90 transition disabled:opacity-50 cursor-pointer">
              {loading && mode === 'multi' ? 'Planning…' :
                loading && mode === 'backtest' ? 'Simulation running…' :
                  loading ? 'Working…' :
                    mode === 'backtest' ? 'Run simulation' :
                      mode === 'dream' ? 'Show dream team' :
                        mode === 'single' ? 'Optimize week' :
                          'Generate plan'}
            </button>
          )}

          {mode === 'multi' && planProblem && !loading && (
            <p className="text-xs text-red-300">{planProblem.title}. The detail is in the plan.</p>
          )}
          {error && mode !== 'multi' && <p className="text-red-400 text-xs">{error}</p>}
        </div>

        {mode === 'analytics' ? (
          <AnalyticsDashboard />
        ) : mode === 'multi' ? (
          <div className="lg:col-span-3" ref={resultsRef}>
            <PlannerView
              plan={multiPlan}
              compareData={compareData}
              selectedKey={selectedHitStrategy}
              onSelectStrategy={(key) => applyStrategy(compareData, key)}
              selectedIndex={selectedGwIndex}
              onSelectWeek={setSelectedGwIndex}
              viewMode={viewMode}
              onViewMode={setViewMode}
              loading={loading}
              elapsed={elapsed}
              gameweeks={horizon}
              problem={planProblem}
            />
          </div>
        ) : (
          <div className="lg:col-span-3 bg-gray-800 rounded-xl p-4 sm:p-6 shadow-lg min-h-[600px] flex flex-col">
            {mode === 'backtest' ? (
              backtestProgress ? (
                <div className="flex-1 flex flex-col items-center justify-center space-y-6">
                  <div className="w-full max-w-md">
                    <div className="flex justify-between text-sm text-gray-400 mb-2"><span>{backtestProgress.message}</span><span>{backtestProgress.progress}%</span></div>
                    <div className="w-full bg-gray-700 rounded-full h-4 overflow-hidden">
                      <div className="bg-gradient-to-r from-fpl-green to-fpl-cyan h-4 rounded-full transition-all duration-300" style={{ width: `${backtestProgress.progress}%` }}></div>
                    </div>
                    <p className="text-center text-xs text-gray-500 mt-4">Solving each week in turn. This stays on screen until the simulation finishes.</p>
                  </div>
                </div>
              ) : backtestResults ? (
                <div className="space-y-6 flex flex-col h-full">
                  <h3 className="text-2xl font-bold text-white border-b border-gray-700 pb-2">Simulation report (GW {startGw}–{endGw})</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-gray-700/30 p-4 rounded-lg text-center border border-gray-600">
                      <p className="text-xs text-gray-400 uppercase tracking-wider">Total actual</p>
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
                      <p className="text-xs text-gray-400 uppercase tracking-wider">Avg pts/GW</p>
                      <p className="text-3xl font-bold text-white">{backtestResults.metrics.avg_points_per_gw.toFixed(1)}</p>
                    </div>
                  </div>
                  <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-1 overflow-y-auto border border-gray-700 rounded-lg max-h-[640px]">
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
                    </div>
                    <div className="lg:col-span-2 bg-gray-900/50 rounded-lg p-2 flex flex-col">
                      {currentData ? (
                        <>
                          <div className="flex justify-between items-center mb-2 px-2">
                            <h4 className="font-bold text-white">GW {currentData.gameweek} lineup</h4>
                            <div className="space-x-2">
                              <button type="button" onClick={() => setViewMode('pitch')} className={`px-2 py-1 rounded text-xs ${viewMode === 'pitch' ? 'bg-fpl-green text-black' : 'bg-gray-700'}`}>Pitch</button>
                              <button type="button" onClick={() => setViewMode('list')} className={`px-2 py-1 rounded text-xs ${viewMode === 'list' ? 'bg-fpl-green text-black' : 'bg-gray-700'}`}>List</button>
                            </div>
                          </div>
                          {currentData.transfers && (currentData.transfers.in?.length || currentData.transfers.out?.length) ? (
                            <div className="mx-2 mb-2 text-sm text-gray-300 bg-gray-900/50 p-3 rounded border-l-2 border-fpl-cyan">
                              <p>Out: {(currentData.transfers.out || []).map((p) => p.name || p).join(', ') || 'none'}</p>
                              <p>In: {(currentData.transfers.in || []).map((p) => p.name || p).join(', ') || 'none'}</p>
                              {currentData.transfers.hits > 0 && <p>Hit cost: -{currentData.transfers.hits * 4}</p>}
                            </div>
                          ) : (
                            <p className="px-2 pb-2 text-xs text-gray-500">No transfers this gameweek.</p>
                          )}
                          {viewMode === 'pitch' ? <Pitch lineup={currentData.squad} /> : (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                              {currentData.squad.map(p => (
                                <div key={p.id} className="bg-gray-700 p-2 rounded flex justify-between text-xs">
                                  <span>{p.web_name} {p.is_captain ? '(C)' : ''}</span>
                                  <span>{p.expected_points?.toFixed(1)} xP</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </>
                      ) : (
                        <div className="flex items-center justify-center h-full text-gray-500">Select a gameweek</div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-center text-gray-400 px-4">
                  <h3 className="text-xl font-bold text-white">Backtest a horizon</h3>
                  <p className="text-sm mt-2 max-w-md">Replay past gameweeks with the same planner. Set a range, then run the simulation.</p>
                </div>
              )
            ) : mode === 'dream' ? (
              dreamTeam ? (
                <div className="space-y-6">
                  <div className="flex flex-wrap justify-between items-center gap-3 border-b border-gray-700 pb-4">
                    <div>
                      <h3 className="text-2xl font-bold text-white">Dream team GW {dreamTeam.gameweek}</h3>
                      <p className="text-gray-400 text-sm mt-1">Best possible XI with no constraints</p>
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-bold text-fpl-cyan">{dreamTeam.total_expected_points} xP</div>
                      <div className="text-sm text-gray-400">Total cost: £{dreamTeam.total_cost}m</div>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {dreamTeam.dream_team.map(player => (
                      <div
                        key={player.id}
                        className={`p-4 rounded-lg flex justify-between items-center ${player.id === dreamTeam.captain.id
                          ? 'bg-gradient-to-r from-yellow-500/20 to-yellow-600/10 border border-yellow-500'
                          : 'bg-gray-700'}`}
                      >
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-sm">{player.web_name}</span>
                            {player.id === dreamTeam.captain.id && (
                              <span className="bg-yellow-500 text-black text-[9px] font-bold px-1.5 py-0.5 rounded">C</span>
                            )}
                          </div>
                          <div className="text-xs text-gray-400 mt-1">{player.team_name} · {player.position}</div>
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
                <div className="flex-1 flex flex-col items-center justify-center text-center text-gray-400 px-4">
                  <h3 className="text-xl font-bold text-white">Dream team</h3>
                  <p className="text-sm mt-2">The best XI with no budget. Use the button to load it.</p>
                </div>
              )
            ) : loading ? (
              <div className="flex-1 flex flex-col items-center justify-center text-gray-300">
                <div className="h-6 w-6 rounded-full border-2 border-fpl-cyan border-t-transparent animate-spin" />
                <p className="text-sm mt-3">Optimizing this week.</p>
              </div>
            ) : !currentData ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center text-gray-400 px-4">
                <h3 className="text-xl font-bold text-white">One gameweek</h3>
                <p className="text-sm mt-2 max-w-md">Build a single squad from the budget, your current squad, or a transfer suggestion. The multi-week plan stays on the Planner tab.</p>
              </div>
            ) : (
              <>
                <div className="flex justify-between items-start mb-6 gap-3">
                  <div>
                    <h3 className="text-2xl font-bold text-white mb-1">
                      This week
                      <span className="text-fpl-green ml-2">{Number(currentData.points || 0).toFixed(1)} xP</span>
                    </h3>
                    {typeof currentData.bank === 'number' && (
                      <p className="text-xs text-gray-400">£{currentData.bank.toFixed(1)}m in the bank</p>
                    )}
                    {currentData.transfers && (currentData.transfers.in?.length || currentData.transfers.out?.length) ? (
                      <div className="text-sm text-gray-300 bg-gray-900/50 p-3 rounded border-l-2 border-fpl-cyan mt-2">
                        <p>Out: {(currentData.transfers.out || []).map((p) => p.name || p.web_name || p).join(', ') || 'none'}</p>
                        <p>In: {(currentData.transfers.in || []).map((p) => p.name || p.web_name || p).join(', ') || 'none'}</p>
                        {currentData.transfers.cost > 0 && <p>Hit cost: -{currentData.transfers.cost}</p>}
                      </div>
                    ) : (
                      <p className="text-sm text-gray-500 mt-2">No transfers on this squad.</p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <button type="button" onClick={() => setViewMode('pitch')} className={`px-3 py-1 rounded text-xs uppercase font-bold ${viewMode === 'pitch' ? 'bg-fpl-green text-gray-900' : 'bg-gray-700 text-gray-400'}`}>Pitch</button>
                    <button type="button" onClick={() => setViewMode('list')} className={`px-3 py-1 rounded text-xs uppercase font-bold ${viewMode === 'list' ? 'bg-fpl-green text-gray-900' : 'bg-gray-700 text-gray-400'}`}>List</button>
                  </div>
                </div>
                {viewMode === 'pitch' ? <Pitch lineup={currentData.squad} /> : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {currentData.squad.map(player => (
                      <div key={player.id} className={`p-3 rounded flex justify-between items-center ${player.is_starter ? 'bg-gray-700' : 'bg-gray-700/50 border border-dashed border-gray-600'}`}>
                        <div>
                          <div className="font-bold text-sm">
                            {player.web_name}
                            {player.is_captain && <span className="ml-2 bg-yellow-500 text-black text-[9px] font-bold px-1 rounded">C</span>}
                            {!player.is_starter && <span className="ml-2 text-[9px] text-gray-300">Bench</span>}
                          </div>
                          <div className="text-xs text-gray-400 mt-1">{player.team_name} · {player.position}</div>
                        </div>
                        <div className="text-fpl-green font-bold">{player.expected_points?.toFixed(1)}</div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
