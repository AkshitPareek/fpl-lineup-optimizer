import { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import PlannerView from './components/planner/PlannerView'
import { chooseStrategy, failureCopy, infeasibleCopy, isUsableMultiPlan } from './plannerFormat'

const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
axios.defaults.baseURL = API_URL

function App() {
  const [lineup, setLineup] = useState(null)
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [planProblem, setPlanProblem] = useState(null)

  const [budget, setBudget] = useState(100.0)
  const [gameweeks, setGameweeks] = useState(8)
  const [managerId, setManagerId] = useState('')
  const [bankedTransfers, setBankedTransfers] = useState(1)
  const [managerChips, setManagerChips] = useState(null)

  const [compareData, setCompareData] = useState(null)
  const [selectedHitStrategy, setSelectedHitStrategy] = useState('with_hits')
  const [viewMode, setViewMode] = useState('pitch')
  const [selectedGwIndex, setSelectedGwIndex] = useState(0)
  const resultsRef = useRef(null)
  const wasPlanning = useRef(false)

  useEffect(() => {
    if (!managerId) {
      setManagerChips(null)
      return undefined
    }
    let cancelled = false
    axios.get(`/api/manager-chips/${managerId}`)
      .then(res => {
        if (cancelled) return
        setManagerChips(res.data)
        if (Number.isInteger(res.data.banked_transfers)) {
          setBankedTransfers(res.data.banked_transfers)
        }
        if (typeof res.data.bank === 'number') {
          setBudget(res.data.bank)
        }
      })
      .catch(err => console.error('Error fetching manager chips', err))
    return () => {
      cancelled = true
    }
  }, [managerId])

  useEffect(() => {
    if (!loading) return undefined
    const started = Date.now()
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [loading])

  useEffect(() => {
    const started = loading && !wasPlanning.current
    const finished = !loading && wasPlanning.current
    wasPlanning.current = loading
    if ((started || finished) && window.matchMedia('(max-width: 1023px)').matches) {
      resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [loading])

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
    setElapsed(0)
    setPlanProblem(null)

    if (managerId && !/^\d+$/.test(String(managerId).trim())) {
      setPlanProblem(failureCopy('Manager ID needs to be the number from your FPL team page.'))
      setLoading(false)
      return
    }

    try {
      const response = await axios.post('/api/optimize/compare', {
        budget: parseFloat(budget),
        gameweeks: parseInt(gameweeks),
        manager_id: managerId ? parseInt(managerId) : null,
        banked_transfers: parseInt(bankedTransfers),
        robust: false,
        uncertainty_budget: 0,
        strategy: 'standard',
        chip_to_use: null
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
    } catch (err) {
      const detail = axios.isAxiosError(err) ? (err.response?.data?.detail || err.message) : err.message
      setPlanProblem(failureCopy(detail))
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const multiPlan = isUsableMultiPlan(lineup) ? lineup : null
  const horizon = Number(gameweeks) || 8
  const transfersKnown = Number.isInteger(managerChips?.banked_transfers)

  return (
    <div className="min-h-screen bg-fpl-purple text-white px-4 py-6 sm:p-8 font-sans">
      <header className="mb-6 text-center">
        <h1 className="text-3xl sm:text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-fpl-green to-fpl-cyan">
          FPL Lineup Optimizer
        </h1>
        <p className="text-gray-400 mt-2 text-sm">Plan transfers, chips, and the starting XI across the next gameweeks.</p>
      </header>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 bg-gray-800 rounded-xl p-5 shadow-lg h-fit space-y-5">
          <h2 className="text-lg font-semibold text-fpl-cyan border-b border-gray-700 pb-2">Plan settings</h2>

          <div>
            <label className="block text-gray-400 mb-2 text-sm" htmlFor="manager-id">Manager ID</label>
            <input id="manager-id" type="text" value={managerId} onChange={(e) => setManagerId(e.target.value)} placeholder="Optional, from your team URL" className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none" />
            {managerChips && (
              <p className="text-xs text-gray-500 mt-1">
                Chips left: {(managerChips.available || []).join(', ') || 'none'}
                {transfersKnown ? ` · ${managerChips.banked_transfers} free transfer${managerChips.banked_transfers === 1 ? '' : 's'} for the next deadline` : ''}
                {managerChips.pending_transfers > 0 ? ` (after the ${managerChips.pending_transfers} already made)` : ''}
              </p>
            )}
            <p className="text-xs text-gray-500 mt-1">Leave this blank to build a new squad from the budget.</p>
          </div>

          <div>
            <label className="block text-gray-400 mb-2 text-sm" htmlFor="horizon">Planning horizon</label>
            <div className="flex items-center gap-4">
              <input id="horizon" type="range" min="3" max="15" value={gameweeks} onChange={(e) => setGameweeks(Number(e.target.value))} className="w-full accent-fpl-cyan" />
              <span className="font-bold text-fpl-cyan w-8">{gameweeks}</span>
            </div>
            <p className="text-xs text-gray-500 mt-1">8 weeks is usually a few seconds. 15 weeks can take about a minute.</p>
          </div>

          {!transfersKnown && (
            <div>
              <label className="block text-gray-400 mb-2 text-sm" htmlFor="banked">Free transfers</label>
              <input id="banked" type="number" min="0" max="5" value={bankedTransfers} onChange={(e) => setBankedTransfers(e.target.value)} className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none" />
            </div>
          )}

          <div>
            <label className="block text-gray-400 mb-2 text-sm" htmlFor="budget">
              {managerId ? 'Bank balance (£m)' : 'Budget (£m)'}
            </label>
            <input id="budget" type="number" value={budget} onChange={(e) => setBudget(e.target.value)} className="w-full bg-gray-700 rounded-lg p-3 text-sm outline-none" step="0.1" />
            {managerId && (
              <p className="text-xs text-gray-500 mt-1">Money in the bank. The squad value comes from the manager ID.</p>
            )}
          </div>

          <button onClick={optimize} disabled={loading} className="w-full bg-gradient-to-r from-fpl-green to-fpl-cyan text-fpl-purple font-bold py-3 rounded-lg hover:opacity-90 transition disabled:opacity-50 cursor-pointer">
            {loading ? 'Planning…' : 'Generate plan'}
          </button>

          {planProblem && !loading && (
            <p className="text-xs text-red-300">{planProblem.title}. The detail is in the plan.</p>
          )}
        </div>

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
      </div>
    </div>
  )
}

export default App
