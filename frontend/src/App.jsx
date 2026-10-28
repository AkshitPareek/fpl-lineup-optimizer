import { useEffect, useMemo, useState } from 'react'
import { api } from './lib/api'
import { cx } from './lib/cx'
import { IconBolt, IconChart, IconHistory, IconRoute, IconStar } from './components/icons'
import PlannerView from './views/PlannerView'
import QuickPickView from './views/QuickPickView'
import DreamTeamView from './views/DreamTeamView'
import BacktestView from './views/BacktestView'
import AnalyticsDashboard from './components/analytics/AnalyticsDashboard'

const TABS = [
  { id: 'planner', label: 'Planner', icon: IconRoute, blurb: 'The team to set for the next deadline' },
  { id: 'quick', label: 'Quick Pick', icon: IconBolt, blurb: 'This gameweek only, on the same projections' },
  { id: 'dream', label: 'Dream Team', icon: IconStar, blurb: 'Best XI, no budget' },
  { id: 'backtest', label: 'Backtest', icon: IconHistory, blurb: 'Replay past gameweeks' },
  { id: 'analytics', label: 'Analytics', icon: IconChart, blurb: 'Market, EV & ownership' },
]

function tabFromHash() {
  const id = window.location.hash.replace(/^#\/?/, '')
  return TABS.some(t => t.id === id) ? id : 'planner'
}

function useDebounced(value, delay) {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(t)
  }, [value, delay])
  return debounced
}

const MANAGER_KEY = 'fpl-manager-id'

function App() {
  const [data, setData] = useState(null)
  const [dataError, setDataError] = useState(false)
  const [tab, setTab] = useState(tabFromHash)
  // Views mount on first visit and then stay mounted so results survive tab switches.
  const [visited, setVisited] = useState(() => new Set([tabFromHash()]))

  // Settings shared by the Planner and Quick Pick views
  const [managerId, setManagerId] = useState(() => localStorage.getItem(MANAGER_KEY) || '')
  const [budget, setBudget] = useState(100.0)
  const [bankedTransfers, setBankedTransfers] = useState(1)
  const [loadedChips, setLoadedChips] = useState(null)
  const debouncedManagerId = useDebounced(managerId.trim(), 500)
  const validManagerId = /^\d+$/.test(debouncedManagerId) ? debouncedManagerId : ''
  const managerChips = loadedChips?.id === validManagerId ? loadedChips.data : null

  const setManagerIdValue = value => {
    setManagerId(value)
    if (!String(value).trim()) setBudget(100)
  }

  useEffect(() => {
    const id = managerId.trim()
    if (/^\d+$/.test(id)) localStorage.setItem(MANAGER_KEY, id)
    else if (!id) localStorage.removeItem(MANAGER_KEY)
  }, [managerId])

  useEffect(() => {
    if (!validManagerId) return
    let cancelled = false
    api.get(`/api/manager-chips/${validManagerId}`)
      .then(res => {
        if (cancelled) return
        setLoadedChips({ id: validManagerId, data: res.data })
        if (Number.isInteger(res.data.banked_transfers)) setBankedTransfers(res.data.banked_transfers)
        if (typeof res.data.bank === 'number') setBudget(res.data.bank)
      })
      .catch(err => {
        console.error('Error fetching manager chips', err)
        if (!cancelled) setLoadedChips({ id: validManagerId, data: { error: true } })
      })
    return () => { cancelled = true }
  }, [validManagerId])

  useEffect(() => {
    api.get('/api/data')
      .then(res => setData(res.data))
      .catch(err => {
        console.error('Error fetching data', err)
        setDataError(true)
      })
  }, [])

  useEffect(() => {
    const onHash = () => {
      const next = tabFromHash()
      setTab(next)
      setVisited(prev => (prev.has(next) ? prev : new Set(prev).add(next)))
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const teamCodes = useMemo(() => {
    const map = {}
    for (const t of data?.static?.teams || []) map[t.id] = t.code
    return map
  }, [data])

  const season = data?.season
  const seasonLabel = season?.label || '2026/27'
  const shared = {
    season, managerId, setManagerId: setManagerIdValue, validManagerId, budget, setBudget, teamCodes,
    managerChips, bankedTransfers, setBankedTransfers,
  }

  const views = {
    planner: <PlannerView {...shared} />,
    quick: <QuickPickView {...shared} />,
    dream: <DreamTeamView {...shared} />,
    backtest: <BacktestView {...shared} />,
    analytics: <AnalyticsDashboard />,
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-ink-700/70 bg-ink-950/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1440px] items-center gap-4 px-4 sm:px-6 h-16">
          <a href="#/planner" className="flex items-center gap-2.5 shrink-0">
            <img src="/favicon.svg" alt="" className="h-8 w-8" />
            <div className="leading-tight">
              <div className="font-display text-[15px] font-semibold tracking-tight">FPL Optimizer</div>
              <div className="text-[11px] text-fg-3">Season {seasonLabel}</div>
            </div>
          </a>

          <nav className="hidden md:flex items-center gap-1 mx-auto rounded-xl bg-ink-900/80 p-1 ring-1 ring-inset ring-ink-700/70">
            {TABS.map(t => (
              <a
                key={t.id}
                href={`#/${t.id}`}
                className={cx(
                  'flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition',
                  tab === t.id ? 'bg-ink-700 text-fg' : 'text-fg-3 hover:text-fg-2',
                )}
              >
                <t.icon size={16} className={tab === t.id ? 'text-fpl-green' : ''} />
                {t.label}
              </a>
            ))}
          </nav>

          <SeasonStatus season={season} loaded={data != null} error={dataError} />
        </div>

        {/* Mobile nav */}
        <nav className="md:hidden flex gap-1 overflow-x-auto scrollbar-thin px-4 pb-3">
          {TABS.map(t => (
            <a
              key={t.id}
              href={`#/${t.id}`}
              className={cx(
                'flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] font-medium',
                tab === t.id ? 'bg-ink-700 text-fg' : 'text-fg-3',
              )}
            >
              <t.icon size={15} className={tab === t.id ? 'text-fpl-green' : ''} />
              {t.label}
            </a>
          ))}
        </nav>
      </header>

      <main className="mx-auto max-w-[1440px] px-4 sm:px-6 py-6 sm:py-8">
        {TABS.map(t => visited.has(t.id) && (
          <section key={t.id} hidden={tab !== t.id} aria-label={t.label}>
            <div className="mb-6">
              <h1 className="font-display text-2xl sm:text-[28px] font-semibold tracking-tight">{t.label}</h1>
              <p className="text-sm text-fg-3 mt-1">{t.blurb}</p>
            </div>
            {views[t.id]}
          </section>
        ))}
      </main>

      <footer className="mx-auto max-w-[1440px] px-4 sm:px-6 pb-10 pt-4 text-[11px] text-fg-3">
        Projections are model estimates, not guarantees. Data from the official FPL API.
      </footer>
    </div>
  )
}

function SeasonStatus({ season, loaded, error }) {
  if (error) {
    return (
      <div className="ml-auto md:ml-0 flex items-center gap-2 rounded-full bg-bad/10 px-3 py-1 text-xs text-[#ffb3ba] ring-1 ring-inset ring-bad/30">
        <span className="h-1.5 w-1.5 rounded-full bg-bad" /> API offline
      </div>
    )
  }
  if (!loaded) {
    return <div className="ml-auto md:ml-0 h-7 w-32 rounded-full bg-ink-800 animate-pulse" />
  }
  // Older backends don't send season metadata; keep the header slot empty.
  if (!season) return <div className="ml-auto md:ml-0 w-32" />
  return (
    <div className="ml-auto md:ml-0 flex items-center gap-3 text-xs shrink-0">
      {season.active_gameweek && (
        <div className="flex items-center gap-2 rounded-full bg-ink-900 px-3 py-1 ring-1 ring-inset ring-ink-700">
          <span className="h-1.5 w-1.5 rounded-full bg-fpl-green shadow-[0_0_8px] shadow-fpl-green" />
          <span className="text-fg-2">Planning</span>
          <span className="font-semibold text-fg tabular">GW{season.active_gameweek}</span>
        </div>
      )}
      {season.chip_half && (
        <div className="hidden lg:block text-fg-3" title={`Chips reset after GW${season.chip_half_split_gw || 19}`}>
          Chip set {season.chip_half} · resets after GW{season.chip_half_split_gw || 19}
        </div>
      )}
    </div>
  )
}

export default App
