import { useState } from 'react'
import OwnershipStats from './OwnershipStats'
import EVCharts from './EVCharts'
import LiveTracker from './LiveTracker'
import { Segmented } from '../ui'

const TABS = [
    { value: 'live', label: 'Live & market' },
    { value: 'ev', label: 'EV & risk' },
    { value: 'ownership', label: 'Ownership' },
]

function AnalyticsDashboard() {
    const [activeTab, setActiveTab] = useState('live')

    return (
        <div className="space-y-5">
            <Segmented options={TABS} value={activeTab} onChange={setActiveTab} />
            <div key={activeTab} className="animate-fade-in">
                {activeTab === 'live' && <LiveTracker />}
                {activeTab === 'ev' && <EVCharts />}
                {activeTab === 'ownership' && <OwnershipStats />}
            </div>
        </div>
    )
}

export default AnalyticsDashboard
