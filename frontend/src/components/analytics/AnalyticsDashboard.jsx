import { useState } from 'react'
import OwnershipStats from './OwnershipStats'
import EVCharts from './EVCharts'
import LiveTracker from './LiveTracker'

function AnalyticsDashboard() {
    const [activeTab, setActiveTab] = useState('live') // 'live', 'ownership', 'ev'

    return (
        <div className="bg-gray-800 rounded-xl p-6 shadow-lg lg:col-span-3 min-h-[600px] flex flex-col">
            {/* Header & Tabs */}
            <div className="flex flex-col md:flex-row justify-between items-center mb-8 border-b border-gray-700 pb-4">
                <div>
                    <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-fpl-pink to-fpl-cyan">
                        Advanced Analytics
                    </h2>
                    <p className="text-gray-400 text-sm mt-1">Data-driven insights to gain an edge</p>
                </div>

                <div className="flex gap-2 mt-4 md:mt-0 bg-gray-900/50 p-1 rounded-lg">
                    <button
                        onClick={() => setActiveTab('live')}
                        className={`px-4 py-2 rounded-md text-sm font-bold transition ${activeTab === 'live' ? 'bg-fpl-green text-gray-900 shadow-lg' : 'text-gray-400 hover:text-white'}`}
                    >
                        Live Tracker
                    </button>
                    <button
                        onClick={() => setActiveTab('ev')}
                        className={`px-4 py-2 rounded-md text-sm font-bold transition ${activeTab === 'ev' ? 'bg-fpl-cyan text-gray-900 shadow-lg' : 'text-gray-400 hover:text-white'}`}
                    >
                        EV & Risk
                    </button>
                    <button
                        onClick={() => setActiveTab('ownership')}
                        className={`px-4 py-2 rounded-md text-sm font-bold transition ${activeTab === 'ownership' ? 'bg-fpl-pink text-white shadow-lg' : 'text-gray-400 hover:text-white'}`}
                    >
                        Ownership
                    </button>
                </div>
            </div>

            {/* Content Area */}
            <div className="flex-1 animate-fade-in">
                {activeTab === 'live' && <LiveTracker />}
                {activeTab === 'ev' && <EVCharts />}
                {activeTab === 'ownership' && <OwnershipStats />}
            </div>
        </div>
    )
}

export default AnalyticsDashboard
