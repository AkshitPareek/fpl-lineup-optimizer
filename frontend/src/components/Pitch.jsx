import React from 'react'

const Pitch = ({ lineup }) => {
    if (!lineup) return null

    // Separate starters and bench
    const starters = lineup.filter(p => p.is_starter)
    const bench = lineup.filter(p => !p.is_starter)

    // Group starters by position
    const gkp = starters.filter(p => p.element_type === 1)
    const def = starters.filter(p => p.element_type === 2)
    const mid = starters.filter(p => p.element_type === 3)
    const fwd = starters.filter(p => p.element_type === 4)

    const getShirtUrl = (player) => {
        const isGK = player.element_type === 1
        const code = player.team_code
        return `https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_${code}${isGK ? '_1' : ''}-66.png`
    }

    const PlayerCard = ({ player, isBench = false, index = 0, total = 1 }) => {
        // Arch logic:
        // If not bench, and row has > 2 players, add some vertical offset.
        // We want the center players to be slightly lower (or higher) than outer.
        // Standard back 4: CBs slightly deeper (lower Y)? or flat?
        // User wants "not a line". FPL view has FBs slightly higher than CBs usually.
        // Let's make outer indices higher (smaller Y, or negative translate).

        let translateY = 0
        if (!isBench && total > 2) {
            const midIndex = (total - 1) / 2
            const distFromMid = Math.abs(index - midIndex)
            // Outer players (high distance) -> Move UP (negative translate)
            // Center players -> Standard
            // Magnitude: 15px per distance unit?
            translateY = -(distFromMid * 15)
        }

        return (
            <div
                className={`flex flex-col items-center justify-start ${isBench ? 'w-20' : 'w-24'} transition-transform hover:scale-110 cursor-pointer group relative`}
                style={{
                    transform: `translateY(${translateY}px)`
                }}
            >

                {/* Shirt Image */}
                <div className="relative mb-1">
                    <img
                        src={getShirtUrl(player)}
                        alt={player.team_name}
                        className={`drop-shadow-lg ${isBench ? "w-10" : "w-12"} object-contain`}
                        onError={(e) => {
                            e.target.style.display = 'none';
                            e.target.nextSibling.style.display = 'block'
                        }}
                    />
                    {/* Fallback SVG if image broken */}
                    <svg width={isBench ? "32" : "40"} height={isBench ? "32" : "40"} viewBox="0 0 64 64" fill="none" className="hidden" xmlns="http://www.w3.org/2000/svg">
                        <path d="M16 12 L8 24 L16 32 L20 28 L20 56 L44 56 L44 28 L48 32 L56 24 L48 12 H16Z" fill={player.element_type === 1 ? "#fbbf24" : "#ef4444"} stroke="white" strokeWidth="2" />
                    </svg>
                </div>

                {/* Info Box */}
                <div className="flex flex-col items-center bg-[#38003c]/90 text-white rounded w-full overflow-hidden border border-gray-600 shadow-xl z-20">
                    <div className={`w-full text-center ${isBench ? 'text-[9px] py-0.5' : 'text-[10px] py-1'} font-bold bg-[#00ff87] text-[#38003c] truncate px-1`}>
                        {player.web_name}
                    </div>
                    {!isBench && (
                        <div className="w-full text-center bg-[#38003c] py-0.5 border-t border-gray-600 flex justify-center items-center gap-1">
                            <span className="text-[10px] font-bold text-gray-200">{player.expected_points}</span>
                            <span className="text-[8px] text-gray-400">XP</span>
                        </div>
                    )}
                </div>
            </div>
        )
    }

    return (
        <div className="w-full select-none">
            {/* Pitch Area */}
            <div className="relative w-full h-[700px] border-4 border-white/40 rounded-xl overflow-hidden shadow-2xl mx-auto bg-green-600">

                {/* Realistic Pitch Background */}
                <div className="absolute inset-0">
                    {/* Grass Stripes Pattern */}
                    <div className="w-full h-full" style={{
                        backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(0,0,0,0.1) 40px, rgba(0,0,0,0.1) 80px)'
                    }}></div>

                    {/* Pitch Markings */}
                    {/* Center Circle */}
                    <div className="absolute top-1/2 left-1/2 w-32 h-32 border-2 border-white/60 rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
                    <div className="absolute top-1/2 left-0 w-full h-0.5 bg-white/60 transform -translate-y-1/2"></div>

                    {/* Penalty Boxes */}
                    <div className="absolute top-0 left-1/2 w-64 h-32 border-2 border-t-0 border-white/60 transform -translate-x-1/2 bg-white/5"></div>
                    <div className="absolute top-0 left-1/2 w-24 h-12 border-2 border-t-0 border-white/60 transform -translate-x-1/2"></div>

                    <div className="absolute bottom-0 left-1/2 w-64 h-32 border-2 border-b-0 border-white/60 transform -translate-x-1/2 bg-white/5"></div>
                    <div className="absolute bottom-0 left-1/2 w-24 h-12 border-2 border-b-0 border-white/60 transform -translate-x-1/2"></div>
                </div>

                {/* Players Layout */}
                <div className="relative z-10 w-full h-full flex flex-col justify-between py-6">

                    {/* GKP */}
                    <div className="flex justify-center items-center h-1/5 pt-4">
                        {gkp.map((p, i) => <PlayerCard key={p.id} player={p} index={i} total={gkp.length} />)}
                    </div>

                    {/* DEF */}
                    <div className="flex justify-center gap-4 sm:gap-12 items-center h-1/5 px-4">
                        {def.map((p, i) => <PlayerCard key={p.id} player={p} index={i} total={def.length} />)}
                    </div>

                    {/* MID */}
                    <div className="flex justify-center gap-4 sm:gap-12 items-center h-1/5 px-4">
                        {mid.map((p, i) => <PlayerCard key={p.id} player={p} index={i} total={mid.length} />)}
                    </div>

                    {/* FWD */}
                    <div className="flex justify-center gap-8 sm:gap-20 items-center h-1/5 px-4 pb-4">
                        {fwd.map((p, i) => <PlayerCard key={p.id} player={p} index={i} total={fwd.length} />)}
                    </div>

                </div>
            </div>

            {/* Bench Area */}
            <div className="mt-6 bg-gradient-to-r from-gray-800 to-gray-900 p-4 rounded-xl border border-gray-700 shadow-lg">
                <h4 className="text-gray-400 text-xs font-bold mb-3 uppercase tracking-[0.2em] ml-2">Substitutes</h4>
                <div className="flex justify-center gap-6 sm:gap-10 overflow-x-auto py-2">
                    {bench.map(p => <PlayerCard key={p.id} player={p} isBench={true} />)}
                </div>
            </div>
        </div>
    )
}

export default Pitch
