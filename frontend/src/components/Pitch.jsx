import { useState } from 'react'
import { cx } from '../lib/cx'
import { num, positionOf, price, shirtUrl } from '../lib/format'

const ROWS = [1, 2, 3, 4] // GKP, DEF, MID, FWD

function Shirt({ player, small }) {
  const [failed, setFailed] = useState(false)
  const size = small ? 'w-9 sm:w-10' : 'w-10 sm:w-14'
  if (failed || !player.team_code) {
    return (
      <svg viewBox="0 0 64 64" className={cx(size, 'drop-shadow-[0_6px_8px_rgb(0_0_0/0.45)]')} aria-hidden="true">
        <path d="M18 10 6 22l8 9 4-3v26h28V28l4 3 8-9-12-12h-8a6 6 0 0 1-12 0h-8Z" fill={player.element_type === 1 ? '#ffc94d' : '#ff2e7e'} stroke="rgb(255 255 255 / 0.6)" strokeWidth="1.5" />
      </svg>
    )
  }
  return (
    <img
      src={shirtUrl(player)}
      alt=""
      loading="lazy"
      onError={() => setFailed(true)}
      className={cx(size, 'object-contain drop-shadow-[0_6px_8px_rgb(0_0_0/0.45)]')}
    />
  )
}

function PlayerToken({ player, small = false, benchIndex }) {
  const xp = player.expected_points
  return (
    <div className={cx('group relative flex flex-col items-center', small ? 'w-[68px] sm:w-20' : 'w-[62px] sm:w-[88px]')}>
      {benchIndex != null && (
        <span className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-fg-3">{benchIndex === 0 ? 'GK' : `Sub ${benchIndex}`}</span>
      )}
      <div className="relative">
        <Shirt player={player} small={small} />
        {(player.is_captain || player.is_vice_captain) && (
          <span
            className={cx(
              'absolute -right-2 -top-1 grid h-5 w-5 place-items-center rounded-full text-[10px] font-bold ring-2 ring-ink-950',
              player.is_captain ? 'bg-fpl-green text-ink-950' : 'bg-fg text-ink-950',
            )}
            title={player.is_captain ? 'Captain' : 'Vice-captain'}
          >
            {player.is_captain ? 'C' : 'V'}
          </span>
        )}
      </div>

      <div className="mt-1 w-full overflow-hidden rounded-md shadow-[0_6px_14px_-6px_rgb(0_0_0/0.6)]">
        <div className="truncate bg-fg px-1 py-[3px] text-center text-[10px] sm:text-[11px] font-semibold leading-tight text-ink-950">
          {player.web_name}
        </div>
        <div className="flex items-center justify-center gap-1 bg-fpl-purple px-1 py-[2px] text-[10px] sm:text-[11px] text-fg tabular">
          <span className="font-semibold">{num(xp, 1)}</span>
          <span className="text-[9px] text-fg-3">xP</span>
        </div>
      </div>

      {/* Hover card */}
      <div className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 w-44 -translate-x-1/2 rounded-xl bg-ink-950/95 p-3 text-left opacity-0 shadow-xl ring-1 ring-ink-700 transition group-hover:opacity-100">
        <div className="text-sm font-semibold text-fg">{player.web_name}</div>
        <div className="text-[11px] text-fg-3">{[player.team_name, positionOf(player)].filter(Boolean).join(' · ')}</div>
        <dl className="mt-2 grid grid-cols-2 gap-y-1 text-[11px] tabular">
          <dt className="text-fg-3">Expected</dt><dd className="text-right text-fg">{num(xp, 2)} xP</dd>
          {player.now_cost != null && (<><dt className="text-fg-3">Price</dt><dd className="text-right text-fg">{price(player.now_cost > 20 ? player.now_cost / 10 : player.now_cost)}</dd></>)}
          {player.analytics?.ownership_pct != null && (<><dt className="text-fg-3">Owned</dt><dd className="text-right text-fg">{player.analytics.ownership_pct}%</dd></>)}
          {player.analytics?.ceiling != null && (<><dt className="text-fg-3">Ceiling</dt><dd className="text-right text-fg">{num(player.analytics.ceiling)}</dd></>)}
        </dl>
      </div>
    </div>
  )
}

function PitchMarkings() {
  // Half-pitch drawn on a fixed viewBox; `slice` keeps the circles round at any aspect.
  return (
    <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 110" preserveAspectRatio="xMidYMin slice" aria-hidden="true">
      <g fill="none" stroke="rgb(255 255 255 / 0.28)" strokeWidth="0.35">
        <rect x="3" y="2" width="94" height="200" />
        <rect x="22" y="2" width="56" height="18" />
        <rect x="37" y="2" width="26" height="6.5" />
        <path d="M40 20a12 12 0 0 0 20 0" />
        <circle cx="50" cy="14" r="0.5" fill="rgb(255 255 255 / 0.35)" />
        <path d="M44 2V0.5h12V2" />
        <line x1="3" y1="104" x2="97" y2="104" />
        <path d="M37 104a13 13 0 0 1 26 0" />
      </g>
    </svg>
  )
}

const Pitch = ({ lineup }) => {
  if (!lineup) return null

  const starters = lineup.filter(p => p.is_starter)
  const bench = lineup.filter(p => !p.is_starter)
  // Bench order: the backup keeper first, then outfield order as returned by the optimizer.
  const orderedBench = [...bench.filter(p => p.element_type === 1), ...bench.filter(p => p.element_type !== 1)]

  return (
    <div className="w-full select-none">
      <div className="relative overflow-hidden rounded-2xl ring-1 ring-inset ring-white/10 bg-[#0f5a35]">
        {/* Mowed stripes + vignette */}
        <div
          className="absolute inset-0"
          style={{
            backgroundImage:
              'repeating-linear-gradient(180deg, rgb(255 255 255 / 0.035) 0 9%, transparent 9% 18%), radial-gradient(120% 80% at 50% 0%, rgb(39 160 96 / 0.55), transparent 70%), linear-gradient(180deg, #13703f 0%, #0c4a2b 100%)',
          }}
        />
        <PitchMarkings />

        <div className="relative z-10 flex flex-col gap-4 sm:gap-6 px-1 sm:px-4 pt-6 pb-8 sm:pt-8 sm:pb-10">
          {ROWS.map(type => {
            const row = starters.filter(p => p.element_type === type)
            return (
              <div key={type} className="flex justify-evenly items-start">
                {row.map(p => <PlayerToken key={p.id} player={p} />)}
              </div>
            )
          })}
        </div>
      </div>

      {orderedBench.length > 0 && (
        <div className="mt-3 rounded-2xl bg-ink-900/70 ring-1 ring-inset ring-ink-700 px-3 pt-3 pb-4">
          <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-fg-3 mb-3 px-1">Bench</div>
          <div className="flex justify-evenly">
            {orderedBench.map((p, i) => <PlayerToken key={p.id} player={p} small benchIndex={i} />)}
          </div>
        </div>
      )}
    </div>
  )
}

export default Pitch
