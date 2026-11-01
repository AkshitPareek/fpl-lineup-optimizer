import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { Badge } from './ui'
import { IconX } from './icons'
import { num, positionOf, price } from '../lib/format'
import { availabilityNote, playerRange } from '../lib/plan'

function cost(player) {
  if (player.now_cost == null) return null
  return player.now_cost > 20 ? player.now_cost / 10 : player.now_cost
}

/**
 * Player details on tap: a bottom sheet on phones, a small dialog on larger screens.
 * `actions` renders extra buttons (keep / ban) for the player.
 */
export default function PlayerSheet({ player, gameweek, fixture, onClose, actions }) {
  const closeRef = useRef(null)
  useEffect(() => {
    if (!player) return
    const previous = document.activeElement
    closeRef.current?.focus()
    const onKey = e => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      previous?.focus?.()
    }
  }, [player, onClose])

  if (!player) return null
  const range = playerRange(player, gameweek)
  const note = availabilityNote(player)
  const a = player.analytics || {}
  const c = cost(player)

  // Portalled to the body: a card's backdrop blur would otherwise trap the fixed overlay inside it.
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center" role="presentation">
      <div className="absolute inset-0 bg-ink-950/70 backdrop-blur-sm" onClick={onClose} aria-hidden="true" />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="player-sheet-title"
        className="relative w-full sm:max-w-sm rounded-t-2xl sm:rounded-2xl bg-ink-850 p-5 pb-[max(1.25rem,env(safe-area-inset-bottom))] ring-1 ring-ink-700 shadow-2xl animate-fade-in"
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 id="player-sheet-title" className="font-display text-lg font-semibold text-fg truncate">{player.web_name}</h3>
            <p className="text-xs text-fg-3">{[player.team_name, positionOf(player), c != null && price(c)].filter(Boolean).join(' · ')}</p>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="grid h-8 w-8 place-items-center rounded-lg text-fg-3 hover:bg-ink-700 hover:text-fg cursor-pointer focus-visible:outline-2 focus-visible:outline-fpl-green"
          >
            <IconX size={16} />
          </button>
        </div>

        {note && <p className="mt-3 rounded-lg bg-warn/10 px-3 py-2 text-xs text-warn ring-1 ring-inset ring-warn/25">{note}</p>}

        <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-sm tabular">
          <dt className="text-fg-3">{gameweek ? `GW${gameweek} projection` : 'Projection'}</dt>
          <dd className="text-right font-semibold text-fg">{num(player.expected_points)} pts</dd>
          {range && (<>
            <dt className="text-fg-3">Likely range</dt>
            <dd className="text-right text-fg-2">{num(range.low, 0)}–{num(range.high, 0)}</dd>
          </>)}
          {fixture && (<><dt className="text-fg-3">Fixture</dt><dd className="text-right text-fg-2">{fixture}</dd></>)}
          {a.ownership_pct != null && (<><dt className="text-fg-3">Owned by</dt><dd className="text-right text-fg-2">{a.ownership_pct}%</dd></>)}
          {player.form != null && (<><dt className="text-fg-3">Form</dt><dd className="text-right text-fg-2">{player.form}</dd></>)}
        </dl>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {player.is_captain && <Badge tone="green">Captain</Badge>}
          {player.is_vice_captain && <Badge>Vice-captain</Badge>}
          {player.is_starter === false && <Badge>Bench</Badge>}
        </div>
        {actions && <div className="mt-4 border-t border-ink-700/70 pt-4">{actions(player)}</div>}
      </div>
    </div>,
    document.body,
  )
}
