import { useCallback, useState } from 'react'
import Pitch from './Pitch'
import PlayerList from './PlayerList'
import PlayerSheet from './PlayerSheet'
import { Segmented } from './ui'
import { IconList, IconPitch } from './icons'

/** Pitch / list switcher. The choice is shared via the optional controlled props. */
export function ViewToggle({ value, onChange }) {
  return (
    <Segmented
      size="sm"
      value={value}
      onChange={onChange}
      options={[
        { value: 'pitch', label: <span className="flex items-center gap-1.5"><IconPitch size={14} />Pitch</span> },
        { value: 'list', label: <span className="flex items-center gap-1.5"><IconList size={14} />List</span> },
      ]}
    />
  )
}

/**
 * Pitch or list of a squad. Tapping a player opens his details; `playerActions`
 * adds buttons there, `fixtures` maps team id to this week's opponents.
 */
export default function SquadView({ squad, view: controlledView, title, actions, gameweek, fixtures, playerActions, kept }) {
  const [localView, setLocalView] = useState('pitch')
  const [selectedId, setSelectedId] = useState(null)
  const view = controlledView ?? localView
  const selected = squad?.find(p => p.id === selectedId) || null
  const onSelect = useCallback(p => setSelectedId(p.id), [])
  const onClose = useCallback(() => setSelectedId(null), [])
  const props = { fixtures, onSelect, kept }

  return (
    <div>
      {(title || controlledView == null) && (
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">{title}</div>
          <div className="flex items-center gap-2">
            {actions}
            {controlledView == null && <ViewToggle value={view} onChange={setLocalView} />}
          </div>
        </div>
      )}
      {view === 'pitch' ? <Pitch lineup={squad} {...props} /> : <PlayerList squad={squad} {...props} />}
      <PlayerSheet player={selected} gameweek={gameweek} fixture={selected && fixtures?.[selected.team]} onClose={onClose} actions={playerActions} />
    </div>
  )
}
