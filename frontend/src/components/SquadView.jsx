import { useState } from 'react'
import Pitch from './Pitch'
import PlayerList from './PlayerList'
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

export default function SquadView({ squad, view: controlledView, title, actions }) {
  const [localView, setLocalView] = useState('pitch')
  const view = controlledView ?? localView

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
      {view === 'pitch' ? <Pitch lineup={squad} /> : <PlayerList squad={squad} />}
    </div>
  )
}
