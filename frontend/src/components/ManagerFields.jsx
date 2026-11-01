import { useState } from 'react'
import { Badge, Field, Spinner, Stepper, TextInput } from './ui'
import { compactRank, parseManagerId, price } from '../lib/format'

/** Whose team this is, once FPL has answered; or why it could not be loaded. */
function ManagerStatus({ account }) {
  if (!account) return null
  if (account.status === 'loading') {
    return <p className="mt-1.5 flex items-center gap-2 text-[11px] text-fg-3" aria-live="polite"><Spinner className="h-3 w-3" /> Looking up your team…</p>
  }
  if (account.status === 'not_found') {
    return (
      <p role="alert" className="mt-1.5 text-[11px] leading-snug text-[#ffb3ba]">
        FPL has no team with this ID. Check the number after <span className="font-mono">/entry/</span> in your Points page address.
      </p>
    )
  }
  if (account.status === 'error') {
    return (
      <p role="alert" className="mt-1.5 text-[11px] leading-snug text-[#ffb3ba]">
        Couldn't load your team from FPL.{' '}
        <button type="button" onClick={account.retry} className="font-semibold text-fpl-green hover:underline cursor-pointer">Retry</button>
      </p>
    )
  }
  const entry = account.entry
  if (account.status !== 'ok') return null
  if (!entry?.name) {
    return <p className="mt-1.5 text-[11px] text-fg-3" aria-live="polite">Squad, bank and free transfers loaded from FPL.</p>
  }
  const rank = compactRank(entry.overall_rank)
  return (
    <div className="mt-2 flex items-center gap-2.5 rounded-xl bg-fpl-green/[0.06] px-3 py-2 ring-1 ring-inset ring-fpl-green/25" aria-live="polite">
      <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-fpl-green/15 text-xs font-bold text-fpl-green" aria-hidden="true">
        {entry.name.slice(0, 1).toUpperCase()}
      </span>
      <div className="min-w-0 leading-tight">
        <div className="truncate text-sm font-semibold text-fg">{entry.name}</div>
        <div className="truncate text-[11px] text-fg-3 tabular">
          {[entry.player_name, rank && `Rank ${rank}`, entry.overall_points != null && `${entry.overall_points} pts`].filter(Boolean).join(' · ')}
        </div>
      </div>
    </div>
  )
}

export default function ManagerFields({ managerId, setManagerId, validManagerId, account, budget, setBudget, bankLocked = false }) {
  const [helpOpen, setHelpOpen] = useState(false)
  const text = managerId.trim()
  const invalid = text && !parseManagerId(text)

  const onChange = value => {
    // A pasted Points-page address becomes its ID, so the field shows what is used.
    const parsed = parseManagerId(value)
    setManagerId(parsed && !/^\d+$/.test(value.trim()) ? parsed : value)
  }

  return (
    <>
      <Field
        label="FPL manager ID"
        htmlFor="manager-id"
        aside={
          account?.status === 'ok' ? <Badge tone="green">Linked</Badge> : (
            <button
              type="button"
              aria-expanded={helpOpen}
              aria-controls="manager-id-help"
              onClick={() => setHelpOpen(o => !o)}
              className="text-[11px] text-fpl-green hover:underline cursor-pointer rounded focus-visible:outline-2 focus-visible:outline-fpl-green"
            >
              Where's my ID?
            </button>
          )
        }
      >
        <TextInput
          id="manager-id"
          inputMode="numeric"
          autoComplete="off"
          value={managerId}
          onChange={e => onChange(e.target.value)}
          placeholder="ID or Points page link"
          aria-invalid={Boolean(invalid) || account?.status === 'not_found'}
          aria-describedby="manager-id-help"
        />
        <div id="manager-id-help">
          {(helpOpen || invalid || (!validManagerId && !text)) && (
            <p className="mt-1.5 text-[11px] leading-snug text-fg-3">
              {invalid && <span className="text-[#ffb3ba]">That isn't an ID. </span>}
              On fantasy.premierleague.com open <span className="text-fg-2">Points</span>; your ID is the number after{' '}
              <span className="font-mono text-fg-2">/entry/</span> in the address. You can paste the whole link.
            </p>
          )}
          {validManagerId && <ManagerStatus account={account} />}
          {account?.status === 'ok' && (
            <button
              type="button"
              aria-expanded={helpOpen}
              onClick={() => setHelpOpen(o => !o)}
              className="mt-1 text-[11px] text-fg-3 hover:text-fg-2 hover:underline cursor-pointer"
            >
              {helpOpen ? 'Hide help' : "Where's my ID?"}
            </button>
          )}
        </div>
      </Field>
      <Field
        label={validManagerId ? 'Money in the bank' : 'Budget'}
        hint={bankLocked ? 'Loaded from your FPL account.' : validManagerId ? 'Used when your bank could not be read from FPL.' : 'Total budget for a squad picked from scratch.'}
      >
        {bankLocked ? (
          <p className="text-sm font-semibold tabular">{price(budget)}</p>
        ) : (
          <Stepper value={budget} onChange={setBudget} min={0} max={200} step={0.5} suffix="£m" />
        )}
      </Field>
    </>
  )
}
