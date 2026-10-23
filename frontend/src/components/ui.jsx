// Shared UI primitives. Keep these dumb: layout and styling only, no data fetching.

import { cx } from '../lib/cx'

export function Card({ className, children, ...props }) {
  return (
    <div
      className={cx('rounded-2xl border border-ink-700/80 bg-ink-850/80 backdrop-blur-sm shadow-[0_1px_0_0_rgb(255_255_255/0.03)_inset]', className)}
      {...props}
    >
      {children}
    </div>
  )
}

export function CardHeader({ title, subtitle, icon, action, className }) {
  return (
    <div className={cx('flex items-start justify-between gap-4', className)}>
      <div className="flex items-start gap-3 min-w-0">
        {icon && <div className="mt-0.5 text-fg-3 shrink-0">{icon}</div>}
        <div className="min-w-0">
          <h3 className="font-display text-[15px] font-semibold tracking-tight text-fg">{title}</h3>
          {subtitle && <p className="text-xs text-fg-3 mt-0.5">{subtitle}</p>}
        </div>
      </div>
      {action}
    </div>
  )
}

export function SectionLabel({ children, className }) {
  return <div className={cx('text-[11px] font-semibold uppercase tracking-[0.12em] text-fg-3', className)}>{children}</div>
}

const BADGE_TONES = {
  neutral: 'bg-ink-700 text-fg-2',
  green: 'bg-fpl-green/12 text-fpl-green ring-1 ring-inset ring-fpl-green/25',
  pink: 'bg-fpl-pink/12 text-fpl-pink ring-1 ring-inset ring-fpl-pink/25',
  warn: 'bg-warn/12 text-warn ring-1 ring-inset ring-warn/25',
  bad: 'bg-bad/12 text-bad ring-1 ring-inset ring-bad/25',
  violet: 'bg-series-predicted/15 text-[#b9a8ff] ring-1 ring-inset ring-series-predicted/30',
}

export function Badge({ tone = 'neutral', className, children, ...props }) {
  return (
    <span
      className={cx('inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide whitespace-nowrap', BADGE_TONES[tone], className)}
      {...props}
    >
      {children}
    </span>
  )
}

export function ConfidenceBadge({ level }) {
  if (!level) return null
  const tone = level === 'high' ? 'green' : level === 'medium' ? 'warn' : 'neutral'
  return <Badge tone={tone}>{level}</Badge>
}

export function Button({ variant = 'primary', size = 'md', className, children, ...props }) {
  const variants = {
    primary: 'bg-fpl-green text-ink-950 hover:bg-[#3dffa2] shadow-[0_8px_24px_-12px_rgb(0_255_135/0.7)]',
    secondary: 'bg-ink-700 text-fg hover:bg-ink-600',
    ghost: 'text-fg-2 hover:text-fg hover:bg-ink-700/60',
  }
  const sizes = {
    sm: 'h-8 px-3 text-xs',
    md: 'h-10 px-4 text-sm',
    lg: 'h-12 px-5 text-[15px]',
  }
  return (
    <button
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-xl font-semibold transition cursor-pointer',
        'disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fpl-green',
        variants[variant], sizes[size], className,
      )}
      {...props}
    >
      {children}
    </button>
  )
}

export function Segmented({ options, value, onChange, size = 'md', className }) {
  return (
    <div className={cx('inline-flex rounded-xl bg-ink-900 p-1 ring-1 ring-inset ring-ink-700', className)} role="tablist">
      {options.map(opt => {
        const active = opt.value === value
        return (
          <button
            key={opt.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(opt.value)}
            className={cx(
              'flex-1 rounded-lg font-semibold transition whitespace-nowrap cursor-pointer',
              size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3 py-1.5 text-sm',
              active ? 'bg-ink-700 text-fg shadow-sm' : 'text-fg-3 hover:text-fg-2',
            )}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}

export function Field({ label, hint, htmlFor, aside, children }) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <label htmlFor={htmlFor} className="text-xs font-medium text-fg-2">{label}</label>
        {aside}
      </div>
      {children}
      {hint && <p className="text-[11px] text-fg-3 mt-1.5 leading-snug">{hint}</p>}
    </div>
  )
}

const inputClass =
  'w-full h-10 rounded-xl bg-ink-900 px-3 text-sm text-fg placeholder:text-fg-3/70 ring-1 ring-inset ring-ink-700 outline-none transition focus:ring-fpl-green/60 tabular'

export function TextInput({ className, ...props }) {
  return <input className={cx(inputClass, className)} {...props} />
}

export function Select({ className, children, ...props }) {
  return (
    <div className="relative">
      <select className={cx(inputClass, 'appearance-none pr-9 cursor-pointer', className)} {...props}>
        {children}
      </select>
      <svg className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-fg-3" width="12" height="12" viewBox="0 0 12 12" fill="none">
        <path d="M3 4.5 6 7.5l3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  )
}

export function Stepper({ value, onChange, min = 0, max = 99, step = 1, suffix }) {
  const n = Number(value)
  const clamp = v => Math.min(max, Math.max(min, v))
  const round = v => Math.round(v * 10) / 10
  return (
    <div className="flex h-10 items-stretch rounded-xl bg-ink-900 ring-1 ring-inset ring-ink-700 focus-within:ring-fpl-green/60">
      <button type="button" aria-label="Decrease" onClick={() => onChange(round(clamp(n - step)))} className="w-10 text-fg-3 hover:text-fg cursor-pointer">−</button>
      <div className="flex flex-1 items-center justify-center gap-1">
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={e => onChange(e.target.value)}
          onBlur={() => onChange(round(clamp(Number(value) || min)))}
          className="w-full bg-transparent text-center text-sm font-semibold text-fg outline-none tabular"
        />
        {suffix && <span className="text-xs text-fg-3 pr-1">{suffix}</span>}
      </div>
      <button type="button" aria-label="Increase" onClick={() => onChange(round(clamp(n + step)))} className="w-10 text-fg-3 hover:text-fg cursor-pointer">+</button>
    </div>
  )
}

export function Toggle({ checked, onChange, label, description }) {
  return (
    <label className="flex items-start gap-3 cursor-pointer select-none">
      <span className="relative mt-0.5 shrink-0">
        <input type="checkbox" className="peer sr-only" checked={checked} onChange={e => onChange(e.target.checked)} />
        <span className="block h-5 w-9 rounded-full bg-ink-700 transition peer-checked:bg-fpl-green/80 peer-focus-visible:outline-2 peer-focus-visible:outline-fpl-green" />
        <span className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-fg shadow transition peer-checked:translate-x-4 peer-checked:bg-ink-950" />
      </span>
      <span>
        <span className="block text-sm font-medium text-fg">{label}</span>
        {description && <span className="block text-[11px] text-fg-3 mt-0.5">{description}</span>}
      </span>
    </label>
  )
}

export function Stat({ label, value, sub, tone, className }) {
  const toneClass = tone === 'green' ? 'text-fpl-green' : tone === 'bad' ? 'text-bad' : tone === 'good' ? 'text-good' : 'text-fg'
  return (
    <div className={cx('rounded-xl bg-ink-900/70 ring-1 ring-inset ring-ink-700/70 px-4 py-3', className)}>
      <SectionLabel>{label}</SectionLabel>
      <div className={cx('font-display text-2xl font-semibold tabular mt-1', toneClass)}>{value}</div>
      {sub && <div className="text-[11px] text-fg-3 mt-0.5">{sub}</div>}
    </div>
  )
}

export function Spinner({ className }) {
  return <span className={cx('inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent', className)} />
}

export function ProgressBar({ value, indeterminate }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-700">
      {indeterminate ? (
        <div className="h-full w-full animate-shimmer bg-[length:200%_100%] bg-gradient-to-r from-transparent via-fpl-green to-transparent" />
      ) : (
        <div className="h-full rounded-full bg-fpl-green transition-all duration-300" style={{ width: `${value}%` }} />
      )}
    </div>
  )
}

export function EmptyState({ icon, title, children, action }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center text-center px-6 py-16">
      {icon && <div className="mb-5 grid h-14 w-14 place-items-center rounded-2xl bg-ink-800 ring-1 ring-inset ring-ink-700 text-fg-2">{icon}</div>}
      <h3 className="font-display text-lg font-semibold text-fg">{title}</h3>
      {children && <p className="mt-1.5 max-w-sm text-sm text-fg-3">{children}</p>}
      {action && <div className="mt-6">{action}</div>}
    </div>
  )
}

export function LoadingState({ title, detail, progress }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-6 py-16">
      <div className="w-full max-w-sm">
        <div className="flex items-center justify-between text-sm mb-3">
          <span className="flex items-center gap-2 text-fg"><Spinner className="text-fpl-green" />{title}</span>
          {progress != null && <span className="text-fg-3 tabular">{Math.round(progress)}%</span>}
        </div>
        <ProgressBar value={progress} indeterminate={progress == null} />
        {detail && <p className="mt-3 text-center text-xs text-fg-3">{detail}</p>}
      </div>
    </div>
  )
}

export function ErrorNote({ children }) {
  if (!children) return null
  return (
    <div role="alert" className="flex items-start gap-2 rounded-xl bg-bad/10 px-3 py-2.5 text-xs text-[#ffb3ba] ring-1 ring-inset ring-bad/30">
      <svg className="mt-px shrink-0" width="14" height="14" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.5" /><path d="M8 4.5v4M8 11h.01" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>
      <span>{children}</span>
    </div>
  )
}
