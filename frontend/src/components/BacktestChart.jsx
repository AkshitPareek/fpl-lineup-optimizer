import { useEffect, useRef, useState } from 'react'
import { num } from '../lib/format'

const HEIGHT = 220
const PAD = { top: 12, right: 8, bottom: 26, left: 32 }

function niceMax(v) {
  if (v <= 0) return 10
  const step = v > 100 ? 25 : v > 50 ? 20 : 10
  return Math.ceil(v / step) * step
}

/**
 * Actual points (columns) vs. predicted points (dots) per gameweek.
 * One y-axis, both series in points. Clicking a gameweek selects it.
 */
export default function BacktestChart({ weeks, selected, onSelect }) {
  const ref = useRef(null)
  const [width, setWidth] = useState(640)
  const [hover, setHover] = useState(null)

  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(280, entry.contentRect.width)))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])

  const max = niceMax(Math.max(...weeks.flatMap(w => [w.actual_points, w.predicted_points])))
  const innerW = width - PAD.left - PAD.right
  const innerH = HEIGHT - PAD.top - PAD.bottom
  const band = innerW / weeks.length
  const barW = Math.min(24, band * 0.6)
  const y = v => PAD.top + innerH - (v / max) * innerH
  const ticks = [0, max / 2, max]
  const labelEvery = Math.ceil(weeks.length / Math.floor(innerW / 36))
  const hovered = hover != null ? weeks[hover] : null

  return (
    <div ref={ref} className="relative">
      <div className="mb-3 flex items-center gap-4 text-xs text-fg-2">
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-series-actual" />Actual</span>
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-series-predicted" />Predicted</span>
      </div>
      <svg width={width} height={HEIGHT} className="block" onMouseLeave={() => setHover(null)}>
        {ticks.map(t => (
          <g key={t}>
            <line x1={PAD.left} x2={width - PAD.right} y1={y(t)} y2={y(t)} stroke="var(--color-ink-700)" strokeWidth="1" />
            <text x={PAD.left - 8} y={y(t)} dy="0.32em" textAnchor="end" className="fill-fg-3 text-[10px] tabular">{t}</text>
          </g>
        ))}
        {weeks.map((w, i) => {
          const cx = PAD.left + band * i + band / 2
          const top = y(w.actual_points)
          const h = Math.max(0, PAD.top + innerH - top)
          const r = Math.min(4, h)
          const x0 = cx - barW / 2
          const isSel = w.gameweek === selected
          const dim = selected != null && !isSel && hover !== i
          return (
            <g key={w.gameweek} opacity={dim ? 0.55 : 1}>
              {(isSel || hover === i) && (
                <rect x={cx - band / 2} y={PAD.top} width={band} height={innerH} fill="var(--color-ink-800)" rx="6" />
              )}
              {/* Column: 4px rounded data-end, square baseline */}
              <path
                d={`M${x0},${PAD.top + innerH} V${top + r} Q${x0},${top} ${x0 + r},${top} H${x0 + barW - r} Q${x0 + barW},${top} ${x0 + barW},${top + r} V${PAD.top + innerH} Z`}
                fill="var(--color-series-actual)"
              />
              <circle cx={cx} cy={y(w.predicted_points)} r="4.5" fill="var(--color-series-predicted)" stroke="var(--color-ink-850)" strokeWidth="2" />
              {i % labelEvery === 0 && (
                <text x={cx} y={HEIGHT - 8} textAnchor="middle" className={`text-[10px] tabular ${isSel ? 'fill-fg' : 'fill-fg-3'}`}>{w.gameweek}</text>
              )}
              {/* Hit target spans the whole band */}
              <rect
                x={cx - band / 2} y={PAD.top} width={band} height={innerH + PAD.bottom}
                fill="transparent" className="cursor-pointer"
                onMouseEnter={() => setHover(i)}
                onClick={() => onSelect(w.gameweek)}
              />
            </g>
          )
        })}
      </svg>
      {hovered && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg bg-ink-950/95 px-3 py-2 text-xs shadow-xl ring-1 ring-ink-700"
          style={{
            left: Math.min(width - 150, Math.max(0, PAD.left + band * hover + band / 2 - 70)),
            top: 24,
          }}
        >
          <div className="font-semibold text-fg mb-1">Gameweek {hovered.gameweek}</div>
          <div className="grid grid-cols-[auto_auto] gap-x-4 gap-y-0.5 tabular">
            <span className="flex items-center gap-1.5 text-fg-2"><span className="h-2 w-2 rounded-sm bg-series-actual" />Actual</span><span className="text-right text-fg">{hovered.actual_points}</span>
            <span className="flex items-center gap-1.5 text-fg-2"><span className="h-2 w-2 rounded-full bg-series-predicted" />Predicted</span><span className="text-right text-fg">{num(hovered.predicted_points)}</span>
            {hovered.max_possible != null && (<><span className="text-fg-3 pl-3.5">Hindsight best</span><span className="text-right text-fg-2">{hovered.max_possible}</span></>)}
          </div>
        </div>
      )}
    </div>
  )
}
