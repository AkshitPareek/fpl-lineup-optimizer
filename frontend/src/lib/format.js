export const CHIP_LABELS = {
  wildcard: 'Wildcard',
  free_hit: 'Free Hit',
  bench_boost: 'Bench Boost',
  triple_captain: 'Triple Captain',
}

export const POSITION_SHORT = { 1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD' }

// The planner names the Free Hit "freehit"; FPL uses "3xc" and "bboost" too.
const CHIP_ALIASES = { freehit: 'free_hit', '3xc': 'triple_captain', bboost: 'bench_boost' }

/** One name per chip: wildcard, free_hit, bench_boost, triple_captain. */
export function chipKey(chip) {
  return chip ? CHIP_ALIASES[chip] || chip : chip
}

export function chipLabel(chip) {
  if (!chip) return ''
  const key = chipKey(chip)
  return CHIP_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

export function num(value, digits = 1) {
  const n = Number(value)
  return Number.isFinite(n) ? n.toFixed(digits) : '–'
}

export function signed(value, digits = 1) {
  const n = Number(value)
  if (!Number.isFinite(n)) return '–'
  const text = Math.abs(n).toFixed(digits)
  if (Number(text) === 0) return text
  return `${n > 0 ? '+' : '−'}${text}`
}

export function price(value) {
  const n = Number(value)
  return Number.isFinite(n) ? `£${n.toFixed(1)}m` : '–'
}

export function shirtUrl(player) {
  const isGK = player.element_type === 1
  return `https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_${player.team_code}${isGK ? '_1' : ''}-66.png`
}

export function positionOf(player) {
  return player.position || POSITION_SHORT[player.element_type] || ''
}

/**
 * A manager ID from what was typed: the number itself, or any FPL URL with
 * /entry/<id>/ in it (the Points page). Empty when there is no ID to use.
 */
export function parseManagerId(input) {
  const text = String(input || '').trim()
  if (/^\d+$/.test(text)) return text
  const match = text.match(/entry\/(\d+)/)
  return match ? match[1] : ''
}

/** 4039613 -> "4.04m", 123456 -> "123k", 950 -> "950". */
export function compactRank(n) {
  const v = Number(n)
  if (!Number.isFinite(v) || v <= 0) return null
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}m`
  if (v >= 1e4) return `${Math.round(v / 1e3)}k`
  return v.toLocaleString()
}

/** "4 min ago", "2 h ago", "3 d ago" from a millisecond timestamp. */
export function timeAgo(ms, now = Date.now()) {
  const s = Math.max(0, Math.round((now - ms) / 1000))
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}

/** "2d 4h", "3h 12m", "8m" until a time; null once it has passed. */
export function countdown(isoTime, now = Date.now()) {
  const ms = new Date(isoTime).getTime() - now
  if (!Number.isFinite(ms) || ms <= 0) return null
  const m = Math.floor(ms / 60000)
  const d = Math.floor(m / 1440)
  const h = Math.floor((m % 1440) / 60)
  if (d > 0) return `${d}d ${h}h`
  if (h > 0) return `${h}h ${m % 60}m`
  return `${m}m`
}

export function clockTime(ms) {
  if (!ms) return ''
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}
