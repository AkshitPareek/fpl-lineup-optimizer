export const CHIP_LABELS = {
  wildcard: 'Wildcard',
  free_hit: 'Free Hit',
  bench_boost: 'Bench Boost',
  triple_captain: 'Triple Captain',
}

export const POSITION_SHORT = { 1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD' }

export function chipLabel(chip) {
  if (!chip) return ''
  return CHIP_LABELS[chip] || chip.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
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
