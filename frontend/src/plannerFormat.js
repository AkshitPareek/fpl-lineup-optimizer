const CHIP_NAMES = {
  triple_captain: 'Triple Captain',
  bench_boost: 'Bench Boost',
  free_hit: 'Free Hit',
  freehit: 'Free Hit',
  wildcard: 'Wildcard',
}

const CHIP_SHORT = {
  triple_captain: 'TC',
  bench_boost: 'BB',
  free_hit: 'FH',
  freehit: 'FH',
  wildcard: 'WC',
}

export const POSITIONS = { 1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD' }

export function chipName(chip) {
  if (!chip) return ''
  return CHIP_NAMES[chip] || String(chip).replaceAll('_', ' ')
}

export function chipShort(chip) {
  if (!chip) return ''
  return CHIP_SHORT[chip] || chipName(chip)
}

export function formatPoints(value, digits = 1) {
  if (typeof value !== 'number' || Number.isNaN(value)) return null
  return value.toFixed(digits)
}

export function formatMoney(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return null
  return `£${value.toFixed(1)}m`
}

export function isUsableMultiPlan(plan) {
  return Boolean(
    plan &&
    plan.status === 'Optimal' &&
    Array.isArray(plan.gameweek_plans) &&
    plan.gameweek_plans.length > 0 &&
    typeof plan.total_expected_points === 'number'
  )
}

export function plansAreIdentical(data) {
  const note = data?.comparison?.note || ''
  if (/identical/i.test(note)) return true
  const withHits = data?.comparison?.with_hits
  const noHits = data?.comparison?.no_hits
  return Boolean(
    withHits?.feasible &&
    noHits?.feasible &&
    (withHits.total_hits || 0) === 0 &&
    data?.comparison?.difference === 0 &&
    withHits.net_xp != null &&
    withHits.net_xp === noHits.net_xp
  )
}

export function chooseStrategy(data) {
  const identical = plansAreIdentical(data)
  const withHits = isUsableMultiPlan(data?.with_hits) ? data.with_hits : null
  const noHits = isUsableMultiPlan(data?.no_hits) ? data.no_hits : null
  const noHitsRecommended = data?.comparison?.no_hits?.feasible !== false && data?.comparison?.no_hits?.recommended
  if (!identical && noHits && (noHitsRecommended || !withHits)) {
    return { key: 'no_hits', plan: noHits, identical }
  }
  return { key: 'with_hits', plan: withHits, identical }
}

export function hitSummary(plan) {
  const count = (plan?.gameweek_plans || []).reduce((sum, week) => sum + (week.transfers?.hits || 0), 0)
  return { count, points: count * 4 }
}

export function planStats(plan) {
  const weeks = plan?.gameweek_plans || []
  const total = plan?.total_expected_points
  const average = weeks.length && typeof total === 'number' ? total / weeks.length : null
  return {
    weeks: weeks.length,
    total,
    average,
    hits: hitSummary(plan),
    firstGw: weeks[0]?.gameweek,
    lastGw: weeks[weeks.length - 1]?.gameweek,
  }
}

export function waitMessage(gameweeks) {
  const n = Number(gameweeks) || 8
  if (n >= 15) return 'A 15-week plan can take about a minute.'
  if (n <= 8) return 'An 8-week plan is usually ready in a few seconds.'
  return 'An 8-week plan is usually ready in a few seconds. A 15-week plan can take about a minute.'
}

export function failureCopy(raw) {
  const detail = String(raw || 'The planner did not return a plan.').replace(/^Failed\.\s*/, '')
  const lower = detail.toLowerCase()
  const hints = []
  if (lower.includes('timeout') || lower.includes('timed out')) {
    hints.push('Shorten the horizon. An 8-week plan is usually ready in a few seconds. A 15-week plan can take about a minute and may time out.')
  } else {
    hints.push('Try 8 gameweeks if a longer horizon does not finish. A 15-week plan can take about a minute.')
  }
  if (lower.includes('manager') || lower.includes('not found')) {
    hints.push('Check the manager ID. It is the number in your FPL team URL. Clear it to plan a new squad from the budget.')
  } else {
    hints.push('Check the manager ID, or clear it and set a budget if you want a new squad.')
  }
  return {
    title: 'The plan did not finish',
    detail,
    hints,
  }
}

export function infeasibleCopy(note) {
  return {
    title: 'No plan fits this squad',
    detail: note || 'The solver could not build a legal plan over this horizon.',
    hints: [
      'Shorten the horizon and generate the plan again.',
      'Confirm the manager ID so the squad, bank, and free transfers are yours. Clear the ID to plan a new squad from the budget.',
    ],
  }
}

export function playerLabel(player) {
  if (!player) return 'Unknown'
  if (typeof player === 'string') return player
  return player.name || player.web_name || 'Unknown'
}

export function plannedPrice(player) {
  if (!player || typeof player !== 'object') return null
  if (typeof player.cost === 'number') return player.cost
  return null
}

export function weekIndexFor(plan, gameweek) {
  const index = (plan?.gameweek_plans || []).findIndex((week) => week.gameweek === gameweek)
  return index >= 0 ? index : null
}
