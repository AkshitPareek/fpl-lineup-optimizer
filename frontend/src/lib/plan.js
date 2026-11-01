// Plain helpers over a /api/optimize/compare response. No React here.

import { chipKey, chipLabel, num, price } from './format'

/**
 * Typical seconds for a warm plan, measured on the live API in Nov 2026:
 * 3 GWs 8.6 s, 5 GWs 16 s, 8 GWs 21 s. A cold start adds up to ~45 s.
 */
export function expectedSeconds(horizon) {
  return Math.round(3 + 2.3 * horizon)
}

/** True when the hit plan took no hits, so the free-transfer plan is the same plan. */
export function strategiesIdentical(compare) {
  const c = compare?.comparison
  return Boolean(c?.with_hits?.feasible !== false && c?.with_hits?.total_hits === 0)
}

/**
 * The plan for one strategy. The API may send `no_hits: null` when it equals
 * `with_hits`, so fall back to the hit plan in that case.
 */
export function planFor(compare, key) {
  if (!compare) return null
  if (compare[key]?.gameweek_plans?.length) return compare[key]
  if (key === 'no_hits' && strategiesIdentical(compare)) return compare.with_hits
  return null
}

export function recommendedStrategy(compare) {
  if (strategiesIdentical(compare)) return 'with_hits'
  const noHits = compare?.comparison?.no_hits
  const feasible = noHits?.feasible !== false && planFor(compare, 'no_hits')
  return feasible && noHits?.recommended ? 'no_hits' : 'with_hits'
}

/** Bench in the order FPL uses: the backup keeper, then substitutes first to last. */
export function benchOrder(bench) {
  const gk = (bench || []).filter(p => p.element_type === 1)
  const outfield = (bench || []).filter(p => p.element_type !== 1)
  return [...gk, ...outfield]
}

const FREE_TRANSFER_CHIPS = new Set(['wildcard', 'free_hit'])

/**
 * Free transfers each week: available at the deadline, used, and carried to the next.
 * Mirrors the planner: one more each week up to the cap; a Wildcard or Free Hit
 * week keeps the saved ones without adding one; hits leave one for next week.
 */
export function freeTransferLedger(gameweekPlans, startFt, cap = 5) {
  let available = Number.isFinite(Number(startFt)) ? Number(startFt) : 1
  return (gameweekPlans || []).map(p => {
    const used = Math.min(available, p.transfers?.ft_used ?? (p.transfers?.in?.length || 0))
    const chipWeek = FREE_TRANSFER_CHIPS.has(chipKey(p.chip_used))
    const next = chipWeek ? available : Math.min(cap, Math.max(0, available - used) + 1)
    const row = { gameweek: p.gameweek, available, used: chipWeek ? 0 : used, next }
    available = next
    return row
  })
}

/** The starters' points multipliers this week: captain x2 (x3 with Triple Captain), bench only on Bench Boost. */
function multipliers(gwPlan) {
  const tc = chipKey(gwPlan.chip_used) === 'triple_captain'
  const bb = chipKey(gwPlan.chip_used) === 'bench_boost'
  const rows = gwPlan.starting_xi.map(p => [p, p.is_captain ? (tc ? 3 : 2) : 1])
  if (bb) rows.push(...gwPlan.bench.map(p => [p, 1]))
  return rows
}

/**
 * A rough 80% range for a week's points from each player's sigma, treating
 * players as independent. Null when the API sent no sigmas.
 */
export function weekRange(gwPlan) {
  if (!gwPlan?.starting_xi?.length) return null
  const key = `sigma_gw${gwPlan.gameweek}`
  let variance = 0
  let seen = false
  for (const [p, m] of multipliers(gwPlan)) {
    const s = Number(p[key])
    if (!Number.isFinite(s)) continue
    seen = true
    variance += (m * s) ** 2
  }
  if (!seen || variance <= 0) return null
  const spread = 1.28 * Math.sqrt(variance)
  const xp = Number(gwPlan.expected_points) || 0
  return { low: Math.max(0, xp - spread), high: xp + spread }
}

/** One player's 80% range this week, from his sigma. */
export function playerRange(player, gameweek) {
  const s = Number(player?.[`sigma_gw${gameweek}`])
  const xp = Number(player?.expected_points)
  if (!Number.isFinite(s) || !Number.isFinite(xp) || s <= 0) return null
  return { low: Math.max(0, xp - 1.28 * s), high: xp + 1.28 * s }
}

/** The captain, the vice, and how far the captain is ahead of the next best starter. */
export function captaincy(gwPlan) {
  const xi = gwPlan?.starting_xi || []
  const captain = xi.find(p => p.is_captain)
  const vice = xi.find(p => p.is_vice_captain)
  const next = xi
    .filter(p => p !== captain)
    .sort((a, b) => (b.expected_points || 0) - (a.expected_points || 0))[0]
  return {
    captain,
    vice,
    next,
    margin: captain && next ? (captain.expected_points || 0) - (next.expected_points || 0) : null,
  }
}

/**
 * Each team's fixtures in one gameweek, as "NFO (A)", from /api/data.
 * Empty when the page has no fixtures (e.g. a slimmer /api/data).
 */
export function fixturesForGameweek(data, gameweek) {
  const teams = data?.static?.teams
  const fixtures = data?.fixtures
  if (!Array.isArray(teams) || !Array.isArray(fixtures) || !gameweek) return {}
  const short = Object.fromEntries(teams.map(t => [t.id, t.short_name]))
  const out = {}
  for (const f of fixtures) {
    if (f.event !== gameweek) continue
    ;(out[f.team_h] ||= []).push(`${short[f.team_a] || '?'} (H)`)
    ;(out[f.team_a] ||= []).push(`${short[f.team_h] || '?'} (A)`)
  }
  return Object.fromEntries(Object.entries(out).map(([k, v]) => [k, v.join(', ')]))
}

/** A short "doubt" note for a flagged player, or null when he is fully available. */
export function availabilityNote(player) {
  if (!player) return null
  const chance = player.chance_of_playing_next_round
  if (player.status === 'a' && (chance == null || chance >= 100)) return null
  if (player.status === 'a' || player.status == null) return chance != null && chance < 100 ? `${chance}% chance of playing` : null
  const label = { d: 'Doubtful', i: 'Injured', s: 'Suspended', u: 'Unavailable', n: 'Not available' }[player.status] || 'Flagged'
  return [label, chance != null ? `${chance}%` : null, player.news].filter(Boolean).join(' · ')
}

/**
 * Squad players whose FPL status or news changed since the plan was made,
 * judged against the page's current FPL data.
 */
export function newsChanges(gwPlan, elements) {
  if (!gwPlan || !Array.isArray(elements)) return []
  const now = new Map(elements.map(e => [e.id, e]))
  const changed = []
  for (const p of [...gwPlan.starting_xi, ...gwPlan.bench]) {
    const cur = now.get(p.id)
    if (!cur) continue
    if (cur.status !== p.status || (cur.news || '') !== (p.news || '') ||
        (cur.chance_of_playing_next_round ?? null) !== (p.chance_of_playing_next_round ?? null)) {
      changed.push({ ...p, ...cur })
    }
  }
  return changed
}

/** Pair each incoming player with an outgoing one in the same position, as the explainer does. */
export function pairTransfers(ins, outs) {
  const remaining = [...(outs || [])]
  const pairs = (ins || []).map(inn => {
    const idx = remaining.findIndex(o => o.element_type === inn.element_type)
    const out = idx >= 0 ? remaining.splice(idx, 1)[0] : null
    return [out, inn]
  })
  return [...pairs, ...remaining.map(o => [o, null])]
}

/** Plain-text checklist for the deadline week, for pasting into notes or a chat. */
export function deadlineText({ gwPlan, ledger, chipAdvice, managerName }) {
  const lines = [`GW${gwPlan.gameweek}${managerName ? ` · ${managerName}` : ''}`]
  const t = gwPlan.transfers || {}
  const pairs = pairTransfers(t.in, t.out)
  if (pairs.length) {
    lines.push('Transfers:')
    for (const [out, inn] of pairs) {
      lines.push(`  ${out ? out.name : '—'} -> ${inn ? inn.name : '—'}`)
    }
    if (t.hits > 0) lines.push(`  Hits: ${t.hits} (-${t.hits * 4})`)
  } else {
    lines.push('Transfers: none (roll the free transfer)')
  }
  if (typeof gwPlan.bank === 'number') lines.push(`Bank after: ${price(gwPlan.bank)}`)
  if (ledger) lines.push(`Free transfers next week: ${ledger.next}`)
  const { captain, vice } = captaincy(gwPlan)
  if (captain) lines.push(`Captain: ${captain.web_name}${vice ? ` · Vice: ${vice.web_name}` : ''}`)
  const byPos = [1, 2, 3, 4].map(type => gwPlan.starting_xi.filter(p => p.element_type === type).map(p => p.web_name).join(', '))
  lines.push(`XI: ${byPos.join(' | ')}`)
  const bench = benchOrder(gwPlan.bench)
  if (bench.length) lines.push(`Bench: ${bench.map((p, i) => `${i === 0 ? 'GK' : i}. ${p.web_name}`).join(', ')}`)
  lines.push(`Chip: ${gwPlan.chip_used ? chipLabel(gwPlan.chip_used) : 'none'}${chipAdvice ? ` (${chipAdvice})` : ''}`)
  lines.push(`Projected: ${num(gwPlan.expected_points)} pts`)
  return lines.join('\n')
}

/** What the chip line says for the deadline week, and a short form for the copied text. */
export function chipLine(gwPlan, plan) {
  if (gwPlan.chip_used) return { text: `Play ${chipLabel(gwPlan.chip_used)} this week.`, short: null, play: true }
  const later = (plan.gameweek_plans || []).find(p => p.chip_used && p.gameweek > gwPlan.gameweek)
  if (later) {
    return { text: `No chip this week. ${chipLabel(later.chip_used)} is planned for GW${later.gameweek}.`, short: `${chipLabel(later.chip_used)} planned GW${later.gameweek}` }
  }
  return { text: 'No chip this week.', short: null }
}
