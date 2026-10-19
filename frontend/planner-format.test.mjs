import assert from 'node:assert/strict'
import {
  chooseStrategy,
  failureCopy,
  formatMoney,
  infeasibleCopy,
  isUsableMultiPlan,
  plansAreIdentical,
  planStats,
  plannedPrice,
  waitMessage,
} from './src/plannerFormat.js'

const plan = {
  status: 'Optimal',
  total_expected_points: 80,
  proven_optimal: false,
  optimality_gap_points: 1.5,
  banked_transfers_after: 2,
  gameweek_plans: [
    { gameweek: 10, expected_points: 40, transfers: { hits: 1, in: [{ name: 'A', cost: 7.5 }], out: [] } },
    { gameweek: 11, expected_points: 44, transfers: { hits: 0, in: [], out: [] } },
  ],
}

assert.equal(isUsableMultiPlan(plan), true)
assert.equal(isUsableMultiPlan({ status: 'Infeasible', gameweek_plans: [], total_expected_points: 0 }), false)
assert.equal(formatMoney(plannedPrice({ cost: 10.2 })), '£10.2m')
assert.equal(plannedPrice('Salah'), null)

const stats = planStats(plan)
assert.equal(stats.average, 40)
assert.deepEqual(stats.hits, { count: 1, points: 4 })

const identical = {
  comparison: {
    note: 'No hits required - both strategies produce identical results',
    difference: 0,
    with_hits: { feasible: true, net_xp: 80, total_hits: 0, recommended: true },
    no_hits: { feasible: true, net_xp: 80, total_hits: 0, recommended: true },
  },
  with_hits: plan,
  no_hits: { ...plan },
}
assert.equal(plansAreIdentical(identical), true)
assert.equal(chooseStrategy(identical).key, 'with_hits')

const different = {
  comparison: {
    note: null,
    difference: 3.5,
    with_hits: { feasible: true, net_xp: 90, total_hits: 1, hit_cost: 4, recommended: true },
    no_hits: { feasible: true, net_xp: 86.5, total_hits: 0, hit_cost: 0, recommended: false },
  },
  with_hits: plan,
  no_hits: { ...plan, total_expected_points: 70 },
}
assert.equal(plansAreIdentical(different), false)
assert.equal(chooseStrategy(different).key, 'with_hits')

const tiedWithoutHits = {
  comparison: {
    note: null,
    difference: 0,
    with_hits: { feasible: true, net_xp: 80, total_hits: 1, recommended: false },
    no_hits: { feasible: true, net_xp: 80, total_hits: 0, recommended: true },
  },
  with_hits: plan,
  no_hits: { ...plan },
}
assert.equal(plansAreIdentical(tiedWithoutHits), false)
assert.equal(chooseStrategy(tiedWithoutHits).key, 'no_hits')

assert.equal(chooseStrategy({
  comparison: { note: 'The hit plan could not be solved for this squad.', difference: null, with_hits: { feasible: false }, no_hits: { feasible: false } },
  with_hits: { status: 'Infeasible', gameweek_plans: [], total_expected_points: 0 },
  no_hits: null,
}).plan, null)

assert.match(waitMessage(8), /few seconds/)
assert.match(waitMessage(15), /about a minute/)
assert.match(waitMessage(12), /few seconds/)
assert.match(waitMessage(12), /about a minute/)
assert.match(failureCopy('Optimization timed out. Try reducing the gameweek horizon.').detail, /timed out/i)
assert.equal(infeasibleCopy('Needs a shorter horizon').hints.length, 2)

console.log('planner format checks passed')
