// Reads the transfer explainer's markdown (backend transfer_explainer.py).

// Plain words for the explainer's shorthand: no colour dots, "difficulty 3/5" for "FDR 3".
const clean = s => s
  .replace(/\*\*/g, '')
  .replace(/[\u{1F7E0}-\u{1F7EB}\u{1F534}\u{1F535}]/gu, '')
  .replace(/\(avg FDR: ([\d.]+)\)/g, '(average difficulty $1 of 5)')
  .replace(/FDR (\d)/g, 'difficulty $1/5')
  .replace(/\s+→\s+/g, ' → ')
  .trim()

// Default lines that say nothing.
const NOISE = /^Risks: No significant risks identified\.?$/

/**
 * Split the explainer's markdown into per-transfer blocks so each block's reasoning
 * can sit under the structured in/out row it describes.
 */
export function parseBlocks(explanation) {
  const blocks = []
  let block = null
  let side = null
  for (const raw of (explanation || '').split('\n')) {
    const line = raw.trim()
    if (line.startsWith('### Transfer')) {
      block = { sell: null, buy: null }
      blocks.push(block)
      side = null
    } else if (block && line.includes('**SELL:**')) {
      side = block.sell = { summary: clean(line.replace('**SELL:**', '')), details: [] }
    } else if (block && line.includes('**BUY:**')) {
      side = block.buy = { summary: clean(line.replace('**BUY:**', '')), details: [] }
    } else if (side && line.startsWith('└')) {
      const detail = clean(line.replace(/^└\s*/, ''))
      if (!NOISE.test(detail)) side.details.push(detail)
    } else if (line.startsWith('---')) {
      block = side = null
    }
  }
  return blocks
}

/** The buy reason for one incoming player, if the explainer gave one. */
export function buyReason(explanation, name) {
  return parseBlocks(explanation).find(b => b.buy?.summary.startsWith(name))?.buy || null
}

/** "this week LEE (A), difficulty 3/5, ..." from "Van Hecke (...) is recommended because of this week ...". */
export function shortReason(summary) {
  if (!summary) return null
  const why = summary.split(/ because of | - /)[1] || ''
  const text = why.replace(/\.$/, '').trim()
  return text ? text[0].toUpperCase() + text.slice(1) : null
}
