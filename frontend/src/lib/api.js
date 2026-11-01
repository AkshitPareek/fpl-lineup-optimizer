import axios from 'axios'

// Production API URL handling
export const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
axios.defaults.baseURL = API_URL

export const api = axios

export function errorMessage(err) {
  if (axios.isAxiosError(err)) return err.response?.data?.detail || err.message
  return err?.message || String(err)
}

/** Whether a failed call is worth one quiet retry: no answer at all, or a host still starting (503). */
function transient(err) {
  if (axios.isCancel(err) || err?.code === 'ECONNABORTED') return false
  return !err?.response || err.response.status === 503
}

/** Call fn(); on a network error or 503 (often a cold start) wait a moment and try once more. */
export async function withRetry(fn, delayMs = 1500) {
  try {
    return await fn()
  } catch (err) {
    if (!transient(err)) throw err
    await new Promise(r => setTimeout(r, delayMs))
    return fn()
  }
}

/**
 * A plain message for a failed call, the raw detail for a "Details" toggle, and
 * whether a shorter plan might help.
 */
export function friendlyError(err) {
  const status = err?.response?.status
  const detail = errorMessage(err)
  if (axios.isCancel?.(err) || err?.name === 'CanceledError') {
    return { kind: 'cancelled', message: 'Cancelled.', detail: null }
  }
  if (err?.code === 'ECONNABORTED' || status === 504) {
    return { kind: 'timeout', message: 'The plan took too long. A shorter horizon solves faster.', detail, shorter: true }
  }
  if (!err?.response) {
    return { kind: 'network', message: "Couldn't reach the server. Check your connection and try again.", detail }
  }
  if (status === 503 || status === 429) {
    return { kind: `http_${status}`, message: 'The server is busy or starting up. Try again in a few seconds.', detail }
  }
  if (status === 404) {
    return { kind: 'http_404', message: 'FPL has no team with this ID.', detail }
  }
  if (status === 422) {
    return { kind: 'http_422', message: 'Some settings were not accepted. Check them and try again.', detail: typeof detail === 'string' ? detail : JSON.stringify(detail) }
  }
  return { kind: status ? `http_${status}` : 'error', message: 'Something went wrong while planning. Try again.', detail }
}

/**
 * POST to a server-sent-events endpoint and call onEvent for every `data:` frame.
 * Resolves when the stream closes.
 */
export async function streamEvents(path, body, onEvent) {
  const response = await fetch(`${API_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(response.statusText)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const frames = buffer.split('\n\n')
    buffer = frames.pop() || ''

    for (const frame of frames) {
      const line = frame.trim()
      if (!line.startsWith('data: ')) continue
      try {
        onEvent(JSON.parse(line.substring(6)))
      } catch (e) {
        console.error('JSON parse error', e)
      }
    }
  }
}
