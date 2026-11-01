import { API_URL, api } from './api'

/*
 * Anonymous usage events, logged by the API (/api/telemetry) to Cloud Logging.
 * No cookies and no manager IDs: a random ID in sessionStorage ties one visit's
 * events together. Off under Do Not Track or Global Privacy Control. Every call
 * swallows its own errors, so telemetry can never break the page.
 */

const SESSION_KEY = 'fpl-telemetry-session'
const MAX_BATCH = 20
const FLUSH_DELAY_MS = 4000

const disabled = (() => {
  try {
    return navigator.doNotTrack === '1' || window.doNotTrack === '1' || navigator.globalPrivacyControl === true
  } catch {
    return true
  }
})()

let session = null
let queue = []
let timer = null
let inFlight = 0

function sessionId() {
  if (session) return session
  const fresh = () => (crypto.randomUUID?.() || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`)
  try {
    session = sessionStorage.getItem(SESSION_KEY)
    if (!session) {
      session = fresh()
      sessionStorage.setItem(SESSION_KEY, session)
    }
  } catch {
    session = fresh()
  }
  return session
}

// The API serves one request per instance and runs at most two, so a telemetry
// post during a slow plan would take the second instance. Wait for API calls to finish.
api.interceptors.request.use(config => {
  inFlight += 1
  return config
})
api.interceptors.response.use(
  res => { inFlight = Math.max(0, inFlight - 1); return res },
  err => { inFlight = Math.max(0, inFlight - 1); return Promise.reject(err) },
)

function send(events) {
  try {
    // text/plain is a simple request: no CORS preflight, so one round trip.
    fetch(`${API_URL}/api/telemetry`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: JSON.stringify({ session: sessionId(), events }),
      credentials: 'omit',
      keepalive: true,
    }).catch(() => {})
  } catch {
    // Telemetry is best effort.
  }
}

function flush(force = false) {
  clearTimeout(timer)
  timer = null
  if (!queue.length) return
  if (inFlight > 0 && !force) {
    schedule()
    return
  }
  while (queue.length) send(queue.splice(0, MAX_BATCH))
}

function schedule() {
  if (!timer) timer = setTimeout(() => flush(), FLUSH_DELAY_MS)
}

if (!disabled) {
  try {
    // Send what is left when the tab is hidden or closed; keepalive lets it finish.
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') flush(true)
    })
    window.addEventListener('pagehide', () => flush(true))
  } catch {
    // No document (tests, prerender).
  }
}

/** Record an event. Props are small flat values: strings, numbers, booleans. */
export function track(event, props = {}) {
  if (disabled) return
  try {
    queue.push({ event, props, t: Math.round(performance.now()) })
    if (queue.length >= MAX_BATCH) flush()
    else schedule()
  } catch {
    // Telemetry is best effort.
  }
}

/** Short, ID-free text for an error event. The server also scrubs long numbers. */
export function errorKind(err) {
  const status = err?.response?.status
  if (status) return `http_${status}`
  if (err?.code === 'ERR_NETWORK') return 'network'
  return err?.name || 'error'
}
