import axios from 'axios'

// Production API URL handling
export const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
axios.defaults.baseURL = API_URL

export const api = axios

export function errorMessage(err) {
  if (axios.isAxiosError(err)) return err.response?.data?.detail || err.message
  return err?.message || String(err)
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
