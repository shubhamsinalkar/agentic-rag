// Mirrors app/main.py (AskRequest / AskResponse).

export type Strategy = 'agentic_rag' | 'plain_rag' | 'cag'

export type Step =
  | { step: 'route'; label: string; ms: number }
  | { step: 'search'; query: string; found: number; ms: number }
  | { step: 'check'; sufficient: boolean; relevant: number; missing: string; ms: number }
  | { step: 'rewrite'; query: string; ms: number }
  | { step: 'pick'; papers: string[]; ms: number }
  | { step: 'answer'; cited: number; ms: number }

export interface Citation {
  n: number
  source: string
  page: number
  text: string
}

export interface AskResponse {
  answer: string
  sources: string[]
  strategy: Strategy
  route: string | null
  retries: number
  latency_s: number
  steps: Step[]
  citations: Citation[]
}

export const IDK = "I don't know based on the provided documents."

export class AskError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

export async function ask(question: string, strategy: Strategy, signal?: AbortSignal) {
  let res: Response
  try {
    res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, strategy }),
      signal,
    })
  } catch (err) {
    if ((err as Error).name === 'AbortError') throw err
    throw new AskError("Can't reach the server. Check your connection and try again.", 0)
  }
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    // The API writes user-facing messages for 429/503; validation errors
    // (422) come as a list, so fall back to a plain sentence for those.
    const detail =
      typeof body.detail === 'string'
        ? body.detail
        : res.status === 422
          ? 'Questions need between 3 and 500 characters.'
          : `The server returned an error (${res.status}). Try again.`
    throw new AskError(detail, res.status)
  }
  return body as AskResponse
}
