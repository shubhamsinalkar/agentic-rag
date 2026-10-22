import { useEffect, useRef, useState } from 'react'

import { Answer } from '@/components/Answer'
import { AskForm, STRATEGIES } from '@/components/AskForm'
import { HowItWorks } from '@/components/HowItWorks'
import { Papers } from '@/components/Papers'
import { Results } from '@/components/Results'
import { Trace } from '@/components/Trace'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ask, AskError, type AskResponse, type Strategy } from '@/lib/api'
import { EXAMPLE, EXAMPLE_QUESTION } from '@/lib/example'

type Status =
  | { kind: 'example' }
  | { kind: 'loading'; question: string; strategy: Strategy }
  | { kind: 'done'; question: string; result: AskResponse }
  | { kind: 'error'; message: string; retryable: boolean }

const WORKING: Record<Strategy, string> = {
  agentic_rag: 'Searching the papers and checking what it finds',
  plain_rag: 'Searching the papers',
  cag: 'Choosing papers and reading them in full',
}

function Elapsed() {
  const [s, setS] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setS((x) => x + 1), 1000)
    return () => clearInterval(t)
  }, [])
  return <span className="tabular-nums">{s} s</span>
}

export default function App() {
  const [question, setQuestion] = useState('')
  const [strategy, setStrategy] = useState<Strategy>('agentic_rag')
  const [status, setStatus] = useState<Status>({ kind: 'example' })
  const inflight = useRef<AbortController | null>(null)
  const resultRef = useRef<HTMLDivElement>(null)

  async function onAsk(override?: string) {
    const q = (override ?? question).trim()
    if (q.length < 3) return
    inflight.current?.abort()
    const ctrl = new AbortController()
    inflight.current = ctrl
    setStatus({ kind: 'loading', question: q, strategy })
    requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
    try {
      const result = await ask(q, strategy, ctrl.signal)
      setStatus({ kind: 'done', question: q, result })
    } catch (err) {
      if ((err as Error).name === 'AbortError') return
      const e = err as AskError
      setStatus({ kind: 'error', message: e.message, retryable: e.status !== 422 })
    }
  }

  const shown = status.kind === 'example' ? { question: EXAMPLE_QUESTION, result: EXAMPLE } : status.kind === 'done' ? status : null

  return (
    <TooltipProvider delayDuration={300}>
      <a
        href="#ask"
        className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:rounded focus:bg-card focus:px-3 focus:py-2"
      >
        Skip to the question box
      </a>

      <header className="sticky top-0 z-40 border-b bg-background/85 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6">
          <a href="#top" className="font-serif text-lg font-semibold tracking-tight">
            Agentic RAG
          </a>
          <nav aria-label="Sections" className="flex gap-5 text-sm text-muted-foreground">
            <a href="#how" className="hover:text-foreground">How it works</a>
            <a href="#results" className="hover:text-foreground">Results</a>
            <a href="#papers" className="hidden hover:text-foreground sm:inline">Papers</a>
          </nav>
        </div>
      </header>

      <main id="top" className="mx-auto max-w-6xl px-4 sm:px-6">
        <section id="ask" className="grid gap-10 pt-14 pb-12 lg:grid-cols-[1fr_minmax(0,34rem)] lg:pt-20">
          <div className="max-w-xl">
            <h1 className="font-serif text-[clamp(2.6rem,6vw,4.4rem)] leading-[1.02] font-medium tracking-[-0.02em] text-balance">
              Answers you can check
            </h1>
            <p className="mt-6 text-lg leading-relaxed text-pretty text-muted-foreground">
              Ask about six landmark papers on retrieval-augmented generation. An agent searches
              them, judges whether what it found really answers you, and searches again when it
              doesn’t. Every step and every quoted passage is shown.
            </p>
          </div>
          <div className="lg:pt-2">
            <AskForm
              question={question}
              strategy={strategy}
              busy={status.kind === 'loading'}
              onQuestion={setQuestion}
              onStrategy={setStrategy}
              onAsk={onAsk}
            />
          </div>
        </section>

        <div ref={resultRef} className="scroll-mt-20 border-t pt-10 pb-20" aria-live="polite">
          {status.kind === 'loading' && (
            <div className="rounded-xl border border-dashed p-6">
              <p className="font-serif text-2xl text-balance">“{status.question}”</p>
              <p className="mt-3 flex items-center gap-3 text-muted-foreground">
                <span className="relative flex size-2.5" aria-hidden>
                  <span className="absolute inline-flex size-full animate-ping rounded-full bg-foreground/40" />
                  <span className="relative inline-flex size-2.5 rounded-full bg-foreground" />
                </span>
                {WORKING[status.strategy]}, <Elapsed />
              </p>
              <p className="mt-2 text-sm text-muted-foreground">
                Usually a few seconds. Google’s free AI tier can take up to a minute at busy times.
              </p>
            </div>
          )}

          {status.kind === 'error' && (
            <div role="alert" className="rounded-xl border border-pen/40 p-6">
              <p className="font-medium text-pen">{status.message}</p>
              {status.retryable && (
                <button
                  type="button"
                  onClick={() => onAsk()}
                  className="mt-3 text-sm underline underline-offset-4"
                >
                  Ask again
                </button>
              )}
            </div>
          )}

          {shown && (
            <>
              <div className="mb-8 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
                <p className="font-serif text-2xl text-balance sm:text-3xl">“{shown.question}”</p>
                <p className="text-sm text-muted-foreground">
                  {status.kind === 'example'
                    ? `An earlier run with ${STRATEGIES.find((s) => s.id === shown.result.strategy)?.name}. Ask your own question above.`
                    : `Answered with ${STRATEGIES.find((s) => s.id === shown.result.strategy)?.name}`}
                </p>
              </div>
              <div className="grid gap-12 lg:grid-cols-[minmax(0,22rem)_1fr]">
                {/* Phones: answer first, then how it was reached. Desktop: side by side. */}
                <div className="order-2 lg:order-1">
                  <Trace steps={shown.result.steps} totalSeconds={shown.result.latency_s} />
                </div>
                <div className="order-1 lg:order-2">
                  <Answer result={shown.result} />
                </div>
              </div>
            </>
          )}
        </div>

        <div className="space-y-24 border-t py-20">
          <HowItWorks />
          <Results />
          <Papers />
        </div>
      </main>

      <footer className="border-t">
        <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-8 text-sm text-muted-foreground sm:flex-row sm:justify-between sm:px-6">
          <p>Built with LangGraph, ChromaDB, Gemini, scikit-learn and FastAPI.</p>
          <p>Answers come only from the six papers above.</p>
        </div>
      </footer>
    </TooltipProvider>
  )
}
