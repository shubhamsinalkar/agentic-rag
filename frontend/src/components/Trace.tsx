import type { Step } from '@/lib/api'
import { paperName } from '@/lib/papers'
import { cn } from '@/lib/utils'

function seconds(ms: number) {
  return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`
}

interface Described {
  title: string
  detail?: string
  quote?: string
  correction?: boolean // drawn in red pen
  done?: boolean
}

function describe(step: Step, searchesBefore: number): Described {
  switch (step.step) {
    case 'route':
      return step.label === 'off_topic'
        ? { title: 'Outside these papers', detail: 'Nothing to search, so nothing to make up.' }
        : {
            title: 'Read the question',
            detail:
              step.label === 'comparison'
                ? 'A comparison, so it gathers 8 passages instead of 4.'
                : `Looks like a ${step.label} question.`,
          }
    case 'search':
      return {
        title: searchesBefore === 0 ? 'Searched the papers' : 'Searched again',
        quote: step.query,
        detail: `${step.found} new passage${step.found === 1 ? '' : 's'} found`,
      }
    case 'check':
      return step.sufficient
        ? {
            title: 'Enough to answer',
            detail:
              step.relevant === 1
                ? '1 passage actually answers the question.'
                : `${step.relevant} passages actually answer the question.`,
            done: true,
          }
        : { title: 'Not enough yet', detail: step.missing, correction: true }
    case 'rewrite':
      return { title: 'Rewrote the search', quote: step.query, correction: true }
    case 'pick':
      return {
        title: step.papers.length ? 'Chose papers to read in full' : 'No paper covers this',
        detail: step.papers.map(paperName).join(' and ') || undefined,
      }
    case 'answer':
      return {
        title: 'Wrote the answer',
        detail:
          step.cited > 0
            ? `Citing ${step.cited} source${step.cited === 1 ? '' : 's'}.`
            : 'The papers don’t answer this, so it says so.',
      }
  }
}

/** A loose, hand-drawn arrow curling back up: "search again". */
function LoopArrow() {
  return (
    <svg viewBox="0 0 60 88" aria-hidden className="h-20 w-14 text-pen">
      <path
        d="M8 80 C 46 74, 52 40, 34 20 C 28 13, 20 10, 12 9"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path d="M19 3 L 10 9 L 18 15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function Trace({ steps, totalSeconds }: { steps: Step[]; totalSeconds: number }) {
  let searches = 0
  const items = steps.map((s) => {
    const d = describe(s, searches)
    if (s.step === 'search') searches += 1
    return { step: s, ...d }
  })
  const rewrites = steps.filter((s) => s.step === 'rewrite').length

  return (
    <section aria-labelledby="trace-title">
      <div className="mb-4 flex items-baseline justify-between gap-4 border-b pb-3">
        <h2 id="trace-title" className="font-serif text-xl">
          What it did
        </h2>
        <p className="text-sm text-muted-foreground tabular-nums">
          {seconds(totalSeconds * 1000)}
          {rewrites > 0 && `, corrected ${rewrites === 1 ? 'once' : `${rewrites} times`}`}
        </p>
      </div>

      <ol className="relative">
        {items.map((it, i) => {
          const last = i === items.length - 1
          return (
            <li
              key={i}
              // Steps appear one after another: the page's one orchestrated
              // motion. Disabled by the reduced-motion rule in index.css.
              style={{ animationDelay: `${i * 120}ms` }}
              className="relative grid animate-step grid-cols-[1.25rem_1fr] gap-x-3 pb-5"
            >
              {/* the rail joining the steps */}
              {!last && (
                <span
                  aria-hidden
                  className={cn(
                    'absolute top-5 bottom-0 left-[0.56rem] w-px',
                    it.correction ? 'bg-pen/40' : 'bg-border',
                  )}
                />
              )}
              <span
                aria-hidden
                className={cn(
                  'mt-1.5 size-3 justify-self-center rounded-full border-2',
                  it.correction
                    ? 'border-pen bg-pen'
                    : it.done
                      ? 'border-foreground bg-foreground'
                      : 'border-foreground bg-card',
                )}
              />
              <div className="min-w-0">
                <div className="flex items-baseline justify-between gap-3">
                  <h3 className={cn('font-medium', it.correction && 'text-pen')}>{it.title}</h3>
                  <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
                    {seconds(it.step.ms)}
                  </span>
                </div>
                {it.quote && (
                  <p
                    className={cn(
                      'mt-1 font-serif text-[0.95rem] leading-snug',
                      it.correction ? 'text-pen' : 'text-foreground/85',
                    )}
                  >
                    “{it.quote}”
                  </p>
                )}
                {it.detail && (
                  <p
                    className={cn(
                      'mt-1 text-sm leading-relaxed text-pretty',
                      it.correction ? 'border-l-2 border-pen/50 pl-3 text-pen' : 'text-muted-foreground',
                    )}
                  >
                    {it.detail}
                  </p>
                )}
                {it.step.step === 'rewrite' && (
                  <div className="mt-1 flex items-end gap-1" aria-hidden>
                    <LoopArrow />
                    <span className="mb-10 -rotate-6 font-hand text-2xl leading-none text-pen">
                      try again
                    </span>
                  </div>
                )}
              </div>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
