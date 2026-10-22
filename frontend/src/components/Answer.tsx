import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { ExternalLink } from 'lucide-react'

import { IDK, type AskResponse, type Citation } from '@/lib/api'
import { highlightedSentences, splitSentences, tidyAnswer } from '@/lib/highlight'
import { arxivUrl, paperName } from '@/lib/papers'
import { cn } from '@/lib/utils'

/**
 * Turn citation markers into markdown links the renderer below understands:
 *   RAG answers:  [2]                       -> [2](#cite-2)
 *   CAG answers:  [corrective_rag.pdf p.3, p.5] -> [CRAG p.3, p.5](#paper=corrective_rag.pdf&page=3)
 * Fragment links survive react-markdown's URL sanitising; raw HTML never renders.
 */
function linkCitations(text: string): string {
  return text
    .replace(/\[([\w-]+\.pdf)((?:[\s,]*p\.\s*\d+)+)\]/g, (_, file: string, pages: string) => {
      const nums = pages.match(/\d+/g) ?? []
      return `[${paperName(file)} p.${nums.join(', ')}](#paper=${file}&page=${nums[0]})`
    })
    .replace(/\[(\d+)\]/g, '[$1](#cite-$1)')
}

function flash(id: string) {
  const el = document.getElementById(id)
  if (!el) return
  el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  el.classList.remove('animate-flash')
  void el.offsetWidth // restart the animation if clicked twice
  el.classList.add('animate-flash')
}

function Passage({ c, answer }: { c: Citation; answer: string }) {
  const marked = highlightedSentences(c.text, answer)
  const [open, setOpen] = useState(false)
  const sentences = splitSentences(c.text)
  // Collapsed, show the highlighted sentences with one sentence of context
  // on each side, so the evidence is visible without expanding.
  const first = Math.min(...marked)
  const last = Math.max(...marked)
  const from = marked.size ? Math.max(0, first - 1) : 0
  const to = marked.size ? Math.min(sentences.length, last + 2) : Math.min(sentences.length, 3)
  const visible = open ? sentences.map((s, i) => [s, i] as const) : sentences.slice(from, to).map((s, k) => [s, from + k] as const)
  const trimmedStart = !open && from > 0
  const trimmedEnd = !open && to < sentences.length
  return (
    <li id={`cite-${c.n}`} className="scroll-mt-24 rounded-lg border bg-card p-4">
      <div className="mb-2 flex items-baseline justify-between gap-3 text-sm">
        <span className="font-medium">
          <span className="mr-2 text-muted-foreground tabular-nums">[{c.n}]</span>
          {paperName(c.source)}, page {c.page}
        </span>
        <a
          href={arxivUrl(c.source, c.page)}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
        >
          Open on arXiv
          <ExternalLink className="size-3.5" aria-hidden />
        </a>
      </div>
      <blockquote className="font-serif text-[1.02rem] leading-relaxed text-foreground/85">
        {trimmedStart && <span className="text-muted-foreground">… </span>}
        {visible.map(([s, i]) =>
          marked.has(i) ? (
            <mark key={i} className="highlighter bg-transparent text-foreground">
              {s}
            </mark>
          ) : (
            <span key={i}>{s}</span>
          ),
        )}
        {trimmedEnd && <span className="text-muted-foreground"> …</span>}
      </blockquote>
      {(trimmedStart || trimmedEnd || open) && (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="mt-2 text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
        >
          {open ? 'Show less' : 'Show the whole passage'}
        </button>
      )}
    </li>
  )
}

export function Answer({ result }: { result: AskResponse }) {
  const idk = result.answer.trim() === IDK

  if (idk) {
    return (
      <section aria-labelledby="answer-title" className="rounded-xl border border-dashed p-6">
        <h2 id="answer-title" className="font-serif text-2xl">
          The papers don’t answer this
        </h2>
        <p className="mt-2 max-w-prose text-muted-foreground">
          Nothing in the six papers covers this question, so the system says so instead of guessing.
          Try asking about retrieval, grading, query routing, long contexts or reasoning agents.
        </p>
      </section>
    )
  }

  const text = tidyAnswer(result.answer)

  return (
    <section aria-labelledby="answer-title">
      <h2 id="answer-title" className="mb-4 border-b pb-3 font-serif text-xl">
        Answer
      </h2>
      <div
        className={cn(
          'max-w-prose font-serif text-[1.2rem] leading-[1.7] text-pretty',
          '[&_p]:mb-4 [&_strong]:font-semibold',
          '[&_ul]:mb-4 [&_ul]:list-disc [&_ul]:space-y-2 [&_ul]:pl-6 [&_ul_ul]:mt-2 [&_ol]:mb-4 [&_ol]:list-decimal [&_ol]:pl-6',
        )}
      >
        <ReactMarkdown
          components={{
            a({ href = '', children }) {
              if (href.startsWith('#cite-')) {
                return (
                  <button
                    type="button"
                    onClick={() => flash(href.slice(1))}
                    aria-label={`Show source ${String(children)}`}
                    className="relative -top-[0.45em] mx-0.5 rounded px-1 font-sans text-[0.7rem] font-semibold text-foreground/70 ring-1 ring-border hover:bg-highlight hover:text-foreground"
                  >
                    {children}
                  </button>
                )
              }
              if (href.startsWith('#paper=')) {
                const params = new URLSearchParams(href.slice(1))
                return (
                  <a
                    href={arxivUrl(params.get('paper') ?? '', Number(params.get('page')))}
                    target="_blank"
                    rel="noreferrer"
                    className="mx-0.5 rounded px-1 font-sans text-[0.72rem] font-medium whitespace-nowrap text-foreground/70 ring-1 ring-border hover:bg-highlight hover:text-foreground"
                  >
                    {children}
                  </a>
                )
              }
              return (
                <a href={href} target="_blank" rel="noreferrer" className="underline underline-offset-4">
                  {children}
                </a>
              )
            },
          }}
        >
          {linkCitations(text)}
        </ReactMarkdown>
      </div>

      {result.citations.length > 0 ? (
        <div className="mt-8">
          <h3 className="mb-1 font-serif text-lg">From the papers</h3>
          <p className="mb-4 text-sm text-muted-foreground">
            The passages the answer cites. Highlighted sentences are the ones it drew on.
          </p>
          <ol className="space-y-3">
            {result.citations.map((c) => (
              <Passage key={c.n} c={c} answer={text} />
            ))}
          </ol>
        </div>
      ) : (
        result.sources.length > 0 && (
          <p className="mt-6 text-sm text-muted-foreground">
            Read in full: {[...new Set(result.sources.map((s) => paperName(s.split(' ')[0])))].join(' and ')}.
            Page links above open the paper on arXiv.
          </p>
        )
      )}
    </section>
  )
}
