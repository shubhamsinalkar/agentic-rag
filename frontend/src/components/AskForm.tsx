import { useRef } from 'react'
import { CornerDownLeft, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { Strategy } from '@/lib/api'

export const STRATEGIES: { id: Strategy; name: string; how: string }[] = [
  {
    id: 'agentic_rag',
    name: 'Agentic RAG',
    how: 'Searches, checks whether the passages answer you, and searches again if they don’t (up to twice).',
  },
  {
    id: 'plain_rag',
    name: 'Plain RAG',
    how: 'Searches once and answers from the four closest passages, without checking them.',
  },
  {
    id: 'cag',
    name: 'Whole paper',
    how: 'Skips search: picks one or two papers and reads them in full (cache-augmented generation).',
  },
]

const EXAMPLES = [
  'How does CRAG decide a retrieved document is incorrect?',
  'How do CRAG and Self-RAG differ in judging retrieved documents?',
  'Why does RAG keep the document encoder fixed during training?',
  'Who won the 2022 FIFA World Cup?',
]

interface Props {
  question: string
  strategy: Strategy
  busy: boolean
  onQuestion: (q: string) => void
  onStrategy: (s: Strategy) => void
  onAsk: (q?: string) => void
}

export function AskForm({ question, strategy, busy, onQuestion, onStrategy, onAsk }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null)
  const tooShort = question.trim().length < 3

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault()
        if (!busy && !tooShort) onAsk()
      }}
      className="rounded-xl border bg-card shadow-[0_1px_0_var(--border),0_12px_32px_-18px_rgb(22_33_58/0.25)]"
    >
      <label htmlFor="question" className="sr-only">
        Your question
      </label>
      <Textarea
        id="question"
        ref={ref}
        value={question}
        maxLength={500}
        onChange={(e) => onQuestion(e.target.value)}
        onKeyDown={(e) => {
          // Enter asks; Shift+Enter adds a new line.
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            if (!busy && !tooShort) onAsk()
          }
        }}
        placeholder="Ask about retrieval-augmented generation…"
        className="min-h-24 resize-none rounded-b-none border-0 bg-transparent px-5 pt-5 font-serif text-xl leading-snug shadow-none focus-visible:ring-0 md:text-xl dark:bg-transparent"
      />

      <div className="flex flex-wrap items-center gap-3 border-t px-4 py-3">
        <Tabs value={strategy} onValueChange={(v) => onStrategy(v as Strategy)}>
          <TabsList aria-label="How to answer">
            {STRATEGIES.map((s) => (
              <Tooltip key={s.id}>
                <TooltipTrigger asChild>
                  <TabsTrigger
                    value={s.id}
                    // The tooltip wrapper overwrites data-state with its own
                    // ("closed"), so style the selected tab via aria-selected.
                    className="px-3 aria-selected:bg-card aria-selected:text-foreground aria-selected:shadow-sm dark:aria-selected:bg-input/40"
                  >
                    {s.name}
                  </TabsTrigger>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-64 text-pretty">
                  {s.how}
                </TooltipContent>
              </Tooltip>
            ))}
          </TabsList>
        </Tabs>

        <Button type="submit" size="lg" disabled={busy || tooShort} className="ml-auto h-9 px-4">
          {busy ? (
            <>
              <Loader2 className="animate-spin" aria-hidden />
              Answering
            </>
          ) : (
            <>
              Ask
              <CornerDownLeft aria-hidden className="opacity-60" />
            </>
          )}
        </Button>
      </div>

      <div className="border-t border-dashed px-4 pt-3 pb-4">
        <p className="mb-2 text-sm text-muted-foreground">Try a question from the benchmark</p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((q) => (
            <button
              key={q}
              type="button"
              disabled={busy}
              onClick={() => {
                onQuestion(q)
                onAsk(q)
              }}
              className="rounded-md border border-transparent bg-secondary px-3 py-1.5 text-left text-sm text-secondary-foreground transition-colors hover:border-input disabled:opacity-50"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </form>
  )
}
