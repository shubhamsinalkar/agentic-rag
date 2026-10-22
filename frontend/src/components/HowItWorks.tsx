import { STRATEGIES } from '@/components/AskForm'

/** Schematic of the agent graph (app/graph.py), drawn in the page's ink + red pen. */
function GraphDiagram() {
  const box = 'fill-card stroke-foreground'
  const label = 'fill-foreground text-[15px] font-medium'
  const note = 'fill-muted-foreground text-[13px]'
  return (
    <svg
      viewBox="0 0 860 300"
      role="img"
      aria-labelledby="diagram-title diagram-desc"
      className="h-auto w-full font-sans"
    >
      <title id="diagram-title">How the agent answers a question</title>
      <desc id="diagram-desc">
        The question is routed. Off-topic questions skip straight to the answer. Otherwise the agent
        searches, checks whether the passages are enough, and if not rewrites the search and tries
        again, at most twice, before answering.
      </desc>
      <defs>
        <marker id="ink-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0 0 L10 5 L0 10 z" className="fill-foreground" />
        </marker>
        <marker id="pen-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0 0 L10 5 L0 10 z" className="fill-pen" />
        </marker>
      </defs>

      {/* question */}
      <text x="20" y="150" className={label}>Question</text>

      {/* main line */}
      {[
        [96, 150, 150, 150],
        [262, 150, 316, 150],
        [428, 150, 482, 150],
        [594, 150, 690, 150],
      ].map(([x1, y1, x2, y2], i) => (
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} className="stroke-foreground" strokeWidth="1.5" markerEnd="url(#ink-arrow)" />
      ))}

      {[
        [150, 'Route', 'scikit-learn, 1 ms'],
        [316, 'Search', 'Chroma, top 4'],
        [482, 'Check', 'enough?'],
        [690, 'Answer', 'cites passages'],
      ].map(([x, title, sub]) => (
        <g key={title as string}>
          <rect x={x as number} y="122" width="112" height="56" rx="8" className={box} strokeWidth="1.5" />
          <text x={(x as number) + 56} y="146" textAnchor="middle" className={label}>{title}</text>
          <text x={(x as number) + 56} y="166" textAnchor="middle" className={note}>{sub}</text>
        </g>
      ))}
      <text x="642" y="140" textAnchor="middle" className={note}>yes</text>

      {/* off-topic shortcut */}
      <path d="M206 122 C 206 50, 746 50, 746 118" fill="none" className="stroke-muted-foreground" strokeWidth="1.5" strokeDasharray="5 5" markerEnd="url(#ink-arrow)" />
      <text x="476" y="62" textAnchor="middle" className={note}>off-topic: no search, says it doesn’t know</text>

      {/* the correction loop, in red pen */}
      <path d="M538 178 L 538 236" fill="none" className="stroke-pen" strokeWidth="1.75" markerEnd="url(#pen-arrow)" />
      <text x="548" y="212" className="fill-pen text-[13px]">no</text>
      <rect x="482" y="240" width="112" height="48" rx="8" className="fill-card stroke-pen" strokeWidth="1.75" />
      <text x="538" y="269" textAnchor="middle" className="fill-pen text-[15px] font-medium">Rewrite</text>
      <path d="M482 264 C 380 264, 372 230, 372 182" fill="none" className="stroke-pen" strokeWidth="1.75" markerEnd="url(#pen-arrow)" />
      <text x="300" y="250" textAnchor="middle" className="fill-pen font-hand text-[22px]">at most twice</text>
    </svg>
  )
}

export function HowItWorks() {
  return (
    <section id="how" aria-labelledby="how-title" className="scroll-mt-20">
      <h2 id="how-title" className="font-serif text-3xl">How it works</h2>
      <p className="mt-3 max-w-prose text-lg text-muted-foreground">
        Plain retrieval hands the model whatever the search returns. This agent first asks whether
        those passages actually answer the question, and when they don’t, it rewrites the search
        around what’s missing.
      </p>

      <figure className="mt-8 overflow-x-auto rounded-xl border bg-card p-4 sm:p-6">
        <div className="min-w-[640px]">
          <GraphDiagram />
        </div>
      </figure>

      <dl className="mt-10 grid gap-8 sm:grid-cols-3">
        {STRATEGIES.map((s) => (
          <div key={s.id} className="border-t-2 border-foreground pt-3">
            <dt className="font-medium">{s.name}</dt>
            <dd className="mt-1 text-muted-foreground">{s.how}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
