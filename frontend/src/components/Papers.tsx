import { PAPERS } from '@/lib/papers'

export function Papers() {
  return (
    <section id="papers" aria-labelledby="papers-title" className="scroll-mt-20">
      <h2 id="papers-title" className="font-serif text-3xl">The six papers</h2>
      <p className="mt-3 max-w-prose text-lg text-muted-foreground">
        Every answer comes from these papers and nothing else. Together they trace how retrieval
        went from a fixed first step to something a model decides, checks and corrects.
      </p>

      <ol className="mt-8 divide-y border-y">
        {PAPERS.map((p) => (
          <li key={p.file} className="grid gap-x-8 gap-y-1 py-5 sm:grid-cols-[10rem_1fr]">
            <div>
              <div className="font-medium">{p.short}</div>
              <div className="text-sm text-muted-foreground tabular-nums">{p.year}</div>
            </div>
            <div>
              <a
                href={`https://arxiv.org/abs/${p.arxiv}`}
                target="_blank"
                rel="noreferrer"
                className="font-serif text-lg leading-snug underline-offset-4 hover:underline"
              >
                {p.title}
              </a>
              <p className="text-sm text-muted-foreground">{p.authors}, arXiv {p.arxiv}</p>
              <p className="mt-1.5 max-w-prose">{p.idea}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}
