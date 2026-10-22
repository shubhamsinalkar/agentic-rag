import benchmarkChart from '../../../docs/benchmark.png'

// From notebooks/02_benchmark.ipynb (30 questions, human-audited grades).
const ROWS = [
  { name: 'Plain RAG', correct: 20, median: 1.8, cost: '$0.5' },
  { name: 'Agentic RAG', correct: 24, median: 3.2, cost: '$2.0' },
  { name: 'Whole paper', correct: 30, median: 2.8, cost: '$8.3' },
]

export function Results() {
  return (
    <section id="results" aria-labelledby="results-title" className="scroll-mt-20">
      <h2 id="results-title" className="font-serif text-3xl">What the benchmark found</h2>
      <p className="mt-3 max-w-prose text-lg text-muted-foreground">
        All three strategies answered the same 30 questions, each checked against a reference answer
        written from the papers.
      </p>

      <div className="mt-8 grid items-start gap-10 lg:grid-cols-[1fr_20rem]">
        <figure>
          <img
            src={benchmarkChart}
            alt="Bar charts. Answers judged correct: plain RAG 67 percent, agentic RAG 80 percent, whole paper 100 percent. Median latency: 1.8, 3.2 and 2.8 seconds."
            className="w-full rounded-xl border bg-white p-3"
            loading="lazy"
          />
          <figcaption className="mt-3 text-sm text-muted-foreground">
            Answers graded by a separate model and every verdict audited by hand. Latency measured on
            Gemini’s free tier in the morning; it is slower at peak hours.
          </figcaption>
        </figure>

        <div>
          <table className="w-full text-left text-sm">
            <caption className="sr-only">Benchmark summary</caption>
            <thead className="text-muted-foreground">
              <tr className="border-b">
                <th scope="col" className="py-2 font-medium">Strategy</th>
                <th scope="col" className="py-2 text-right font-medium">Correct</th>
                <th scope="col" className="py-2 text-right font-medium">Median</th>
                <th scope="col" className="py-2 text-right font-medium">Per 1k</th>
              </tr>
            </thead>
            <tbody className="tabular-nums">
              {ROWS.map((r) => (
                <tr key={r.name} className="border-b">
                  <th scope="row" className="py-2.5 font-medium">{r.name}</th>
                  <td className="py-2.5 text-right">{r.correct}/30</td>
                  <td className="py-2.5 text-right">{r.median} s</td>
                  <td className="py-2.5 text-right">{r.cost}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-6 space-y-4 text-[0.95rem] leading-relaxed">
            <p>
              <span className="font-medium">Reading whole papers won</span> because six papers fit in
              the prompt. It sends about 25 times more text per question and stops working once a
              collection outgrows the model’s context.
            </p>
            <p>
              <span className="font-medium">Checking before answering</span> fixed five questions
              plain search got wrong and broke one. With 30 questions that’s promising, not proof.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
