// The six papers in the corpus (pinned in app/corpus.py).

export interface Paper {
  file: string
  short: string
  title: string
  authors: string
  year: number
  arxiv: string
  idea: string
}

export const PAPERS: Paper[] = [
  {
    file: 'rag_lewis2020.pdf',
    short: 'RAG',
    title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
    authors: 'Lewis et al.',
    year: 2020,
    arxiv: '2005.11401v4',
    idea: 'The original recipe: a retriever finds Wikipedia passages, a generator answers from them.',
  },
  {
    file: 'self_rag.pdf',
    short: 'Self-RAG',
    title: 'Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection',
    authors: 'Asai et al.',
    year: 2023,
    arxiv: '2310.11511v1',
    idea: 'The model decides when to retrieve and grades its own passages with reflection tokens.',
  },
  {
    file: 'corrective_rag.pdf',
    short: 'CRAG',
    title: 'Corrective Retrieval Augmented Generation',
    authors: 'Yan et al.',
    year: 2024,
    arxiv: '2401.15884v3',
    idea: 'A small evaluator scores retrieved documents and triggers a correction when they fall short.',
  },
  {
    file: 'adaptive_rag.pdf',
    short: 'Adaptive-RAG',
    title: 'Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity',
    authors: 'Jeong et al.',
    year: 2024,
    arxiv: '2403.14403v2',
    idea: 'A classifier routes each question to no retrieval, one search, or several.',
  },
  {
    file: 'lost_in_the_middle.pdf',
    short: 'Lost in the Middle',
    title: 'Lost in the Middle: How Language Models Use Long Contexts',
    authors: 'Liu et al.',
    year: 2023,
    arxiv: '2307.03172v3',
    idea: 'Models use facts at the start and end of a long prompt far better than facts in the middle.',
  },
  {
    file: 'react.pdf',
    short: 'ReAct',
    title: 'ReAct: Synergizing Reasoning and Acting in Language Models',
    authors: 'Yao et al.',
    year: 2022,
    arxiv: '2210.03629v3',
    idea: 'A model alternates between reasoning and actions such as searching Wikipedia.',
  },
]

const BY_FILE = Object.fromEntries(PAPERS.map((p) => [p.file, p]))

export function paperName(file: string): string {
  return BY_FILE[file]?.short ?? file.replace(/\.pdf$/, '')
}

export function arxivUrl(file: string, page?: number): string | undefined {
  const p = BY_FILE[file]
  if (!p) return undefined
  return `https://arxiv.org/pdf/${p.arxiv}${page ? `#page=${page}` : ''}`
}
