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
    file: 'chexnet.pdf',
    short: 'CheXNet',
    title: 'CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning',
    authors: 'Rajpurkar et al.',
    year: 2017,
    arxiv: '1711.05225v3',
    idea: 'Medicine: a neural network reads chest X-rays and spots pneumonia better than the average radiologist.',
  },
  {
    file: 'four_degrees.pdf',
    short: 'Four Degrees',
    title: 'Four Degrees of Separation',
    authors: 'Backstrom et al.',
    year: 2012,
    arxiv: '1111.4570v3',
    idea: 'Social networks: any two Facebook users are on average just 3.74 friends apart, not six.',
  },
  {
    file: 'generative_ai_at_work.pdf',
    short: 'Generative AI at Work',
    title: 'Generative AI at Work',
    authors: 'Brynjolfsson, Li & Raymond',
    year: 2023,
    arxiv: '2304.11771v2',
    idea: 'Economics: an AI assistant made customer support agents 15% more productive, newcomers most of all.',
  },
  {
    file: 'gravitational_waves.pdf',
    short: 'Gravitational Waves',
    title: 'Observation of Gravitational Waves from a Binary Black Hole Merger',
    authors: 'LIGO and Virgo Collaborations',
    year: 2016,
    arxiv: '1602.03837v1',
    idea: 'Physics: the first direct detection of gravitational waves, from two black holes merging.',
  },
  {
    file: 'planet_nine.pdf',
    short: 'Planet Nine',
    title: 'Evidence for a Distant Giant Planet in the Solar System',
    authors: 'Batygin & Brown',
    year: 2016,
    arxiv: '1601.05438v1',
    idea: 'Astronomy: strangely aligned orbits at the edge of the solar system point to an unseen giant planet.',
  },
  {
    file: 'transformer.pdf',
    short: 'Transformer',
    title: 'Attention Is All You Need',
    authors: 'Vaswani et al.',
    year: 2017,
    arxiv: '1706.03762v7',
    idea: 'Computer science: the Transformer, a model built only on attention, which powers modern chatbots.',
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
