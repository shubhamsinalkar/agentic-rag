// Which sentences of a quoted passage did the answer actually use?
// Scored by shared content words: cheap, transparent, and good enough to
// draw the reader's eye (it's a highlighter, not a fact checker).

const STOP = new Set(
  'the a an and or of to in on for with by is are was were be been being as at from that this these those it its into than then there their they which who whom what when where how why can could would should may might will shall do does did not no also such each other more most only both all any some our we you your'.split(' '),
)

function words(text: string): string[] {
  return (text.toLowerCase().match(/[a-z][a-z0-9-]{2,}/g) ?? []).filter((w) => !STOP.has(w))
}

export function splitSentences(text: string): string[] {
  // Keep the punctuation and trailing space with each sentence so joining
  // the pieces gives back the original text exactly.
  return text.match(/[^.!?]+(?:[.!?]+(?:\s+|$)|$)/g) ?? [text]
}

export function highlightedSentences(passage: string, answer: string): Set<number> {
  const answerWords = new Set(words(answer))
  const sentences = splitSentences(passage)
  const scores = sentences.map((s) => {
    const w = words(s)
    if (w.length === 0) return 0
    const shared = new Set(w.filter((x) => answerWords.has(x))).size
    return shared >= 3 ? shared / Math.sqrt(w.length) : 0
  })
  const best = Math.max(...scores)
  const picked = new Set<number>()
  if (best <= 0) return picked
  scores.forEach((s, i) => {
    if (s >= best * 0.7) picked.add(i)
  })
  return picked
}

/** Model answers often open with this phrase; the UI already says where answers come from. */
export function tidyAnswer(answer: string): string {
  const trimmed = answer.trim().replace(/^based on the provided (documents?|papers?|text),?\s*/i, '')
  return trimmed.charAt(0).toUpperCase() + trimmed.slice(1)
}
