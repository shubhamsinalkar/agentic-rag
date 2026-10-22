// A real response from POST /ask (agentic RAG), captured 2026-10-22.
// Shown before the first question so visitors see a full run without waiting.
import type { AskResponse } from './api'

export const EXAMPLE_QUESTION = "How does CRAG decide a retrieved document is incorrect?"

export const EXAMPLE: AskResponse = {
  "answer": "Based on the provided document, CRAG assumes a retrieval is incorrect when the confidence scores of all retrieved documents are below the lower threshold [1].",
  "sources": [
    "corrective_rag.pdf p.5"
  ],
  "strategy": "agentic_rag",
  "route": "conceptual",
  "retries": 1,
  "latency_s": 20.05,
  "steps": [
    {
      "step": "route",
      "label": "conceptual",
      "ms": 70
    },
    {
      "step": "search",
      "query": "How does CRAG decide a retrieved document is incorrect?",
      "found": 4,
      "ms": 5
    },
    {
      "step": "check",
      "sufficient": false,
      "relevant": 1,
      "missing": "The chunks mention that a retrieval evaluator estimates a relevance score and quantifies it into confidence degrees to trigger the Incorrect action, but they do not state the exact threshold values or the specific criteria/rules used to classify a document as incorrect.",
      "ms": 5692
    },
    {
      "step": "rewrite",
      "query": "Corrective Retrieval-Augmented Generation retrieval evaluator confidence scores thresholds classification rules incorrect action",
      "ms": 3388
    },
    {
      "step": "search",
      "query": "Corrective Retrieval-Augmented Generation retrieval evaluator confidence scores thresholds classification rules incorrect action",
      "found": 4,
      "ms": 831
    },
    {
      "step": "check",
      "sufficient": true,
      "relevant": 1,
      "missing": "",
      "ms": 4896
    },
    {
      "step": "answer",
      "cited": 1,
      "ms": 5163
    }
  ],
  "citations": [
    {
      "n": 1,
      "source": "corrective_rag.pdf",
      "page": 5,
      "text": "when the confidence score of at least one retrieved document is higher than the upper threshold. If so, it means that there are relevant documents in the retrieved results, and the knowledge from the retrieval results is supposed to be more reliable and accurate. However, even if a relevant document can be found, there is inevitably some noisy knowledge strips in this document. To extract the most critical knowledge strips within this document, a knowledge refinement method is further designed which will be elaborated in Section 4.4. Incorrect Besides, a retrieval is assumed Incorrect when the confidence scores of all retrieved documents are below the lower threshold. This indicates that all retrieved documents are considered irrelevant, which are unhelpful for generation. Once the knowledge from the retrieval results is judged to be inaccurate, it is unwise to still get stuck in it, which is likely to result in fabricated facts. Therefore, we need to seek new"
    }
  ]
}
