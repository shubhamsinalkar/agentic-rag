**Result (30 questions, human-audited grades):** CAG answered **30/30** correctly, agentic RAG **24/30**, plain RAG **20/30**. All three stayed inside the 10-second budget at p90.

**When each strategy wins**

- **CAG wins when the knowledge fits in the context window.** Here each question needed 1–2 papers (~26k tokens on average), and CAG was right every time. That includes the detail questions and cross-paper comparisons that both search-based strategies missed (q17, q19, q27, q28). The price: ~25× the input tokens of plain RAG and ~10× agentic RAG (≈ $8.3 vs $2.0 vs $0.5 per 1,000 questions at list price, before the cache discount; 38% of CAG's input was served from Gemini's prompt cache). **It doesn't scale:** with thousands of documents you can't put them all in the prompt, and choosing which ones to include becomes a retrieval problem again. Its paper picker had an easy job with a 6-paper catalog.

- **Agentic RAG beats plain RAG when the first search lands on the wrong chunks.** The correction loop fixed 5 questions plain RAG got wrong or half-right (including the CRAG demo, q14) and broke 1 (q06). That's +13 points, but the 95% bootstrap interval runs from 0 to +30, so with 30 questions it's **suggestive, not proven**. Cost: ~3 LLM calls instead of 1, and +1.4 s median latency.

- **Retries have diminishing returns.** Questions answered with no rewrite were 94% correct, after one rewrite 75%, after two only 25% (1 of 4). The second retry mostly adds latency, so a cap of 1 may be the better trade-off (on only 4 questions, so worth re-testing).

- **Where search-based RAG still fails:** answers spread across a table or list (Self-RAG's reflection-token table, q17), and comparisons needing one fact from each of two papers (q27, q28). Rewriting the query didn't fix these. The natural next steps are **query decomposition** (one sub-question per paper) and table-aware chunking.

- **Plain RAG** is the cheapest and fastest (1.8 s median, ~$0.5 per 1,000 questions). It's enough when the question uses the same words as a single passage.

**Recommendation:** for a small, stable corpus like this one, **CAG**. For the usual production case, a corpus too big for the prompt, use **agentic RAG over plain RAG**, and invest next in query decomposition for comparison questions.

**Caveats:** one run per question (LLM output varies between runs); 30 questions written from the papers by the builder, which may favour single-paper facts; graded by a small LLM judge, audited by hand (98% agreement); latency measured on the free tier.
