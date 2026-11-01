# SunGrid Copilot — Design Write-Up (Part C)

**Shape of the system.** Every request runs `classify → understand → (eligibility tool + hard gate | filtered hybrid retrieval → grounded answer) → guard`. The category classifier only builds the retrieval filter. The intent step chooses the branch. The eligibility service is a separate component wrapped by a fail-closed adapter.

## 1. Evaluating before shipping

**Offline (built and running in CI):** `sungrid eval` runs 92 golden cases, written separately from the classifier's training queries (a test enforces no overlap), and gates CI on thresholds.

| Layer | Metric | Why it matters |
|---|---|---|
| Classifier | accuracy, confusion matrix, accepted-label sets for multi-category questions | Mistakes shrink the filter, so measure them per category |
| Retrieval | hit@k / MRR, plus a **filter ablation** (none / hard / soft+fallback / oracle) | Shows how much the filter helps or hurts (see below) |
| Routing | intent accuracy, tool precision/recall | A missed eligibility intent is the dangerous error |
| Tool | decision accuracy vs. the stub, argument extraction cases | The agent must pass the member's real values |
| **Safety** | **hard-gate violations = 0** (absolute gate), plus a 150-example property test | The one requirement that cannot regress |
| Answers | fact recall, citation validity, numeric grounding, abstention on off-topic | Faithfulness over fluency |

The ablation produced a useful finding. On 14 documents, unfiltered retrieval already reaches 100% hit@4. A **hard** filter from predicted labels drops to 98.3%, while the shipped **soft filter + fallback** keeps 100%. At this size the filter is about precision and ranking, not recall. Its real value comes at volume (point 2), and the fallback is what makes it safe to use.

Before launch I would add: (a) an **LLM-as-judge faithfulness pass** on generated answers, calibrated against ~100 human labels; (b) **50–100 real member questions** from support tickets, labelled by the support team, because synthetic questions overstate accuracy; (c) a **red-team set** for the gate (indirect estimate requests, multi-turn pressure, injected instructions).

**Online:** log every trace step (already structured). Track: abstention rate, unfiltered-fallback rate (a rising rate means the taxonomy is drifting), classifier margin distribution, tool error/timeout rate (fail-closed answers are silent failures to the member), clarification turns per eligibility session, thumbs-up/down plus escalation to a human, and p95 latency per step. Sample 1–2% of conversations for weekly human review. Run a nightly **canary eval** against production config, and an alert on any gate-guard trigger, since the guard should never fire.

## 2. What breaks first at 10x documents / 100x queries

**10x documents (~140 docs, ~700 chunks):**
1. **The hand-curated manifest breaks first.** Hand-labelling every document doesn't scale. *Change:* auto-tag at ingestion with the classifier (already implemented for unmapped docs), send only low-confidence or high-entropy documents to human review (`sungrid audit` computes this), and move from document-level to **section-level** labels.
2. **Ranking precision.** More near-duplicate policy text means top-4 fills with similar chunks; this is where the category filter starts to earn its keep. *Change:* add a cross-encoder reranker on the top ~30, and keep hybrid search inside Qdrant (sparse + dense vectors) instead of in-process BM25.
3. **Ingestion cost** is not a problem: ids are content hashes, so re-ingest only embeds changed sections.

**100x queries:**
1. **In-process state breaks first.** The embedded Qdrant file lock (one writer), in-memory BM25 and in-memory sessions don't work across replicas. *Change:* Qdrant server/cloud (same client code, only the constructor changes), Redis for sessions, and stateless API pods behind a load balancer.
2. **LLM cost and rate limits.** *Change:* the design already minimises calls: the local classifier handles most queries, the LLM is consulted for understanding only on eligibility-like messages, and eligibility answers are templated (no generation). Add semantic caching for repeated FAQ answers, provider fallback (the adapter is provider-agnostic), and async/batched embedding.
3. **The eligibility service** becomes a dependency to protect: circuit breaker, per-member rate limits, idempotent request ids. The adapter already has timeouts and fails closed.

## 3. Adding a new document category next month

**Changes (config and data, no code):**
- `config/taxonomy.yaml`: add `{id, label, description}`. The description feeds both the LLM adjudication prompt and the classifier's training data.
- `config/kb_manifest.yaml`: map the new documents (or let auto-tagging propose and a human approve).
- `data/classifier/seed_queries.jsonl`: ~15–25 example questions. The classifier retrains at start-up in under a second.
- `data/eval/golden_set.jsonl`: new cases, plus a check of the confusion matrix against neighbouring categories.
- Then run `sungrid ingest` (incremental) and `sungrid eval --strict`.

**Stays the same:** chunking, embeddings, vector store and filter logic (categories are just strings in a list payload), the agent flow, the eligibility tool and gate, guards, the API and the CLI. Category ids are validated against the taxonomy, so a typo fails at load time. **Possible code change:** only if the new category brings a new *tool* (e.g. a battery eligibility service), which would be a new adapter in `agent/tools.py` and a new program in `intent.py`.

## 4. One non-obvious trade-off

**The eligibility answer is a template, not LLM-generated, and the router is deliberately biased toward the tool path.**

An LLM-written eligibility answer would read better and could explain the reason in context. But the hard gate is about what the model *adds*: a "helpful" model can say "you'd have received about $3,200 if your income were lower", which is exactly the partial estimate the brief forbids. A template cannot add that, so the gate becomes a property of the code instead of a hope about the prompt. Guards on generated text (numeric grounding) are kept as defence in depth.

The same reasoning sets the routing bias. Personal or estimate-style wording ("how much would I get") goes to the tool path, even when a document could answer it. A false positive costs one clarifying question; a false negative risks an unchecked estimate. I accepted slightly more friction for members in exchange for a decision that can be audited.

A smaller trade-off: **multi-label documents, single-label queries**. Duplicating doc 06 under both categories would have been simpler, but it doubles results in unfiltered search and splits future edits across copies. See `CATEGORY_DECISIONS.md`.

## Out of scope for the time budget (stated, not skipped)

Authentication and linking a conversation to a verified member account (in production, eligibility inputs should come from account records, not chat). A front-end UI. Persistent conversation storage and analytics. A cross-encoder reranker. Online monitoring infrastructure (the trace has the fields; dashboards were not built). LLM-as-judge evaluation. All of these slot into the current seams (`retrieval.py`, `factory.py`, `api.py`) without restructuring.
