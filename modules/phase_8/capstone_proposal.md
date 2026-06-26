# Capstone Proposal — LexIQ: An Open-Weight Contract Intelligence SLM

> **Phase 9 Graduation Project**
> Author: arbarik@gmail.com · Date: 2026-06-26
> Status: PROPOSAL — awaiting review before implementation begins

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Why Build a Small Language Model](#2-why-build-a-small-language-model)
3. [Technical Benefits](#3-technical-benefits)
4. [What LexIQ Does](#4-what-lexiq-does)
5. [New Concepts This Project Introduces](#5-new-concepts-this-project-introduces)
6. [System Architecture](#6-system-architecture)
7. [Data Sources and Data Pipeline](#7-data-sources-and-data-pipeline)
8. [Training Plan](#8-training-plan)
9. [Evaluation Framework](#9-evaluation-framework)
10. [LLMOps Pipeline](#10-llmops-pipeline)
11. [Deployment Architecture](#11-deployment-architecture)
12. [Budget Breakdown](#12-budget-breakdown)
13. [Timeline](#13-timeline)
14. [Risks and Mitigations](#14-risks-and-mitigations)
15. [Success Criteria](#15-success-criteria)

---

## 1. Problem Statement

### The contract bottleneck

Every company in every industry signs contracts. Employment agreements, vendor agreements, SaaS subscriptions, NDAs, licensing deals, construction contracts, lease agreements, loan covenants — the volume is enormous and growing. A mid-size company with 500 employees can easily have 2,000–5,000 active contracts at any point. An enterprise with acquisitions, global operations, and complex supply chains can have hundreds of thousands.

The hidden cost of this is staggering:

- **Review time:** A trained contract attorney reviews 8–15 pages per hour. A 40-page SaaS enterprise agreement takes 3–5 hours of billable time at $300–700/hour.
- **Throughput bottleneck:** Legal teams are chronically understaffed. Contracts queue for days or weeks, blocking procurement, hiring, and revenue recognition.
- **Inconsistency:** Different reviewers catch different risks. Clause interpretation varies across attorneys, offices, and fatigue levels.
- **Obligation blindness:** Once signed, contracts sit in a document management system and are never read again — until something goes wrong. Renewal dates are missed. Auto-renewal traps trigger. Liability caps are exceeded unknowingly.
- **Scale impossibility:** When a company needs to review 10,000 legacy contracts for GDPR compliance or post-M&A risk assessment, no human team can do it in weeks.

### What currently exists — and why it falls short

General-purpose LLMs (GPT-4, Claude) can read a contract. But deploying them for real contract intelligence has three hard blockers:

1. **Attorney-client privilege.** Sending contracts containing trade secrets, merger negotiations, litigation strategy, or employee compensation to a third-party cloud API is a violation of attorney-client privilege in many jurisdictions. Large law firms and in-house legal teams **cannot** use cloud APIs for real matters. This is not a policy preference — it is a professional ethics requirement.

2. **Auditability.** When a model says "this indemnification clause creates unlimited liability exposure," a client will ask *how* that conclusion was reached. A black-box API call with no inspectable weights cannot satisfy this requirement. Regulatory frameworks (EU AI Act Article 13, forthcoming UK AI Liability regime) require explainability for high-risk AI applications, and contract review is explicitly listed as a high-risk use case.

3. **Cost at scale.** GPT-4 at $0.03/1K output tokens for a 40-page contract (≈ 20,000 tokens output) costs $0.60 per contract. For a firm reviewing 50,000 contracts per year this is $30,000/year in API fees — before volume discounts. A fine-tuned on-premises SLM has marginal cost near zero after the initial training investment.

---

## 2. Why Build a Small Language Model

The standard question: why not just prompt GPT-4?

| Criterion | GPT-4 API | LexIQ (1.5B SLM) |
|---|---|---|
| **Privacy / privilege** | ❌ Data leaves premises | ✅ Runs fully on-premises |
| **Auditability** | ❌ Black box, closed weights | ✅ Open weights, inspectable |
| **Marginal cost per contract** | ~$0.60 (40-page contract) | ~$0.002 (GPU electricity) |
| **Latency (clause classification)** | 800–2000 ms (API round-trip) | 15–40 ms (local encoder) |
| **Legal clause vocabulary** | Generic; hallucinates clause names | Trained on 500K+ legal clauses |
| **Structured output reliability** | Inconsistent JSON compliance | Grammar-constrained decoding |
| **Customisable to firm's playbook** | Requires prompt engineering only | Fine-tunable on internal data |
| **Offline / air-gapped** | ❌ Requires internet | ✅ GGUF runs fully offline |
| **Integration into doc workflow** | REST only | FastAPI + Office/iManage plugin |

The conclusion: for legal contract intelligence, a domain-adapted SLM running on-premises is not a compromise — it is the superior architecture for every criterion that matters in a legal environment.

---

## 3. Technical Benefits

Beyond the business case, there are five genuine technical reasons why a fine-tuned SLM outperforms a prompted general LLM on this task:

### 3.1 Domain-adapted representations

Legal language is highly specialised. Terms like *indemnification*, *force majeure*, *representations and warranties*, *liquidated damages*, *covenant not to compete*, and *most-favoured-nation* have precise legal meanings that differ from their lay meanings. A general LLM's tokenizer fragments these terms suboptimally because they appear rarely in internet text. DAPT (domain adaptive pretraining) on a 5 GB legal corpus before task fine-tuning produces better subword representations for legal vocabulary, measurably improving downstream task accuracy.

### 3.2 Clause-type specialisation reduces hallucination

General LLMs tend to hallucinate clause types, confusing *limitation of liability* with *indemnification*, or generating non-existent clause names. A fine-tuned encoder classifying across the 41 CUAD clause types achieves >0.92 macro-F1 after training on 10,000 labelled examples — far exceeding zero-shot performance of any general model on the same task.

### 3.3 Structured output reliability via constrained decoding

Contract intelligence requires structured output: extracted parties, dates, clause types, risk scores in JSON format. A fine-tuned decoder combined with grammar-constrained decoding (outlines / llama.cpp GBNF grammars) guarantees valid JSON output 100% of the time. GPT-4 with JSON mode achieves ~98% compliance; worse, its failures are unpredictable and require retry logic.

### 3.4 RAFT improves grounded reasoning over retrieved clauses

Standard RAG fine-tunes a decoder to generate answers. RAFT (Retrieval Augmented Fine-Tuning) trains the decoder to explicitly *reason over* retrieved context — to cite specific clause text, explain *why* a clause creates risk, and distinguish its own parametric knowledge from the retrieved evidence. This produces answers that are more faithful to the actual contract text, reducing the single most dangerous failure mode in legal AI: confident incorrect answers.

### 3.5 Latency enables real-time workflow integration

A contract management system needs clause classification in real-time as a user uploads or edits a document. At 15–40 ms per clause (local DeBERTa encoder), LexIQ can classify a 50-clause contract in under 2 seconds — fast enough for interactive UI feedback. GPT-4's 800+ ms API latency per clause makes this interaction model impossible.

---

## 4. What LexIQ Does

LexIQ is a two-model system (encoder + decoder) that answers three question types over any uploaded contract:

### Task 1 — Clause Intelligence (Encoder)

Given a contract, extract and classify every clause:
- **41-type clause classification** (CUAD taxonomy: indemnification, limitation of liability, change of control, most-favoured nation, IP ownership, arbitration, governing law, ...)
- **Named entity extraction**: parties (company names, roles), effective date, termination date, renewal date, notice period, governing jurisdiction, payment amounts
- **Risk signal detection**: uncapped liability, unilateral termination rights, automatic renewal traps, non-compete scope

Output: a structured JSON clause map of the entire contract, ready for display in a contract management UI.

### Task 2 — Risk Q&A (Decoder + RAG)

Answer questions about the contract grounded in retrieved clause text:
- "What is our liability exposure if we breach the SLA?"
- "Can the vendor terminate without cause, and with how much notice?"
- "Does this agreement include a non-compete? What is its geographic scope?"
- "When does this contract auto-renew, and what is the notice period to cancel?"

Output: a cited answer referencing specific clause text, with faithfulness score.

### Task 3 — Playbook Comparison (Decoder + RAG)

Compare contract clauses against a firm's internal negotiation playbook and flag deviations:
- "Our standard playbook requires mutual indemnification. This contract has unilateral indemnification in the vendor's favour. Deviation: HIGH RISK."
- "Governing law is Delaware (acceptable). Arbitration is required (non-standard for this contract type — flag for review)."

Output: a clause-by-clause deviation report with risk severity ratings.

---

## 5. New Concepts This Project Introduces

LexIQ builds on every concept from Phases 1–7 and adds six new concepts not covered in DeskMate:

| # | New Concept | Why it's needed | Where it appears |
|---|---|---|---|
| 1 | **Hierarchical document encoding** | Contracts are 5,000–50,000 tokens — beyond any single-pass encoder | Data pipeline + encoder |
| 2 | **Multi-task fine-tuning (MTL)** | Train one encoder for classification + NER + span extraction simultaneously | Encoder training |
| 3 | **Constrained / grammar-guided decoding** | Guarantee valid JSON output for structured extraction | Decoder inference |
| 4 | **RAFT (Retrieval Augmented Fine-Tuning)** | Teach the decoder to reason over retrieved context, not ignore it | Decoder training |
| 5 | **Constitutional AI / RLAIF for DPO data** | Generate preference data at scale without expensive human annotators | DPO data pipeline |
| 6 | **Active learning for annotation prioritisation** | Label the most informative contracts first (uncertainty sampling) | Data acquisition |

### 5.1 Hierarchical document encoding

A 40-page contract is ~10,000–30,000 tokens. DeBERTa's context window is 512 tokens. The solution is a two-level hierarchical encoder:

```
Contract (30,000 tokens)
  ├── Split into overlapping chunks of 512 tokens (stride 128)
  ├── Encode each chunk → 768-dim CLS embedding
  ├── Stack chunk embeddings → (N_chunks × 768) matrix
  └── Feed through a lightweight transformer aggregator (2 layers, 4 heads)
      → document-level embedding for contract-level tasks
      → clause-level embeddings for span tasks
```

This is a standard technique in long-document NLP but was not covered in DeskMate (which handles short tickets).

### 5.2 Multi-task fine-tuning

A single DeBERTa model is fine-tuned on three heads simultaneously:
- **Head 1:** 41-class clause type classifier (cross-entropy loss)
- **Head 2:** NER tagger for parties, dates, amounts (token-level cross-entropy)
- **Head 3:** Span extractor for obligation clauses (start/end position regression)

Joint training with a weighted loss `L = α·L_cls + β·L_ner + γ·L_span` forces shared representations that generalise better than three separate models.

### 5.3 Constrained / grammar-guided decoding

Using `outlines` (Python library) or llama.cpp GBNF grammar files, the decoder is constrained to emit only valid JSON conforming to a predefined schema:

```python
from outlines import models, generate

model  = models.transformers("lexiq-decoder-qlora")
schema = '{"risk_level": {"type": "string", "enum": ["low","medium","high","critical"]}, ...}'
gen    = generate.json(model, schema)
result = gen("Assess this indemnification clause: ...")
# Guaranteed to return valid JSON matching schema
```

This eliminates the retry-loop overhead of unconstrained generation and makes the system safe to integrate into downstream automated workflows.

### 5.4 RAFT (Retrieval Augmented Fine-Tuning)

Standard SFT trains the decoder: `prompt → answer`. RAFT trains it: `prompt + retrieved_context → answer_with_citation`. Critically, RAFT training includes **distractor documents** (irrelevant retrieved chunks) so the model learns to ignore noise:

```
Training examples:
  - 70%: correct retrieved clause + correct answer citing it (learn to use context)
  - 30%: irrelevant retrieved clause + correct answer from parametric memory (learn to ignore noise)
```

The result is a decoder that, in production, is significantly more faithful to retrieved contract text than a standard SFT model.

### 5.5 Constitutional AI / RLAIF for DPO

Human annotation of DPO preference pairs costs $5–15 per pair. For 5,000 pairs that is $25,000–75,000 — not feasible for an independent project. Constitutional AI solves this: use a capable LLM (Claude claude-sonnet-4-6 or GPT-4o) as the "constitution" to generate and rank preference pairs automatically.

```
For each contract clause and two candidate answers (A and B):
  1. Ask GPT-4o: "Which answer is more faithful to the clause text? More specific? Less hallucinated?"
  2. Record (A, B, winner) as a DPO preference pair
  3. Use these AI-labelled pairs for DPO fine-tuning

Cost: ~50M tokens of GPT-4o input/output ≈ $150 for 5,000 pairs
vs.
Human annotation: $25,000–75,000 for the same 5,000 pairs
```

Quality of RLAIF-generated preferences is 85–90% correlated with human preferences on legal tasks (per recent research), making this a viable substitute for a research/education project.

### 5.6 Active learning for annotation prioritisation

Not all unlabelled contracts are equally informative. Active learning selects the contracts where the model is most uncertain — where labelling would improve the model most:

```python
def uncertainty_score(probabilities: np.ndarray) -> float:
    # Margin sampling: gap between top-2 predictions
    sorted_probs = np.sort(probabilities)[::-1]
    return 1.0 - (sorted_probs[0] - sorted_probs[1])

# Rank unlabelled contracts by uncertainty; label the top-K
```

This allows the annotation budget to be used 2–3× more efficiently than random sampling.

---

## 6. System Architecture

```
                         ┌──────────────────────────────────────┐
                         │  LexIQ Service (FastAPI)             │
                         │                                      │
  Document upload  ──►   │  ┌───────────────────────────────┐   │
  (PDF / DOCX)           │  │  Document Parser              │   │
                         │  │  (pdfplumber / python-docx)   │   │
                         │  └──────────────┬────────────────┘   │
                         │                 │ text + pages        │
                         │  ┌──────────────▼────────────────┐   │
                         │  │  Hierarchical Chunker         │   │
                         │  │  512-token chunks, stride 128 │   │
                         │  └──────────────┬────────────────┘   │
                         │                 │ chunks              │
                         │  ┌──────────────▼────────────────┐   │
                         │  │  Encoder SLM (DeBERTa-base)   │   │
                         │  │  Multi-task fine-tuned        │   │
                         │  │  · Clause type (41 classes)   │   │
                         │  │  · NER (parties, dates)       │   │
                         │  │  · Span extraction            │   │
                         │  └──────────────┬────────────────┘   │
                         │                 │ clause map JSON     │
                         │                 │                     │
  Q&A query ──────────►  │  ┌──────────────▼────────────────┐   │
                         │  │  Retriever (FAISS + BM25)     │   │
                         │  │  · Search within THIS contract │   │
                         │  │  · Search regulation corpus   │   │
                         │  │  · Search playbook KB         │   │
                         │  └──────────────┬────────────────┘   │
                         │                 │ top-3 chunks        │
                         │  ┌──────────────▼────────────────┐   │
                         │  │  Decoder SLM (Qwen2.5-1.5B)  │   │
                         │  │  RAFT + QLoRA fine-tuned      │   │
                         │  │  + Constrained JSON decoding  │   │
                         │  └──────────────┬────────────────┘   │
                         │                 │ cited answer        │
                         │  ┌──────────────▼────────────────┐   │
                         │  │  Output Guardrails            │   │
                         │  │  · Citation enforcement       │   │
                         │  │  · Risk score range check     │   │
                         │  │  · Confidence gate            │   │
                         │  └──────────────┬────────────────┘   │
                         │                 │                     │
                         └─────────────────┼────────────────────┘
                                           │
                                   Final response
                             (clause map + Q&A + deviations)
```

### Knowledge bases (3 RAG indices)

| Index | Content | Size | Update frequency |
|---|---|---|---|
| **Contract index** | Chunks of the uploaded contract (per-session) | Dynamic | Per upload |
| **Regulation corpus** | GDPR, California CCPA, NY Labour Law, UCC Articles 1-9, CISG | ~2 M tokens | Quarterly |
| **Playbook KB** | Firm's negotiation standards, preferred clause templates, red lines | ~200 K tokens | As edited by firm |

---

## 7. Data Sources and Data Pipeline

All sources are publicly available or generatable with no proprietary data required.

### 7.1 Pre-training / DAPT corpus

| Source | Volume | Use | Licence |
|---|---|---|---|
| **Pile of Law** (HF: `pile-of-law/pile-of-law`) | 256 GB total; use 5 GB filtered subset | DAPT legal domain adaptation | Various open licences |
| **SEC EDGAR** (full-text search API) | ~500K 10-K, 10-Q filings | DAPT for corporate/commercial language | Public domain (US gov) |
| **CourtListener / FreeLaw** (`free-law/courtlistener`) | 6.9M court opinions; use 500K | DAPT for legal reasoning patterns | CC-BY |
| **EUR-Lex English** | EU legislation, directives, regulations | DAPT for regulatory language | Public domain |

Total DAPT corpus: **~5 GB** of legal text after filtering for quality (remove boilerplate headers, duplicate filings, documents shorter than 512 tokens).

### 7.2 Encoder fine-tuning data

| Source | Examples | Labels | Licence |
|---|---|---|---|
| **CUAD** (`cuad`) | 510 contracts, 13,000+ annotated spans | 41 clause types | CC-BY-4.0 |
| **ContractNLI** | 607 contracts, 17,000 NLI labels | Entailment / neutral / contradiction | MIT |
| **EDGAR clause extraction** (synthetic) | 10,000 generated clause-label pairs from GPT-4o | 41 clause types | Synthetic / own work |
| **Synthetic NER** (Claude generation) | 5,000 contract snippets with party/date/amount annotations | NER tags | Synthetic / own work |

Total encoder training data: **~28,000 labelled examples** across clause classification + NER + span extraction.

**Active learning strategy:** Start with the 13,000 CUAD examples. After each training round, run uncertainty sampling on 5,000 unlabelled EDGAR contracts. Select the top-500 most uncertain → auto-label with GPT-4o → add to training pool. Three rounds doubles the effective training set size while spending annotation budget 2.5× more efficiently than random sampling.

### 7.3 Decoder SFT data

| Source | Examples | Format | Notes |
|---|---|---|---|
| **CUAD + GPT-4o synthesis** | 15,000 (clause, question, grounded answer) triples | SFT prompt-completion | GPT-4o generates answers, human spot-checks 500 |
| **ContractNLI converted** | 5,000 entailment pairs → Q&A format | SFT | Automatic conversion |
| **Playbook deviation examples** | 3,000 synthetic deviation reports | SFT | Templated + GPT-4o |
| **RAFT distractors** | 30% of above, with irrelevant context injected | RAFT SFT | Model must ignore distractors |

Total decoder SFT data: **~23,000 training examples**.

### 7.4 DPO preference data (RLAIF)

- **Method:** Constitutional AI with GPT-4o as judge
- **Volume:** 5,000 preference pairs (winner / loser per Q&A question)
- **Constitution principles:**
  1. The answer must cite the specific clause text (not paraphrase from memory)
  2. Risk ratings must be justified with explicit clause language
  3. The answer must not make legal claims beyond what the clause says
  4. If the clause is ambiguous, the answer must say so explicitly
- **Cost:** ~50M GPT-4o tokens ≈ **$150** (input + output)
- **Validation:** human review of 200 randomly sampled pairs to measure RLAIF-human agreement

### 7.5 Regulation corpus (RAG knowledge base)

| Document | Tokens | Source |
|---|---|---|
| GDPR full text | 85K | EUR-Lex (public domain) |
| CCPA + CPRA | 45K | California AG (public) |
| UCC Articles 1-9 | 120K | Cornell LII (public) |
| CISG (international sale of goods) | 30K | UN (public) |
| Standard contract clause templates (NDA, SaaS, SLA) | 200K | Own synthesis |
| Firm playbook (stub for demo) | 50K | Synthetic |

---

## 8. Training Plan

### Stage 1 — Domain Adaptive Pretraining (DAPT)

| Item | Value |
|---|---|
| Base model | `Qwen2.5-1.5B` (decoder) + `microsoft/deberta-v3-base` (encoder) |
| Corpus | 5 GB filtered legal text |
| Method | Causal LM on decoder; MLM on encoder |
| Duration | ~12 GPU-hours (A100) for decoder; ~4 GPU-hours for encoder |
| Output | `lexiq-decoder-dapt` + `lexiq-encoder-dapt` |
| Eval | Perplexity on held-out legal text vs base model (expect 15–25% reduction) |

### Stage 2 — Encoder Multi-Task Fine-Tuning

| Item | Value |
|---|---|
| Base | `lexiq-encoder-dapt` |
| Data | 28,000 examples (clause class + NER + span) |
| Architecture | DeBERTa-base + hierarchical aggregator + 3 task heads |
| Loss | `0.5·L_cls + 0.3·L_ner + 0.2·L_span` |
| Training | 5 epochs, LR 2e-5, batch 32, gradient checkpointing |
| Duration | ~8 GPU-hours (A100) |
| Eval target | Clause type macro-F1 ≥ 0.88; NER F1 ≥ 0.85 |

### Stage 3 — Decoder RAFT + QLoRA Fine-Tuning

| Item | Value |
|---|---|
| Base | `lexiq-decoder-dapt` |
| Data | 23,000 RAFT SFT examples (incl. 30% distractor) |
| Method | QLoRA (4-bit NF4 base, LoRA rank 16, α 32) |
| Duration | ~24 GPU-hours (A100) |
| Eval target | ROUGE-L ≥ 0.55 on held-out Q&A; faithfulness ≥ 0.75 |

### Stage 4 — DPO Alignment

| Item | Value |
|---|---|
| Base | `lexiq-decoder-qlora` merged |
| Data | 5,000 RLAIF preference pairs |
| Method | DPO (β 0.1) |
| Duration | ~8 GPU-hours (A100) |
| Eval target | Win rate of DPO vs SFT on GPT-4o judge ≥ 65% |

### Stage 5 — Quantisation and Export

| Format | Target | Use case |
|---|---|---|
| fp16 | Server deployment | vLLM / FastAPI |
| GPTQ int4 | Reduced-VRAM server | 8 GB VRAM machines |
| GGUF Q4_K_M | Local / offline | Ollama on firm's laptop |
| ONNX (encoder only) | CPU deployment | Document management plugin |

---

## 9. Evaluation Framework

### Encoder evaluations

| Metric | Task | Dataset | Target |
|---|---|---|---|
| Macro-F1 | Clause type classification (41 classes) | CUAD test split | ≥ 0.88 |
| Entity F1 | NER (parties, dates, amounts) | ContractNLI test | ≥ 0.85 |
| Span Exact Match | Obligation extraction | CUAD annotations | ≥ 0.72 |
| Inference latency | Single clause (512 tokens) | Local A10G | ≤ 40 ms |

### Decoder evaluations

| Metric | Task | Dataset | Target |
|---|---|---|---|
| ROUGE-L | Risk Q&A answers | Held-out CUAD Q&A | ≥ 0.55 |
| Citation precision | Answer cites actual clause text | Manual, 100 samples | ≥ 0.90 |
| Faithfulness score | N-gram overlap with retrieved context | Automated | ≥ 0.75 |
| Hallucination rate | Answers contain facts not in contract | GPT-4o judge, 200 samples | ≤ 5% |
| JSON validity | Constrained output always parses | Automated, 1000 generations | 100% |
| DPO win rate | RLAIF judge prefers DPO over SFT | GPT-4o, 200 pairs | ≥ 65% |

### System-level evaluations

| Metric | Target |
|---|---|
| Full contract processing time (40 pages) | ≤ 15 seconds |
| Playbook deviation recall (simulated) | ≥ 0.82 |
| No-citation guardrail fire rate | ≤ 3% |
| Escalation rate on out-of-scope questions | ≥ 90% correctly refused |

---

## 10. LLMOps Pipeline

Applies everything from Phases 7.4 and 7.6, adapted to legal document review:

### Logging schema

```json
{
  "request_id":      "abc123",
  "contract_hash":   "sha256-of-document",
  "task_type":       "clause_classify | risk_qa | playbook_compare",
  "n_clauses":       48,
  "clauses_flagged": 3,
  "mean_confidence": 0.87,
  "faithfulness":    0.79,
  "latency_ms":      8300,
  "guardrail_flags": {"no_citation": false, "low_faith": false},
  "user_feedback":   null
}
```

### Drift detection signals (legal-domain specific)

| Signal | Threshold | Interpretation |
|---|---|---|
| Clause type entropy rising | KL > 0.20 vs baseline | New contract types appearing |
| Mean encoder confidence falling | < 0.78 for 7 days | Vocabulary shift or new clause styles |
| No-citation rate rising | > 8% for 3 days | Decoder drifting from grounding |
| Faithfulness score falling | CUSUM alarm | Output quality degrading |

### Retraining trigger

Two-signal rule (same as DeskMate):
- **Scheduled:** weekly, incorporates newly processed contracts as weakly-labelled examples
- **Alert-driven:** any two signals above → trigger fine-tuning run on accumulated data

### CI gate (stricter than DeskMate — legal context)

| Gate | Condition |
|---|---|
| Clause macro-F1 | candidate ≥ prod − 0.005 (tighter than general — legal errors costly) |
| NER F1 | candidate ≥ prod − 0.01 |
| Hallucination rate | candidate ≤ 5% (hard ceiling; no regression allowed) |
| JSON validity | candidate = 100% |
| No-citation rate | candidate ≤ 5% |

### Canary deployment

- 10% of new uploads → candidate model; 90% → incumbent
- 48-hour monitoring window (longer than DeskMate — legal reviews are lower volume, need more data)
- Rollback trigger: faithfulness delta > 0.03 or hallucination rate > 7%

---

## 11. Deployment Architecture

### Primary deployment (law firm / in-house legal team)

```
Law firm server (on-premises, no internet required)
  │
  ├── LexIQ API server (FastAPI + vLLM)
  │     Encoder:  DeBERTa ONNX on CPU (≤ 40 ms)
  │     Decoder:  Qwen2.5-1.5B GPTQ-int4 on GPU (1× RTX 3090 24 GB)
  │
  ├── FAISS indices (3: contract, regulation, playbook)
  │
  ├── Document management integration
  │     → iManage / NetDocuments webhook
  │     → Microsoft 365 Word add-in (REST calls to LexIQ API)
  │
  └── Monitoring stack
        → Structured logs → Elasticsearch
        → Grafana dashboard (clause distribution, latency, confidence)
        → Alertmanager → email + Slack alerts
```

### Secondary deployment (solo practitioner / small firm)

```
Local laptop / desktop (Mac or Windows)
  └── Ollama with lexiq-Q4_K_M.gguf
        Encoder: ONNX Runtime Mobile (CPU)
        Decoder: GGUF via llama.cpp (CPU-only, no GPU required)
        Runs fully offline — no internet connection needed
```

Expected performance on laptop (Apple M2 Pro, 16 GB):
- Encoder (single clause): 8 ms (ONNX CPU)
- Decoder (200-token risk answer): ~12 seconds (GGUF CPU Q4_K_M)
- Full 40-page contract scan: ~3 minutes (encoder only), ~8 minutes (with risk answers)

---

## 12. Budget Breakdown

### Compute

| Task | GPU | Hours | Rate | Cost |
|---|---|---|---|---|
| DAPT — decoder (5 GB corpus) | A100 80GB | 12 | $1.20/hr (Lambda) | $14.40 |
| DAPT — encoder (5 GB corpus) | A100 40GB | 4 | $1.00/hr | $4.00 |
| Encoder MTL fine-tuning | A100 40GB | 8 | $1.00/hr | $8.00 |
| Decoder RAFT QLoRA fine-tuning | A100 80GB | 24 | $1.20/hr | $28.80 |
| DPO fine-tuning | A100 40GB | 8 | $1.00/hr | $8.00 |
| Evaluation runs (×5) | A10G | 10 | $0.75/hr | $7.50 |
| Smoke-test iterations (×10) | T4 (Colab free) | 30 | $0 | $0 |
| **Total compute** | | | | **$70.70** |

### Annotation and synthetic data

| Item | Volume | Method | Cost |
|---|---|---|---|
| DPO preference pairs (RLAIF) | 5,000 pairs | GPT-4o ($0.003/1K tokens, ~50M tokens) | $150 |
| Encoder synthetic data | 15,000 examples | Claude claude-sonnet-4-6 ($0.003/1K, ~20M tokens) | $60 |
| Human spot-check (encoder QA) | 500 examples | Upwork ($0.50/example) | $250 |
| Human spot-check (decoder QA) | 200 examples | Upwork ($2/example) | $400 |
| **Total annotation** | | | **$860** |

### Infrastructure

| Item | Monthly | Months | Cost |
|---|---|---|---|
| HuggingFace Hub private repos | $9 | 3 | $27 |
| Vast.ai A10G inference server (dev/test) | $50 | 2 | $100 |
| Weights & Biases (Starter plan) | $0 | 3 | $0 |
| Storage (S3-compatible, 100 GB) | $2 | 3 | $6 |
| **Total infrastructure** | | | **$133** |

### Total project budget

| Category | Cost |
|---|---|
| Compute | $71 |
| Annotation / synthetic data | $860 |
| Infrastructure | $133 |
| **TOTAL** | **$1,064** |

> **Budget note:** The $860 annotation cost can be reduced to ~$210 by using only RLAIF (no human spot-check), accepting slightly lower quality assurance. The compute cost is irreducible but modest — the entire model lifecycle costs less than 3 hours of a senior attorney's time.

---

## 13. Timeline

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Data acquisition + preprocessing | 5 GB DAPT corpus, 28K encoder dataset, 23K decoder dataset |
| 2 | DAPT runs (encoder + decoder) | `lexiq-encoder-dapt`, `lexiq-decoder-dapt` |
| 3 | Encoder MTL fine-tuning + eval | Macro-F1 ≥ 0.88; NER F1 ≥ 0.85 |
| 4 | RAFT SFT data generation + fine-tuning | `lexiq-decoder-qlora` |
| 5 | RLAIF DPO data generation + DPO run | `lexiq-decoder-dpo` |
| 6 | RAG index construction (3 indices) | Working retrieval pipeline |
| 7 | System integration + FastAPI service | End-to-end pipeline on 5 test contracts |
| 8 | Quantisation + Ollama packaging | GPTQ-int4, GGUF Q4_K_M |
| 9 | LLMOps setup (logging, monitoring, CI/CD) | GitHub Actions pipeline; Grafana dashboard |
| 10 | Full evaluation + ablations | Eval scorecard; ablation: RAFT vs no-RAFT; DPO vs SFT |
| 11 | Model card + datasheet + portfolio writeup | Final portfolio artifacts |
| 12 | Demo deployment + capstone presentation | Live demo accessible at localhost or Hugging Face Spaces |

---

## 14. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CUAD data too small for reliable 41-class classification | Medium | High | Augment with 10K GPT-4o synthetic examples + active learning |
| RAFT training makes model worse (ignores retrieved context) | Low | High | Compare RAFT vs SFT on held-out set; fall back to SFT if delta < 0.02 |
| Constrained decoding (outlines) incompatible with GGUF | Medium | Medium | Fall back to JSON mode + retry; document as known limitation |
| DPO RLAIF preference quality insufficient (< 75% human agreement) | Medium | Medium | Human spot-check 200 pairs; if agreement < 75%, weight RLAIF pairs by confidence |
| Hierarchical encoder aggregator doesn't converge | Low | High | Simplify to mean-pooling of chunk embeddings as fallback |
| GPU cost overruns due to repeated runs | Low | Low | All runs smoke-tested on Colab T4 (free) before paid GPU launch |

---

## 15. Success Criteria

LexIQ is successful when all of the following hold:

**Functional:** A lawyer uploads a 40-page SaaS agreement and receives within 15 seconds: (1) a structured clause map identifying all 41 clause types present, parties, key dates; (2) answers to 3 risk questions citing specific clause text; (3) a playbook deviation report flagging non-standard terms.

**Technical:** All evaluation targets from Section 9 are met. Hallucination rate ≤ 5%. JSON output validity 100%. RAFT demonstrably outperforms standard SFT on faithfulness (delta ≥ 0.03).

**Operational:** CI/CD pipeline successfully promotes a retrained model through eval gate and canary without manual intervention. Monitoring dashboard shows drift detection working on a simulated distribution shift.

**Portfolio:** A published model card on HuggingFace documenting architecture, training data, evaluation results, limitations, and intended/prohibited uses. A Ollama-based local demo with a simple Gradio UI that anyone can run with `ollama run lexiq`.

**Mastery test (Phase 9 checkpoint):** "Could you rebuild this for a different domain from scratch?" Yes: the same pipeline (DAPT → encoder MTL → decoder RAFT + QLoRA → DPO via RLAIF → RAG → vLLM + FastAPI → monitoring + CI/CD) applies to any document-heavy, privacy-sensitive domain — clinical notes, financial filings, scientific papers, regulatory submissions.

---

## Appendix: Concept-to-Phase Mapping

| Concept | Source phase | LexIQ application |
|---|---|---|
| Tokenizer + BPE | Phase 0.4 | Legal domain vocabulary analysis |
| DAPT decision | Phase 1.7 | DAPT justified: legal is far from web text |
| Synthetic data generation | Phase 2.2 | GPT-4o generates 15K encoder examples |
| Encoder fine-tuning | Phase 2.4 | DeBERTa clause classifier |
| NER / span extraction | Phase 2.5 | Party + date + obligation extraction |
| Encoder evaluation | Phase 2.6 | Macro-F1, NER-F1 on CUAD test |
| QLoRA fine-tuning | Phase 3.4 | Decoder RAFT SFT |
| DPO alignment | Phase 3.7 | RLAIF-generated preference pairs |
| RAG chunking + retrieval | Phases 4.2-4.3 | 3 separate FAISS indices |
| Grounded generation + faithfulness | Phase 4.4 | RAFT training objective |
| Quantisation (GPTQ + GGUF) | Phase 5.3 | int4 server + Q4_K_M local |
| ONNX export | Phase 5.4 | Encoder → ONNX for CPU deployment |
| vLLM + FastAPI | Phase 6.2-6.3 | Production server |
| Ollama local | Phase 6.4 | Offline deployment |
| Orchestration + guardrails | Phase 7.1 | Full pipeline with hallucination guard |
| Monitoring + drift | Phase 7.4 | Clause distribution drift; CUSUM faithfulness |
| CI/CD eval gate + canary | Phase 7.6 | Automated retrain → promote pipeline |
| **Hierarchical encoding** | **NEW** | Long document handling |
| **Multi-task fine-tuning** | **NEW** | Joint cls + NER + span training |
| **Constrained decoding** | **NEW** | outlines / GBNF grammar-guided output |
| **RAFT** | **NEW** | Teach decoder to reason over context |
| **Constitutional AI / RLAIF** | **NEW** | DPO data at $150 instead of $25,000 |
| **Active learning** | **NEW** | Annotation budget efficiency |
