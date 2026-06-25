# DeskMate: RAG vs Fine-Tuning Decision Note

**Date:** Phase 4 entry  
**Decision:** Use both — fine-tuning for behaviour, RAG for knowledge.

---

## Question Being Answered

> For each DeskMate capability, should we use fine-tuning, RAG, or both?

---

## Decision Table

| Capability | Mechanism | Rationale |
|---|---|---|
| Route ticket to correct intent | Encoder fine-tune (2.4) | Stable label schema; behaviour learned from examples |
| Extract product / version / OS / error code | Encoder NER (2.5) | Span boundaries are a learned behaviour; schema does not change per-request |
| Generate a concise, empathetic reply | Decoder SFT (3.4) | Format and tone are stable; taught once, used always |
| Avoid unsafe outputs (share PII, impersonate human) | DPO (3.7) | Safety boundaries are behaviour, not facts |
| Current product pricing and plan details | RAG (4.2–4.4) | Pricing changes frequently; baking into weights would go stale |
| Known bugs and release notes | RAG (4.2–4.4) | New releases ship weekly; retrieval is the only scalable mechanism |
| SLA commitments per plan tier | RAG (4.2–4.4) | Policy changes require no retraining when served via RAG |
| Step-by-step how-to answers | RAG (4.2–4.4) | Long-tail FAQ content; fine-tuning cannot guarantee recall of 500+ articles |

---

## The Key Constraint That Drives RAG

DeskMate documentation updates on a **weekly** cadence (release notes, bug fixes, policy changes). Fine-tuning cannot keep pace:

- A fine-tune run takes 1–2 hours of compute and requires curating new training data.
- Weights cannot be surgically updated — a new fact requires a full or partial re-run.
- Stale facts in weights are worse than no facts: the model confidently states outdated information.

RAG sidesteps all three: update the doc store, re-index overnight, no model change required.

---

## What Fine-Tuning Is Responsible For

The fine-tuned decoder (Phase 3) is responsible for **all behaviour that does not depend on current facts**:

- Reply structure: 2–4 sentences, action-first
- Tone: empathetic, professional, no filler
- Task grounding: given intent + retrieved context, produce a reply
- Safety: never claim to be human, never share other customers' data

These behaviours are stable. They do not change when docs update. Fine-tuning them once is the right call.

---

## Combined Architecture

```
Ticket → Encoder (intent + fields)
       → Retriever (top-3 FAQ chunks)
       → Prompt (ticket + intent + chunks)
       → Fine-tuned decoder
       → Reply with citations
```

The decoder's job at inference time is narrow: *given this context, write a 2–4 sentence support reply in DeskMate's voice.* Everything else — knowing what the context should contain — is the retriever's job.

---

## Rejected Alternatives

| Alternative | Why rejected |
|---|---|
| RAG only (no fine-tuning) | Base model generates generic, verbose replies; no safety behaviour; poor format adherence |
| Fine-tuning on all docs | Docs update weekly — weights go stale; hallucination on low-frequency content; no auditability |
| Prompt engineering only (no fine-tune, no RAG) | API cost too high at scale; no latency SLA; no field extraction capability |

---

## Sign-off Criteria

This decision is revisited if:

1. Documentation update cadence drops to monthly or less → consider fine-tuning on docs.
2. Retriever hit-rate@3 falls below 80% on the gold eval set → fix retrieval before blaming the model.
3. Fine-tuned decoder hallucination rate (Module 3.6) rises above 10% despite good retrieval → re-examine SFT data quality.
