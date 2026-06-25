# Decision Memo — Why DeskMate Should Be Built from SLMs

> **Instructions:** Fill in each section in your own words after reading `0.1_what_is_slm.md`.
> Do not copy from the module file — writing it yourself is the learning. Remove this instruction
> block when done.

---

## The problem

<!-- Describe DeskMate's three jobs in one paragraph: triage, extraction, reply generation -->

## Why SLMs are the right tool

### Cost

<!-- State the cost argument for DeskMate's expected ticket volume with rough numbers -->

### Latency

<!-- Why triage in particular cannot tolerate 500 ms API round-trips -->

### Privacy

<!-- What customer data is in a support ticket and why it cannot leave the premises -->

### Specialization / control

<!-- Why a general model is a liability for a branded support product -->

## Why a frontier LLM is the wrong default

<!-- Argue the flip side — what specifically breaks if DeskMate uses GPT-4o API calls for everything -->

## The two components and their model types

### Component 1 — Encoder SLM (triage + extraction)

- **Task type:**
- **Why encoder, not decoder:**
- **Latency target:** < 20 ms
- **Volume:** every incoming ticket

### Component 2 — Decoder SLM (reply generation)

- **Task type:**
- **Why decoder, not encoder:**
- **Latency target:** < 2 s for a full reply
- **Volume:**

## Decision

<!-- One paragraph: the chosen architecture and why it is the right call for DeskMate -->
