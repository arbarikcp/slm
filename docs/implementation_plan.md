# DeskMate SLM — Implementation Plan & Progress Tracker

> This document is the live execution record. Update the **Status** column after completing each module, add notes in the **Notes** column, and record the date when you pass the checkpoint. The curriculum document (`SLM_curriculum.md`) is the full reference; this file is the day-to-day tracker.

---

## Legend

| Status | Meaning |
|---|---|
| ⬜ Not started | Module not yet begun |
| 🔵 In progress | Currently working on this module |
| ✅ Done | Deliverable committed, checkpoint passed |
| ⏭️ Skipped | Optional module — consciously skipped |

---

## Phase 0 — Foundations & Setup

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 0.1 | What an SLM is, and when small wins | ⬜ | `decision_memo.md` | ☐ | | |
| 0.2 | Environment, repo, free-tier compute | ⬜ | Repo skeleton + W&B + HF Hub checkpoint | ☐ | | |
| 0.3 | The Transformer — conceptual preview pass | ⬜ | Annotated diagrams + glossary (6 terms) | ☐ | | |
| 0.4 | Tokenization deep dive | ⬜ | Trained domain BPE tokenizer + token-count comparison notebook | ☐ | | |

**Module files:**
- 0.1 → [modules/phase_0/0.1_what_is_slm/0.1_what_is_slm.md](../modules/phase_0/0.1_what_is_slm/0.1_what_is_slm.md)
- 0.1 deliverable → [modules/phase_0/0.1_what_is_slm/decision_memo.md](../modules/phase_0/0.1_what_is_slm/decision_memo.md)
- 0.2 theory → [modules/phase_0/0.2_environment/0.2_environment.md](../modules/phase_0/0.2_environment/0.2_environment.md)
- 0.2 notebook → [modules/phase_0/0.2_environment/00_hello_gpu.ipynb](../modules/phase_0/0.2_environment/00_hello_gpu.ipynb)
- 0.3 theory → [modules/phase_0/0.3_transformer_conceptual/0.3_transformer_conceptual.md](../modules/phase_0/0.3_transformer_conceptual/0.3_transformer_conceptual.md)
- 0.4 theory → [modules/phase_0/0.4_tokenization/0.4_tokenization.md](../modules/phase_0/0.4_tokenization/0.4_tokenization.md)
- 0.4 notebook → [modules/phase_0/0.4_tokenization/01_tokenization.ipynb](../modules/phase_0/0.4_tokenization/01_tokenization.ipynb)

---

## Phase 1 — Build a Transformer from Scratch

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 1.1 | Data, batching, training loop (bigram baseline) | ⬜ | Bigram model + loss curve | ☐ | | |
| 1.2 | Self-attention by hand | ⬜ | `Head` module + attention heatmap | ☐ | | |
| 1.3 | Multi-head attention, MLP, and the Block | ⬜ | `Block` module (stackable) | ☐ | | |
| 1.4 | Assemble nano-SLM, train, generate | ⬜ | Full nano-SLM generating domain text + saved weights | ☐ | | |
| 1.5 | Why we stop here: scaling & pretrained bases | ⬜ | Written analysis with scaling-law numbers | ☐ | | |
| 1.6 | Modern architecture variants *(new)* | ⬜ | nano-GPT vs modern SLM comparison table (RoPE, GQA, SwiGLU, Flash Attention) | ☐ | | |
| 1.7 | Bridge: nano-GPT → pretrained bases; DAPT intro *(new)* | ⬜ | DAPT decision note | ☐ | | |

**Module files:**
- 1.1 theory → [modules/phase_1/1.1_bigram/1.1_bigram.md](../modules/phase_1/1.1_bigram/1.1_bigram.md)
- 1.1 notebook → [modules/phase_1/1.1_bigram/02_bigram.ipynb](../modules/phase_1/1.1_bigram/02_bigram.ipynb)
- 1.2 theory → [modules/phase_1/1.2_self_attention/1.2_self_attention.md](../modules/phase_1/1.2_self_attention/1.2_self_attention.md)
- 1.2 notebook → [modules/phase_1/1.2_self_attention/03_self_attention.ipynb](../modules/phase_1/1.2_self_attention/03_self_attention.ipynb)
- 1.3 theory → [modules/phase_1/1.3_multihead_block/1.3_multihead_block.md](../modules/phase_1/1.3_multihead_block/1.3_multihead_block.md)
- 1.3 notebook → [modules/phase_1/1.3_multihead_block/04_block.ipynb](../modules/phase_1/1.3_multihead_block/04_block.ipynb)
- 1.4 theory → [modules/phase_1/1.4_nano_slm/1.4_nano_slm.md](../modules/phase_1/1.4_nano_slm/1.4_nano_slm.md)
- 1.4 notebook → [modules/phase_1/1.4_nano_slm/05_nano_slm.ipynb](../modules/phase_1/1.4_nano_slm/05_nano_slm.ipynb)
- 1.5 theory → [modules/phase_1/1.5_scaling_pretrained/1.5_scaling_pretrained.md](../modules/phase_1/1.5_scaling_pretrained/1.5_scaling_pretrained.md)
- 1.6 theory → [modules/phase_1/1.6_modern_arch_variants/1.6_modern_arch_variants.md](../modules/phase_1/1.6_modern_arch_variants/1.6_modern_arch_variants.md)
- 1.7 theory → [modules/phase_1/1.7_bridge_dapt/1.7_bridge_dapt.md](../modules/phase_1/1.7_bridge_dapt/1.7_bridge_dapt.md)

**Notebooks:** `02_bigram`, `03_self_attention`, `04_block`, `05_nano_slm`, `06_modern_arch_variants`, `07_bridge_dapt_intro`

---

## Phase 2 — The Encoder SLM: Triage & Extraction

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 2.1 | Problem framing, label schema, data acquisition | ⬜ | Unified public dataset + gold set on HF Hub + datasheet | ☐ | | |
| 2.2 | Synthetic data generation | ⬜ | Generator script + ~5k validated labeled examples with extraction spans | ☐ | | |
| 2.3 | Data prep for encoder fine-tuning | ⬜ | Tokenized `datasets` pipeline yielding batched tensors | ☐ | | |
| 2.4 | Fine-tune encoder for intent + priority | ⬜ | Intent + priority classifier (DistilBERT/MiniLM) | ☐ | | |
| 2.5 | Token classification for field extraction (NER) | ⬜ | Field extractor → JSON output per ticket | ☐ | | |
| 2.6 | Domain-specific evaluation | ⬜ | Eval report: confusion matrix + top-10 error patterns on gold set | ☐ | | |
| 2.7 *(opt)* | Distillation: shrink it further | ⬜ | Distilled classifier benchmarked vs teacher on quality + latency | ☐ | | |

**Module files:**
- 2.1 theory → [modules/phase_2/2.1_data_acquisition/2.1_data_acquisition.md](../modules/phase_2/2.1_data_acquisition/2.1_data_acquisition.md)
- 2.1 notebook → [modules/phase_2/2.1_data_acquisition/08_data_acquisition.ipynb](../modules/phase_2/2.1_data_acquisition/08_data_acquisition.ipynb)
- 2.2 theory → [modules/phase_2/2.2_synthetic_data/2.2_synthetic_data.md](../modules/phase_2/2.2_synthetic_data/2.2_synthetic_data.md)
- 2.2 notebook → [modules/phase_2/2.2_synthetic_data/09_synthetic_data.ipynb](../modules/phase_2/2.2_synthetic_data/09_synthetic_data.ipynb)
- 2.3 theory → [modules/phase_2/2.3_encoder_dataprep/2.3_encoder_dataprep.md](../modules/phase_2/2.3_encoder_dataprep/2.3_encoder_dataprep.md)
- 2.3 notebook → [modules/phase_2/2.3_encoder_dataprep/10_encoder_dataprep.ipynb](../modules/phase_2/2.3_encoder_dataprep/10_encoder_dataprep.ipynb)
- 2.4 theory → [modules/phase_2/2.4_encoder_finetune/2.4_encoder_finetune.md](../modules/phase_2/2.4_encoder_finetune/2.4_encoder_finetune.md)
- 2.4 notebook → [modules/phase_2/2.4_encoder_finetune/11_encoder_finetune.ipynb](../modules/phase_2/2.4_encoder_finetune/11_encoder_finetune.ipynb)
- 2.5 theory → [modules/phase_2/2.5_ner_extraction/2.5_ner_extraction.md](../modules/phase_2/2.5_ner_extraction/2.5_ner_extraction.md)
- 2.5 notebook → [modules/phase_2/2.5_ner_extraction/12_ner_extraction.ipynb](../modules/phase_2/2.5_ner_extraction/12_ner_extraction.ipynb)
- 2.6 theory → [modules/phase_2/2.6_evaluation/2.6_evaluation.md](../modules/phase_2/2.6_evaluation/2.6_evaluation.md)
- 2.6 notebook → [modules/phase_2/2.6_evaluation/13_evaluation.ipynb](../modules/phase_2/2.6_evaluation/13_evaluation.ipynb)
- 2.7 theory → [modules/phase_2/2.7_distillation/2.7_distillation.md](../modules/phase_2/2.7_distillation/2.7_distillation.md)
- 2.7 notebook → [modules/phase_2/2.7_distillation/14_distillation.ipynb](../modules/phase_2/2.7_distillation/14_distillation.ipynb)

**Notebooks:** `08_data_acquisition`, `09_synthetic_data_gen`, `10_encoder_dataprep`, `11_encoder_finetune`, `12_field_extraction_ner`, `13_encoder_eval`, `14_distillation`

**Compute:** All free (Colab/Kaggle T4). Teacher LLM for Module 2.2: ~$1–2 via API, or free via local Ollama.

---

## Phase 3 — The Decoder SLM: Domain-Adapted Generation

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 3.0 | Prompting baselines — zero/few-shot *(new)* | ⬜ | Scored baseline report: zero-shot vs few-shot on ~100 gold examples | ☐ | | |
| 3.1 | Choosing a base model | ⬜ | Comparison table of 3 candidates + recommendation | ☐ | | |
| 3.1a *(opt)* | Domain-Adaptive Pretraining (DAPT) | ⬜ | DAPT checkpoint OR reasoned skip note + perplexity comparison | ☐ | | |
| 3.2 | Data prep for SFT (chat format + loss masking) | ⬜ | Formatted SFT dataset (~500–3k high-quality pairs) | ☐ | | |
| 3.2a | Structured / constrained decoding *(new)* | ⬜ | `outlines`-based extractor benchmarked vs NER head (Module 2.5) | ☐ | | |
| 3.3 | PEFT theory: LoRA & QLoRA | ⬜ | Annotated memory math diagram | ☐ | | |
| 3.4 | Run the QLoRA fine-tune (smoke-test free → rent) | ⬜ | QLoRA adapter + before/after sample outputs | ☐ | | |
| 3.5 | Is bigger actually better? (paid GPU) | ⬜ | 3-way table: 1–3B QLoRA vs full fine-tune vs 7–8B QLoRA | ☐ | | |
| 3.6 | Evaluating generative output | ⬜ | Eval harness + scorecard (tuned vs base) | ☐ | | |
| 3.7 *(opt)* | DPO preference tuning | ⬜ | DPO-tuned variant on tone/format | ☐ | | |

**Notebooks:** `15_prompting_baseline`, `16_base_model_selection`, `17_dapt_run`, `18_sft_dataset`, `19_constrained_decoding`, `20_lora_qlora_theory`, `21_qlora_finetune`, `22_bigger_vs_smaller`, `23_gen_eval_harness`, `24_dpo_tuning`

**Compute:** Modules 3.0–3.2a on free tier. Module 3.4: smoke-test free, full run ~1–2 hrs on rented A100 (~$2–4). Module 3.5: ~3 hrs on A100 (~$3–6).

---

## Phase 4 — Retrieval-Augmented Generation (Grounding)

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 4.1 | RAG vs fine-tuning decision framework | ⬜ | Decision note for DeskMate | ☐ | | |
| 4.2 | Chunking & embeddings | ⬜ | Chunked + embedded doc corpus in vector DB (FAISS/Chroma) | ☐ | | |
| 4.3 | Retrieval that actually works (hybrid + rerank) | ⬜ | Retriever measured by hit-rate@k | ☐ | | |
| 4.4 | Wire RAG to the decoder + evaluate grounding | ⬜ | End-to-end grounded answers with citations + faithfulness scorecard | ☐ | | |
| 4.5 *(opt)* | GraphRAG | ⬜ | GraphRAG experiment vs baseline | ☐ | | |

**Notebooks:** `25_rag_index`, `26_rag_retriever`, `27_rag_answer_eval`, `28_graphrag`

**Compute:** All free (CPU for indexing; T4 for embedding if needed).

---

## Phase 5 — Optimization for Production

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 5.1 | Precision formats (fp32/fp16/bf16/int8/int4) | ⬜ | Memory estimates notebook | ☐ | | |
| 5.2 | 8-bit quantization | ⬜ | 8-bit DeskMate benchmarked vs fp16 | ☐ | | |
| 5.3 | 4-bit quantization: GPTQ, AWQ, GGUF | ⬜ | GPTQ/AWQ build + GGUF export, each benchmarked | ☐ | | |
| 5.4 | ONNX export & runtime | ⬜ | Encoder + decoder exported to ONNX, benchmarked vs PyTorch | ☐ | | |
| 5.5 | Profiling & graph optimization | ⬜ | Flame view + before/after latency improvement | ☐ | | |
| 5.6 | Advanced quantization (read-level) | ⬜ | SmoothQuant / FlexGen / BitNet — one-sentence summary each | ☐ | | |
| 5.7 | Speculative decoding *(new)* | ⬜ | Speculative decoding benchmark: tokens/sec, acceptance rate, output quality | ☐ | | |
| 5.8 *(opt)* | Model merging: SLERP & DARE-TIES | ⬜ | Merged SFT+DPO model benchmarked vs each parent | ☐ | | |

**Notebooks:** `29_precision_formats`, `30_quantize_8bit`, `31_quantize_4bit_gguf`, `32_onnx_export`, `33_onnx_profile`, `34_advanced_quant`, `35_speculative_decoding`, `36_model_merging`

**Compute:** All free (CPU + T4).

---

## Phase 6 — Deployment & Serving

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 6.1 | Inference economics: cost, batching, throughput | ⬜ | Cost analysis notebook | ☐ | | |
| 6.2 | Serving with vLLM | ⬜ | DeskMate served via vLLM, load-tested | ☐ | | |
| 6.3 | FastAPI service + benchmarking | ⬜ | `/triage` + `/answer` API with latency benchmarks | ☐ | | |
| 6.4 | Local deployment: Ollama, llama.cpp | ⬜ | DeskMate running locally via Ollama from your GGUF | ☐ | | |
| 6.5 | On-device / mobile (overview) | ⬜ | Conceptual notes + MLC LLM demo | ☐ | | |

**Notebooks:** `37_serve_vllm_fastapi`, `38_local_ollama`

**Compute:** Free (CPU for FastAPI/Ollama; T4 for vLLM load test).

---

## Phase 7 — End-to-End Application & Agent

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 7.1 | System architecture & orchestration | ⬜ | Integrated DeskMate service (triage → route → retrieve → generate → guardrails) | ☐ | | |
| 7.2 | Make it an agent | ⬜ | DeskMate with 1–2 real tools (e.g. order lookup, ticket escalation) | ☐ | | |
| 7.3 | Memory: short- and long-term | ⬜ | Multi-turn DeskMate with conversation + persistent memory | ☐ | | |
| 7.4 | Monitoring, eval-in-prod, drift detection | ⬜ | Monitoring + feedback loop design + basic dashboards | ☐ | | |
| 7.5 *(opt)* | Test-time compute | ⬜ | DeskMate with test-time compute on hard tickets: quality/latency trade | ☐ | | |
| 7.6 | CI/CD for the model lifecycle *(new)* | ⬜ | CI/CD pipeline: drift trigger → fine-tune → eval gate → promote or reject | ☐ | | |

**Notebooks:** `39_deskmate_orchestrator`, `40_deskmate_agent_memory`, `41_monitoring_drift`, `42_test_time_compute`, `43_cicd_model_lifecycle`

---

## Phase 8 — DeskMate Capstone & Portfolio

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 8 | Ship DeskMate + write-up | ⬜ | Deployed demo (vLLM/FastAPI + local Ollama), model card, datasheet, eval scorecard, quantization report, architecture doc, portfolio writeup | ☐ | | |

**Notebook:** `44_capstone_writeup`

---

## Phase 9 — Graduation Project (new domain, solo)

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 9.1 | Pick domain + define problem | ⬜ | Problem statement + data plan | ☐ | | |
| 9.2 | Source / generate data | ⬜ | Dataset on Hub + datasheet + gold set | ☐ | | |
| 9.3 | Train model (encoder and/or decoder) | ⬜ | Fine-tuned model(s) | ☐ | | |
| 9.4 | Ground with RAG (if knowledge-heavy) | ⬜ | RAG pipeline (or reasoned skip) | ☐ | | |
| 9.5 | Optimize (quantize / ONNX) | ⬜ | Optimized model with benchmarks | ☐ | | |
| 9.6 | Deploy + serve | ⬜ | Live API + local build | ☐ | | |
| 9.7 | Write-up + model card | ⬜ | Complete portfolio piece | ☐ | | |

**Candidate domains:** SQL/codegen assistant · Clinical NER + Q&A · Legal-clause classifier · Fintech dispute assistant · Scientific structure generation

---

## Overall Progress

```
Phase 0  [⬜⬜⬜⬜]          0 / 4  modules
Phase 1  [⬜⬜⬜⬜⬜⬜⬜]     0 / 7  modules
Phase 2  [⬜⬜⬜⬜⬜⬜⬜]     0 / 7  modules (1 optional)
Phase 3  [⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜] 0 / 10 modules (2 optional)
Phase 4  [⬜⬜⬜⬜⬜]          0 / 5  modules (1 optional)
Phase 5  [⬜⬜⬜⬜⬜⬜⬜⬜]    0 / 8  modules (1 optional)
Phase 6  [⬜⬜⬜⬜⬜]          0 / 5  modules
Phase 7  [⬜⬜⬜⬜⬜⬜]        0 / 6  modules (1 optional)
Phase 8  [⬜]                 0 / 1  modules
Phase 9  [⬜⬜⬜⬜⬜⬜⬜]     0 / 7  modules
─────────────────────────────────────────────
Total    0 / 60 modules   (6 optional, can skip)
```

---

## Key decisions log

> Record significant design or scope decisions here as you go. This prevents re-debating them later.

| Date | Decision | Rationale |
|---|---|---|
| — | *(fill in as you work)* | |

---

## Paid GPU spend log

> Track rented GPU time to stay within budget.

| Date | Module | Provider | GPU | Duration | Cost |
|---|---|---|---|---|---|
| — | 3.4 QLoRA fine-tune | — | A100 | ~2 hrs | ~$2–4 |
| — | 3.5 bigger-vs-smaller | — | A100 | ~3 hrs | ~$3–6 |
| — | *(other)* | | | | |

**Estimated total course GPU spend:** $10–20 if smoke-testing is done on free tier first.

---

## Checkpoint questions quick-reference

> Use this to self-test before marking a module Done.

| Module | Checkpoint question |
|---|---|
| 0.1 | Give 3 production scenarios and argue SLM vs LLM for each in 2 sentences. |
| 0.2 | Your runtime dies mid-training. Can you resume with zero progress loss? |
| 0.3 | Why does a decoder use a causal mask but an encoder doesn't? |
| 0.4 | Why can the same sentence cost very different token counts across models? |
| 1.1 | What exactly is the model predicting at each position, and what is the loss measuring? |
| 1.2 | Remove the causal mask — what breaks, and why? |
| 1.3 | What do residual connections buy you as depth grows? |
| 1.4 | Explain the full path from a prompt string to the next generated token. |
| 1.5 | Given a fixed compute budget, train bigger or train longer — why is this the wrong question without data? |
| 1.6 | Why does GQA reduce KV-cache size, and why does that matter more for serving than training? |
| 1.7 | DAPT and SFT both train on domain text — what is fundamentally different about their objectives and data? |
| 2.1 | Name two leakage traps in support data and how you avoided them. |
| 2.2 | How do you keep synthetic data diverse, and why is a real-data anchor essential? |
| 2.3 | What does the attention mask do for padded sequences? |
| 2.4 | Val accuracy is high but production accuracy is low — list three likely causes. |
| 2.5 | Why does subword tokenization complicate token-level labels, and how do you align them? |
| 2.6 | When is macro-F1 the right metric over accuracy? |
| 2.7 | What does the student learn from soft labels that hard labels don't teach? |
| 3.0 | Few-shot baseline hits 72%. How do you decide if fine-tuning to 88% is worth the effort? |
| 3.1 | Why might you pick a base model over an instruct model for fine-tuning? |
| 3.1a | DAPT lowers perplexity but SFT quality doesn't improve — what does that tell you? |
| 3.2 | Why do we mask the prompt tokens from the loss? |
| 3.2a | Constrained decoding guarantees valid JSON structure. What does it NOT guarantee? |
| 3.3 | What fraction of parameters does LoRA train, and why does that slash memory? |
| 3.4 | Which two memory-saving techniques let QLoRA fit in 16 GB, and what do they cost you? |
| 3.5 | At what quality gap would the bigger model become worth its cost for DeskMate? |
| 3.6 | Name two failure modes of LLM-as-judge and how you'd mitigate them. |
| 3.7 | What kind of problem does preference tuning fix that more SFT data won't? |
| 4.1 | Your docs change weekly — fine-tune or RAG? Why? |
| 4.2 | How do chunk size and overlap trade off recall against noise? |
| 4.3 | When does pure vector search fail and BM25 save you? |
| 4.4 | Retrieval is perfect but answers still hallucinate — where do you look? |
| 4.5 | What query type does GraphRAG help with that top-k vector search can't? |
| 5.1 | Estimate memory to load a 3B model in fp16 vs int4. |
| 5.2 | What problem do outlier features cause, and how does LLM.int8() handle it? |
| 5.3 | GPTQ vs GGUF — when do you reach for each? |
| 5.4 | What does I/O binding remove from the hot path? |
| 5.5 | What's your single slowest op, and what did you do about it? |
| 5.6 | One sentence each on what SmoothQuant, FlexGen, and BitNet solve. |
| 5.7 | Why does correct speculative decoding produce outputs identical to standard decoding? |
| 5.8 | What symptom tells you weight interference is degrading the merged model? |
| 6.1 | Why does batching raise throughput but can hurt p99 latency? |
| 6.2 | What does paged attention do for memory? |
| 6.3 | Where do you put the encoder vs decoder vs retriever in the request path? |
| 6.4 | Why is local deployment a feature, not just a convenience, for some customers? |
| 6.5 | What changes when your "server" is a phone? |
| 7.1 | Where does low-confidence triage short-circuit the pipeline? |
| 7.2 | What guardrail prevents an agent tool-call loop from running forever? |
| 7.3 | How do you keep a long conversation inside the context window? |
| 7.4 | What signal tells you it's time to retrain? |
| 7.5 | When is paying more compute per request worth it? |
| 7.6 | Your eval gate compares against only the gold set. What blind spot does this leave? |
