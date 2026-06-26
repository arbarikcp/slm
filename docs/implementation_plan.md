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

**Module files:**
- 3.0 theory → [modules/phase_3/3.0_prompting_baselines/3.0_prompting_baselines.md](../modules/phase_3/3.0_prompting_baselines/3.0_prompting_baselines.md)
- 3.0 notebook → [modules/phase_3/3.0_prompting_baselines/15_prompting_baselines.ipynb](../modules/phase_3/3.0_prompting_baselines/15_prompting_baselines.ipynb)
- 3.1 theory → [modules/phase_3/3.1_base_model_selection/3.1_base_model_selection.md](../modules/phase_3/3.1_base_model_selection/3.1_base_model_selection.md)
- 3.1a theory → [modules/phase_3/3.1a_dapt/3.1a_dapt.md](../modules/phase_3/3.1a_dapt/3.1a_dapt.md) *(optional — deferred; see TODO in file)*
- 3.2 theory → [modules/phase_3/3.2_sft_dataprep/3.2_sft_dataprep.md](../modules/phase_3/3.2_sft_dataprep/3.2_sft_dataprep.md)
- 3.2 notebook → [modules/phase_3/3.2_sft_dataprep/16_sft_dataprep.ipynb](../modules/phase_3/3.2_sft_dataprep/16_sft_dataprep.ipynb)
- 3.2a theory → [modules/phase_3/3.2a_constrained_decoding/3.2a_constrained_decoding.md](../modules/phase_3/3.2a_constrained_decoding/3.2a_constrained_decoding.md)
- 3.2a notebook → [modules/phase_3/3.2a_constrained_decoding/17_constrained_decoding.ipynb](../modules/phase_3/3.2a_constrained_decoding/17_constrained_decoding.ipynb)
- 3.3 theory → [modules/phase_3/3.3_lora_qlora/3.3_lora_qlora.md](../modules/phase_3/3.3_lora_qlora/3.3_lora_qlora.md)
- 3.4 theory → [modules/phase_3/3.4_qlora_finetune/3.4_qlora_finetune.md](../modules/phase_3/3.4_qlora_finetune/3.4_qlora_finetune.md)
- 3.4 notebook → [modules/phase_3/3.4_qlora_finetune/18_qlora_finetune.ipynb](../modules/phase_3/3.4_qlora_finetune/18_qlora_finetune.ipynb)
- 3.5 theory → [modules/phase_3/3.5_bigger_vs_smaller/3.5_bigger_vs_smaller.md](../modules/phase_3/3.5_bigger_vs_smaller/3.5_bigger_vs_smaller.md)
- 3.5 notebook → [modules/phase_3/3.5_bigger_vs_smaller/19_bigger_vs_smaller.ipynb](../modules/phase_3/3.5_bigger_vs_smaller/19_bigger_vs_smaller.ipynb)
- 3.6 theory → [modules/phase_3/3.6_gen_eval/3.6_gen_eval.md](../modules/phase_3/3.6_gen_eval/3.6_gen_eval.md)
- 3.6 notebook → [modules/phase_3/3.6_gen_eval/20_gen_eval_harness.ipynb](../modules/phase_3/3.6_gen_eval/20_gen_eval_harness.ipynb)
- 3.7 theory → [modules/phase_3/3.7_dpo/3.7_dpo.md](../modules/phase_3/3.7_dpo/3.7_dpo.md) *(optional — run only if 3.6 shows tone/safety issues)*
- 3.7 notebook → [modules/phase_3/3.7_dpo/21_dpo_tuning.ipynb](../modules/phase_3/3.7_dpo/21_dpo_tuning.ipynb)

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

**Notebooks:** `22_rag_chunking_embeddings`, `23_rag_retriever`, `24_rag_answer_eval`, `25_graphrag`

**Module files:**
- 4.1 theory → [modules/phase_4/4.1_rag_vs_finetune/4.1_rag_vs_finetune.md](../modules/phase_4/4.1_rag_vs_finetune/4.1_rag_vs_finetune.md)
- 4.1 deliverable → [modules/phase_4/4.1_rag_vs_finetune/decision_note_rag_vs_finetune.md](../modules/phase_4/4.1_rag_vs_finetune/decision_note_rag_vs_finetune.md)
- 4.2 theory → [modules/phase_4/4.2_chunking_embeddings/4.2_chunking_embeddings.md](../modules/phase_4/4.2_chunking_embeddings/4.2_chunking_embeddings.md)
- 4.2 notebook → [modules/phase_4/4.2_chunking_embeddings/22_rag_chunking_embeddings.ipynb](../modules/phase_4/4.2_chunking_embeddings/22_rag_chunking_embeddings.ipynb)
- 4.3 theory → [modules/phase_4/4.3_retrieval/4.3_retrieval.md](../modules/phase_4/4.3_retrieval/4.3_retrieval.md)
- 4.3 notebook → [modules/phase_4/4.3_retrieval/23_rag_retriever.ipynb](../modules/phase_4/4.3_retrieval/23_rag_retriever.ipynb)
- 4.4 theory → [modules/phase_4/4.4_rag_decoder/4.4_rag_decoder.md](../modules/phase_4/4.4_rag_decoder/4.4_rag_decoder.md)
- 4.4 notebook → [modules/phase_4/4.4_rag_decoder/24_rag_answer_eval.ipynb](../modules/phase_4/4.4_rag_decoder/24_rag_answer_eval.ipynb)
- 4.5 theory → [modules/phase_4/4.5_graphrag/4.5_graphrag.md](../modules/phase_4/4.5_graphrag/4.5_graphrag.md) *(optional — revisit if multi-hop HR@3 < 0.70)*

**Compute:** All free (CPU for indexing; T4 for embedding if needed).

---

## Phase 5 — Optimization for Production

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 5.1 | Precision formats (fp32/fp16/bf16/int8/int4) | ✅ | Theory `.md` — memory formula, format table, quantisation trade-offs | ✅ | 2026-06-26 | Theory-only; checkpoint: 3B fp16=6 GB, int4=1.5 GB |
| 5.2 | 8-bit quantization | ✅ | Theory `.md` + `30_quantize_8bit.ipynb` — VRAM/throughput/ROUGE-L | ✅ | 2026-06-26 | LLM.int8() outlier handling; quality gate delta ≤ 0.01 |
| 5.3 | 4-bit quantization: GPTQ, AWQ, GGUF | ✅ | Theory `.md` + `31_quantize_4bit_gguf.ipynb` — GPTQ/AWQ/GGUF benchmarked | ✅ | 2026-06-26 | Checkpoint: GPTQ=GPU serving, GGUF=CPU/local |
| 5.4 | ONNX export & runtime | ✅ | Theory `.md` + `32_onnx_export.ipynb` — encoder/decoder ONNX, I/O binding benchmark | ✅ | 2026-06-26 | Checkpoint: I/O binding removes CPU↔GPU copies from hot path |
| 5.5 | Profiling & graph optimization | ✅ | Theory `.md` + `33_onnx_profile.ipynb` — Chrome trace, op ranking, fusion + thread fix | ✅ | 2026-06-26 | Checkpoint: MatMul ~54% of time; ORT optimizer + thread tuning |
| 5.6 | Advanced quantization (read-level) | ✅ | Theory `.md` — SmoothQuant / FlexGen / BitNet; one-sentence summaries | ✅ | 2026-06-26 | Read-level; no notebook |
| 5.7 | Speculative decoding *(new)* | ✅ | Theory `.md` + `35_speculative_decoding.ipynb` — speedup, α, γ sweep, identity gate | ✅ | 2026-06-26 | Checkpoint: rejection sampling guarantees marginal p on every token |
| 5.8 *(opt)* | Model merging: SLERP & DARE-TIES | ✅ | Theory `.md` + `36_model_merging.ipynb` — linear/SLERP/DARE-TIES vs parents | ✅ | 2026-06-26 | Optional; checkpoint: ROUGE-L < both parents = interference → increase DARE sparsity |

**Module files:**
- 5.1 theory → [modules/phase_5/5.1_precision_formats/5.1_precision_formats.md](../modules/phase_5/5.1_precision_formats/5.1_precision_formats.md)
- 5.2 theory → [modules/phase_5/5.2_quantize_8bit/5.2_quantize_8bit.md](../modules/phase_5/5.2_quantize_8bit/5.2_quantize_8bit.md)
- 5.2 notebook → [modules/phase_5/5.2_quantize_8bit/30_quantize_8bit.ipynb](../modules/phase_5/5.2_quantize_8bit/30_quantize_8bit.ipynb)
- 5.3 theory → [modules/phase_5/5.3_quantize_4bit_gguf/5.3_quantize_4bit_gguf.md](../modules/phase_5/5.3_quantize_4bit_gguf/5.3_quantize_4bit_gguf.md)
- 5.3 notebook → [modules/phase_5/5.3_quantize_4bit_gguf/31_quantize_4bit_gguf.ipynb](../modules/phase_5/5.3_quantize_4bit_gguf/31_quantize_4bit_gguf.ipynb)
- 5.4 theory → [modules/phase_5/5.4_onnx_export/5.4_onnx_export.md](../modules/phase_5/5.4_onnx_export/5.4_onnx_export.md)
- 5.4 notebook → [modules/phase_5/5.4_onnx_export/32_onnx_export.ipynb](../modules/phase_5/5.4_onnx_export/32_onnx_export.ipynb)
- 5.5 theory → [modules/phase_5/5.5_profiling/5.5_profiling.md](../modules/phase_5/5.5_profiling/5.5_profiling.md)
- 5.5 notebook → [modules/phase_5/5.5_profiling/33_onnx_profile.ipynb](../modules/phase_5/5.5_profiling/33_onnx_profile.ipynb)
- 5.6 theory → [modules/phase_5/5.6_advanced_quant/5.6_advanced_quant.md](../modules/phase_5/5.6_advanced_quant/5.6_advanced_quant.md)
- 5.7 theory → [modules/phase_5/5.7_speculative_decoding/5.7_speculative_decoding.md](../modules/phase_5/5.7_speculative_decoding/5.7_speculative_decoding.md)
- 5.7 notebook → [modules/phase_5/5.7_speculative_decoding/35_speculative_decoding.ipynb](../modules/phase_5/5.7_speculative_decoding/35_speculative_decoding.ipynb)
- 5.8 theory → [modules/phase_5/5.8_model_merging/5.8_model_merging.md](../modules/phase_5/5.8_model_merging/5.8_model_merging.md)
- 5.8 notebook → [modules/phase_5/5.8_model_merging/36_model_merging.ipynb](../modules/phase_5/5.8_model_merging/36_model_merging.ipynb)

**Compute:** All free (CPU + T4).

---

## Phase 6 — Deployment & Serving

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 6.1 | Inference economics: cost, batching, throughput | ✅ | Theory `.md` — cost formula, batching, KV cache, throughput/latency trade-off | ✅ | 2026-06-26 | Checkpoint: batching raises throughput but increases p99 (head-of-line blocking) |
| 6.2 | Serving with vLLM | ✅ | Theory `.md` — paged attention, continuous batching, OpenAI API | ✅ | 2026-06-26 | Checkpoint: paged attention drops KV cache waste from 30-60% to <4% |
| 6.3 | FastAPI service + benchmarking | ✅ | Theory `.md` + `37_serve_vllm_fastapi.ipynb` — /triage + /answer, concurrency benchmark | ✅ | 2026-06-26 | Checkpoint: encoder→retriever→decoder order; decoder via vLLM |
| 6.4 | Local deployment: Ollama, llama.cpp | ✅ | Theory `.md` + `38_local_ollama.ipynb` — Modelfile, full local pipeline | ✅ | 2026-06-26 | Checkpoint: local = privacy + compliance + offline availability |
| 6.5 | On-device / mobile (overview) | ✅ | Theory `.md` — MLC LLM, Android, mobile constraints | ✅ | 2026-06-26 | Checkpoint: 5 changes (shared RAM, throttling, battery, diversity, update path) |

**Module files:**
- 6.1 theory → [modules/phase_6/6.1_inference_economics/6.1_inference_economics.md](../modules/phase_6/6.1_inference_economics/6.1_inference_economics.md)
- 6.2 theory → [modules/phase_6/6.2_vllm/6.2_vllm.md](../modules/phase_6/6.2_vllm/6.2_vllm.md)
- 6.3 theory → [modules/phase_6/6.3_fastapi/6.3_fastapi.md](../modules/phase_6/6.3_fastapi/6.3_fastapi.md)
- 6.3 notebook → [modules/phase_6/6.3_fastapi/37_serve_vllm_fastapi.ipynb](../modules/phase_6/6.3_fastapi/37_serve_vllm_fastapi.ipynb)
- 6.4 theory → [modules/phase_6/6.4_ollama/6.4_ollama.md](../modules/phase_6/6.4_ollama/6.4_ollama.md)
- 6.4 notebook → [modules/phase_6/6.4_ollama/38_local_ollama.ipynb](../modules/phase_6/6.4_ollama/38_local_ollama.ipynb)
- 6.5 theory → [modules/phase_6/6.5_on_device/6.5_on_device.md](../modules/phase_6/6.5_on_device/6.5_on_device.md)

**Compute:** Free (CPU for FastAPI/Ollama; T4 for vLLM load test).

---

## Phase 7 — End-to-End Application & Agent

| # | Module | Status | Deliverable | Checkpoint passed? | Date | Notes |
|---|---|---|---|---|---|---|
| 7.1 | System architecture & orchestration | ✅ | Integrated DeskMate service (triage → route → retrieve → generate → guardrails) | ✅ Short-circuit after encoder if confidence < 0.70; eliminates retriever + LLM call for ambiguous tickets | 2026-06-26 | [7.1_orchestration.md](../modules/phase_7/7.1_orchestration/7.1_orchestration.md) · [39_orchestration.ipynb](../modules/phase_7/7.1_orchestration/39_orchestration.ipynb) |
| 7.2 | Make it an agent | ✅ | DeskMate with 1–2 real tools (lookup_order, file_bug, escalate_to_human); loop guard MAX_TURNS=5 + context-limit guard | ✅ Turn counter (MAX_TURNS) stops infinite loops; fires escalation not fabricated reply | 2026-06-26 | [7.2_agent.md](../modules/phase_7/7.2_agent/7.2_agent.md) · [40_deskmate_agent.ipynb](../modules/phase_7/7.2_agent/40_deskmate_agent.ipynb) |
| 7.3 | Memory: short- and long-term | ✅ | Multi-turn DeskMate with ConversationMemory (sliding window + summarise-and-replace) + CustomerProfile long-term store | ✅ Sliding window + summarise-and-replace + proactive token budgeting; invariant: prompt fits context window before each call | 2026-06-26 | [7.3_memory.md](../modules/phase_7/7.3_memory/7.3_memory.md) · [41_memory.ipynb](../modules/phase_7/7.3_memory/41_memory.ipynb) |
| 7.4 | Monitoring, eval-in-prod, drift detection | ✅ | Structured logging, online metrics, KL divergence + CUSUM drift detection, two-signal retrain trigger, 4-panel dashboard, CI gate | ✅ Two-signal retrain trigger: any 2 of {confidence<0.72, esc_rate>25%, KL>0.25, CUSUM alarm} for 24h | 2026-06-26 | [7.4_monitoring.md](../modules/phase_7/7.4_monitoring/7.4_monitoring.md) · [42_monitoring.ipynb](../modules/phase_7/7.4_monitoring/42_monitoring.ipynb) |
| 7.5 *(opt)* | Test-time compute | ✅ | Best-of-N, self-consistency, iterative refinement; TTC router for high-urgency/enterprise; quality/latency scatter chart | ✅ Pay more compute when: reliable verifier exists + N×latency < SLA + ticket is hard (delta ≥ 0.03 ROUGE-L) | 2026-06-26 | [7.5_test_time_compute.md](../modules/phase_7/7.5_test_time_compute/7.5_test_time_compute.md) · [43_test_time_compute.ipynb](../modules/phase_7/7.5_test_time_compute/43_test_time_compute.ipynb) |
| 7.6 | CI/CD for the model lifecycle *(new)* | ✅ | Full pipeline: trigger → QLoRA fine-tune → 4-check eval gate (incl. per-category F1) → canary 5% → rollback or promote; GitHub Actions YAML written | ✅ Gold set blind spot = live distribution gap; canary closes it with 24h real traffic before full rollout | 2026-06-26 | [7.6_cicd.md](../modules/phase_7/7.6_cicd/7.6_cicd.md) · [44_cicd.ipynb](../modules/phase_7/7.6_cicd/44_cicd.ipynb) |

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
Phase 7  [✅✅✅✅✅✅]        6 / 6  modules (1 optional) ✅ COMPLETE
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
