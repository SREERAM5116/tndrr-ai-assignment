# tndrr AI/ML Internship – Assignment

Three tasks, one folder each. All experiments were run on a free Google Colab **Tesla T4 (16 GB)** GPU. Every number below comes from these runs.

| Folder | Task |
|---|---|
| [`task1_finetune/`](task1_finetune) | LoRA fine-tuning of Qwen2.5-0.5B-Instruct for expense-message → JSON |
| [`task2_rag/`](task2_rag) | Retrieval-augmented QA from scratch (FAISS + BGE embeddings, no frameworks) |
| [`task3_local_inference/`](task3_local_inference) | Running Qwen2.5-1.5B-Instruct with Ollama at FP16 vs 4-bit (Q4_K_M) |

The full write-up with all answers is in the PDF report submitted with this repository.

---

## Task 1 – Fine-tuning with LoRA

**Task:** turn a casual expense message into fixed JSON, e.g.
`"Paid Swiggy Rs 450 for biryani via UPI"` → `{"amount": 450, "currency": "INR", "category": "food", "merchant": "Swiggy", "payment_method": "upi"}`

- Data: 1,000 generated training examples, 100 held-out test examples (zero overlap). The test set is deliberately harder: merchants, sentence templates, payment phrases and amount formats (e.g. "₹1.5k") that never appear in training.
- LoRA: rank 16, alpha 32, dropout 0.05, all attention + MLP projections, learning rate 2e-4, 2 epochs, effective batch 16.

| Metric (100 test examples) | Base model | Fine-tuned |
|---|---|---|
| **Exact match (all 5 fields)** | **7%** | **89%** |
| Valid JSON | 100% | 100% |
| Amount | 100% | 97% |
| Currency | 87% | 100% |
| Category | 40% | 100% |
| Merchant | 66% | 94% |
| Payment method | 40% | 97% |

Remaining errors fall into three groups: "k" amounts (₹6.7k → 67000), payment apps never seen in training (PhonePe/Paytm), and an unusual spelling (Smaaash → Samaaash).

Files: `task1_finetune.ipynb`, `train.jsonl`, `test.jsonl`, `task1_base_vs_finetuned.csv`, `task1_test_predictions.csv`, `lora_adapter/adapter_config.json`.
(The trained adapter weights, `adapter_model.safetensors`, are 35 MB, above GitHub's web-upload limit; they are reproduced by running the notebook.)

## Task 2 – RAG from scratch

- Documents: the LoRA paper (arXiv 2106.09685) and the QLoRA paper (arXiv 2305.14314), about 50 pages.
- Pipeline: pypdf → word-window chunks → `BAAI/bge-small-en-v1.5` embeddings → FAISS `IndexFlatIP` (cosine) → top-5 chunks → Qwen2.5-1.5B-Instruct writes the answer with page citations.
- 15 test questions, each with an exact evidence phrase from the source passage. A retrieved chunk is a hit if it contains that phrase.

| Retrieval (15 questions) | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|
| Small chunks (100 words) | 6.7% | **40.0%** | 46.7% | 0.226 |
| Large chunks (300 words) | 6.7% | 26.7% | 60.0% | 0.184 |

| recall@5 | Small | Large |
|---|---|---|
| Embedding only | 40.0% | 26.7% |
| BM25 only | 46.7% | 33.3% |
| **Hybrid (RRF)** | **60.0%** | **66.7%** |

Files: `task2_rag.ipynb`, `task2_questions.csv`, `task2_chunk_size_results.csv`, `task2_per_question_ranks.csv`, `task2_method_comparison.csv`.

## Task 3 – Local inference and quantisation

Qwen2.5-1.5B-Instruct served with Ollama, same 10 prompts, temperature 0, context 2048, max 300 new tokens.

| Setting | Avg speed | GPU memory (Ollama) |
|---|---|---|
| FP16 | 65.7 tok/s | 3,067 MB |
| Q4_K_M | 135.2 tok/s | 1,057 MB |

Quality on the 10 prompts: 6 the same, 2 slightly worse with Q4, 1 clearly worse (did not follow "nothing else"), 1 failed in both (Hindi translation).

Files: `task3_local_inference.ipynb`, `task3_results.csv` (all 20 answers with tokens/sec).

---

## How to reproduce

Open any notebook in Google Colab, choose **Runtime → Change runtime type → T4 GPU**, and run the cells in order. Each notebook installs what it needs.

## Note on tools

I used an AI assistant (Claude) as a learning aid while building this: to help write and debug the notebook code and to explain concepts. I ran every experiment myself, and all results are from my own runs.
