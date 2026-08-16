# Pretrain-LM: Building and Domain-Adapting a Small Language Model from Scratch

A **32.7M-parameter decoder-only Transformer** implemented and pretrained from scratch, then continued-pretrained on a specialized medical-domain corpus and evaluated through structural, corpus-grounded, expert, and held-out evaluations.

This project began as a way to understand language models below the API level before moving deeper into **LLM research**. Instead of treating an existing model as a black box, I built the language-model stack directly in PyTorch and took it through tokenizer training, general-language pretraining, domain adaptation, checkpoint selection, and evaluation.

> **Current status:** Model development and domain adaptation are complete. The resulting model will next serve as a controlled system for LLM security and mechanistic experiments.

---

## Key Results

| Metric | Base Model | Domain-Adapted Model (Epoch 4) | Change |
|---|---:|---:|---:|
| **Medical-domain test loss ↓** | 6.1217 | **3.4565** | **−43.54%** |
| **Medical-domain test perplexity ↓** | 455.64 | **31.70** | −93.04% |
| WikiText test loss ↓ | **3.5562** | 4.4561 | +0.8999 |
| WikiText test perplexity ↓ | **35.03** | 86.15 | +145.94% |

Continued domain pretraining produced a substantial improvement on **unseen medical-domain text**, reducing held-out cross-entropy loss by **43.5%** and perplexity from **455.6 to 31.7**.

The improvement was not free: performance on held-out WikiText deteriorated after specialization. This exposes a clear **domain-specialization vs. general-language retention tradeoff**, rather than suggesting an across-the-board improvement.

Importantly, **Epoch 4 was selected before the held-out test sets were evaluated**.

---

## Model Architecture

The model is a decoder-only autoregressive Transformer trained for causal next-token prediction.

| Component | Configuration |
|---|---:|
| Parameters | **32,714,496** |
| Vocabulary size | 24,000 |
| Context length | 256 tokens |
| Transformer layers | 8 |
| Attention heads | 6 |
| Embedding dimension | 384 |
| Feed-forward dimension | 1,536 |
| Training objective | Causal next-token prediction |

### Forward Path

```text
Token IDs
   │
   ▼
Token Embeddings + Positional Embeddings
   │
   ▼
┌─────────────────────────────────┐
│      Transformer Block × 8      │
│                                 │
│  LayerNorm                      │
│      ↓                          │
│  Causal Multi-Head Attention    │
│      ↓                          │
│  Residual Connection            │
│                                 │
│  LayerNorm                      │
│      ↓                          │
│  Feed-Forward Network           │
│      ↓                          │
│  Residual Connection            │
└─────────────────────────────────┘
   │
   ▼
Final LayerNorm
   │
   ▼
Language-Model Head
   │
   ▼
Next-Token Logits
```

The Transformer was implemented directly in **PyTorch**, rather than assembled from a pretrained model library. Embeddings, causal self-attention, Transformer blocks, the Transformer stack, and the language-model wrapper are separated in the codebase, keeping the model's internal computation explicit and accessible for later experimentation.

---

## Building and Training the Model

The project progressed from a small Transformer implementation into a complete pretraining, domain-adaptation, and evaluation pipeline.

```text
WikiText-103
      │
      ▼
24K BPE Tokenizer
      │
      ▼
Base Pretraining
      │
      ▼
32.7M-Parameter Base LM
      │
      ▼
Medical-Domain Continued Pretraining
      │
      ▼
5 Domain Checkpoints
      │
      ▼
Rule-Based + NLI + Expert Evaluation
      │
      ▼
Selected Epoch 4 Model
      │
      ▼
Untouched Test Sets
```

### Base Pretraining

The base model was pretrained on **WikiText-103** for five epochs using causal next-token prediction.

The training configuration included:

- **24,000-token BPE vocabulary**
- **256-token context length**
- batch size of **32**
- AdamW optimization
- initial learning rate of **3 × 10⁻⁴**
- minimum learning rate of **3 × 10⁻⁵**
- weight decay of **0.01**
- validation during training
- checkpointing with optimizer and scheduler state
- resumable training
- autoregressive generation for qualitative inspection

The final base checkpoint was reached after **71,605 training steps**, with a best WikiText validation loss of **3.5351**.

### Medical-Domain Adaptation

The base model was then continued-pretrained on a specialized medical-domain corpus drawn from **homeopathic literature**.

The domain dataset contained:

| Split | Tokens |
|---|---:|
| Training | **3,162,676** |
| Validation | 280,238 |
| Held-out test | 322,118 |

The architecture, tokenizer, context length, and batch size remained unchanged. Continued pretraining used a lower initial learning rate of **1 × 10⁻⁴**, decaying to **1 × 10⁻⁵**, and ran for five epochs.

Later checkpoints showed continued improvement in medical-domain validation loss alongside progressive deterioration in general-language validation:

| Checkpoint | Domain Validation Loss ↓ | WikiText Validation Loss ↓ |
|---|---:|---:|
| Epoch 3 | 3.4492 | **4.4200** |
| **Epoch 4** | **3.4312** | 4.4511 |
| Epoch 5 | 3.4271 | 4.4723 |

Epoch 5 achieved a marginally lower domain validation loss than Epoch 4. However, language-model loss alone does not establish whether generated content is structurally sound or grounded in the source corpus. Final checkpoint selection therefore used additional downstream evaluations.

---

## Evaluation & Checkpoint Selection

Evaluation became a significant part of the project because improved next-token prediction does not necessarily imply improved generated content.

Rather than selecting a checkpoint using a single number, I evaluated the later checkpoints through **three complementary approaches**. A **rule-based structural evaluation** tested measurable properties of generated outputs; a **corpus-grounded NLI evaluation** retrieved relevant source passages and tested whether generated claims were actually supported by them; and a **blinded domain-expert evaluation** provided an independent qualitative assessment of domain correctness. Together, these evaluations favored **Epoch 4**, despite Epoch 5 having the slightly lower domain validation loss.

### Corpus-Grounded Support

For the factual component, I built a retrieval-and-inference pipeline instead of relying on an LLM to freely judge whether generated text "looked correct."

```text
Generated Text
      │
      ▼
Atomic Claims
      │
      ▼
Remedy-Aware Retrieval
      │
      ▼
Semantic Passage Retrieval
      │
      ▼
Retrieved Corpus Evidence
      │
      ▼
DeBERTa-v3 NLI
      │
      ▼
SUPPORTED / NOT ESTABLISHED
```

The medical corpus was split into retrievable passages and embedded for semantic search. Retrieval was made **remedy-aware** to reduce contamination from similarly named but distinct entries in the corpus.

Retrieved evidence was then evaluated against generated claims using the `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` NLI model.

A conservative consensus criterion was used for grounded support:

```text
SUPPORTED
    max entailment ≥ 0.70
    AND top-2 mean entailment ≥ 0.55

otherwise
    NOT ESTABLISHED
```

`NOT ESTABLISHED` does not mean that a claim is necessarily false. It means that the evaluation pipeline could not establish sufficient support for it from the retrieved corpus evidence.

The later checkpoints produced:

| Checkpoint | Grounded Support |
|---|---:|
| Epoch 3 | 23.4% |
| **Epoch 4** | **30.6%** |
| Epoch 5 | 9.7% |

This provided a useful result that validation loss alone missed: **Epoch 5 continued improving slightly at next-token prediction while its generated claims became substantially less well-supported by retrieved domain evidence.**

### Expert Evaluation

Automated evaluation was complemented by a **blinded assessment from a practicing domain expert**.

Generated samples were assessed for domain correctness, helping identify outputs that could appear linguistically plausible while still containing substantive domain-specific errors. This provided a human signal complementary to the deterministic and NLI-based evaluations.

The expert evaluation was intentionally lightweight and is treated as supporting evidence rather than a standalone statistical benchmark.

---

## Final Held-Out Evaluation

After Epoch 4 was selected, it and the original base model were evaluated on untouched test sets covering **322,048 predicted medical-domain tokens** and **280,576 predicted WikiText tokens**.

| Dataset | Base Loss | E4 Loss | Base PPL | E4 PPL |
|---|---:|---:|---:|---:|
| **Medical domain** | 6.1217 | **3.4565** | 455.64 | **31.70** |
| **WikiText** | **3.5562** | 4.4561 | **35.03** | 86.15 |

The selected model reduced held-out medical-domain loss by 43.54%, confirming that the specialization generalized beyond the training and validation data. This came with a substantial degradation in WikiText performance, demonstrating the tradeoff between domain specialization and general-language retention.

---

## Repository Structure

```text
pretrain-lm/
│
├── model/
│   ├── embeddings.py
│   ├── self_attention.py
│   ├── transformer_block.py
│   ├── transformer.py
│   └── language_model.py
│
├── tokenizer/
│   ├── output/
│   │   └── tokenizer.json
│   └── scripts/
│
├── training/
│   ├── prepare_dataset.py
│   ├── tokenize_datasets.py
│   ├── train_base.py
│   └── train_domain.py
│
├── evaluation/
│   ├── compare_checkpoints.py
│   ├── structure/
│   ├── factuality/
│   │   ├── build_retrieval_index.py
│   │   ├── build_embeddings.py
│   │   ├── remedy_aliases.py
│   │   └── evaluate_grounded_support.py
│   └── final_test/
│       └── final_test_evaluation.py
│
├── results/
│   ├── final_test_results.json
│   └── final_test_results.csv
│
├── scripts/
├── requirements.txt
└── README.md
```

Training corpora, tokenized datasets, model checkpoints, embedding caches, and other large generated artifacts are intentionally excluded from the repository.

### Setup

```bash
git clone https://github.com/DevAFK25/pretrain-lm.git
cd pretrain-lm

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

The repository contains the model architecture, tokenizer, training pipelines, evaluation code, and final evaluation results. Re-running training or corpus-grounded evaluation requires the corresponding datasets and checkpoints, which are not distributed through this repository.

---

## Limitations & Next Steps

This is a **small experimental language model**, not a production or clinical system. Homeopathic literature was used as the specialized medical-domain corpus for adaptation and evaluation; generated outputs should not be interpreted as medical advice.

Several technical limitations remain:

- The **32.7M-parameter model** and **256-token context** are intentionally small by modern LLM standards.
- Domain specialization caused substantial degradation on WikiText, demonstrating incomplete retention of general-language capability.
- Corpus-grounded support measures whether retrieved source material supports a generated claim; it does **not** establish universal medical truth.
- Retrieval quality places an upper bound on the grounded-support evaluation.
- Expert evaluation was deliberately lightweight and complements rather than replaces systematic automated evaluation.

### Next: LLM Security Experiments

The original motivation for implementing the model from the architecture upward was to understand language models internally before studying their security.

With the architecture, weights, training pipeline, intermediate checkpoints, and evaluation framework under direct control, the model now provides a compact testbed for **LLM security and mechanistic experiments**.

The next phase will investigate internal model behavior — including **residual-stream representations, controlled interventions, and changes in internal representations under security-relevant or adversarial conditions**.
