# Knowledge-Aware Evaluation for Legal Document Summarization

This branch contains an extension of the intent-based evaluation framework of [Mullick et al. (2022)](https://aclanthology.org/2022.lrec-1.495/) for legal document summarization.

**Authors:** Aditi Singh (E23CSEU1484), Vaishnavi (E23CSEU1537), Vasundhra Singh (E23CSEU1179) — Bennett University

## Overview

Standard summarization metrics (BLEU, ROUGE, BERTScore) measure surface-level overlap and do not capture whether a legal summary preserves the case-defining *intent* of a document. Mullick et al. (2022) introduced an Intent Metric based on exact-substring matching between annotated intent phrases and summary sentences, which addresses this gap but is brittle to paraphrase. We extend their framework along two axes:

1. **Semantic Intent Metric (SIM)** — a windowed cosine-similarity match over sentence-tuned MPNet embeddings, replacing the exact-substring match. SIM correctly identifies paraphrased intent phrases that the original metric misses.

2. **LLM-as-Judge** — a reference-free LLM-based evaluator that scores how well a summary preserves the legal intent of the source document, following recent work on LLM-based evaluation (Zheng et al., 2023; Liu et al., 2023).

## Headline Result

On 93 Indian legal documents, **SIM is the only automated metric showing a positive Spearman rank correlation with LLM-as-Judge** (ρ = +0.255, p = 0.014). BLEU, ROUGE-L, and the original exact-match Intent Metric all show weak negative correlations.

| Metric | Spearman ρ | p-value |
|---|---|---|
| BLEU | −0.173 | 0.097 |
| ROUGE-L | −0.164 | 0.116 |
| Original Intent F1 | −0.212 | 0.041 * |
| **SIM F1 (ours)** | **+0.255** | **0.014 *** |

\* indicates p < 0.05

The improvement is largest on cases involving abstract legal intent (Corruption, Land Dispute) and modest on cases with concrete physical descriptors (Murder, Robbery), suggesting semantic matching is most valuable when source phrases are likely to be paraphrased.

See `paper/paper.pdf` for the full write-up.

## Repository Structure

```
.
├── README.md
├── Knowledge_Aware_Legal_Eval_Indian.ipynb       Main notebook — Indian-Data run (executed)
├── Knowledge_Aware_Legal_Eval_Australian.ipynb   Australian-Data notebook (ready to run)
├── legal_dataset.tar.gz                           Indian-Data (93 docs) + Australian-Data (59 docs)
├── results/
│   ├── results_per_doc_indian.csv                Per-document metric scores (Indian)
│   └── summary_results_indian.json               Aggregated metrics + config (Indian)
└── paper/
    ├── paper.tex                                  IEEE LaTeX source
    ├── paper.pdf                                  Compiled PDF (5 pages, 2-column)
    ├── make_figures.py                            Script to regenerate figures from CSVs
    └── figures/                                   PDF + PNG versions of all 4 figures
```

## Running the Notebooks

### Google Colab (recommended)

1. Open `Knowledge_Aware_Legal_Eval_Indian.ipynb` in Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. Upload `legal_dataset.tar.gz` to the Colab Files panel (drag-and-drop).
4. `Runtime → Run all`.

For the Australian dataset, use `Knowledge_Aware_Legal_Eval_Australian.ipynb` with the same dataset archive.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_JUDGE_MODE` | `'llama_local'` | Backend: `'anthropic'`, `'openai'`, `'llama_local'`, or `'skip'` |
| `ANTHROPIC_API_KEY` | `''` | Required for `'anthropic'` backend |
| `OPENAI_API_KEY` | `''` | Required for `'openai'` backend |
| `SIM_THRESHOLD` | `0.55` | Initial cosine threshold (auto-tuned by sweep cell) |
| `MAX_DOCS` | `None` | Limit number of documents (set integer for quick runs) |
| `SUMMARY_RATIO` | `0.3` | Summary length as a fraction of source document length |

### LLM-as-Judge Backends

| Backend | API key | Quality | Cost |
|---|---|---|---|
| `anthropic` | yes | high (Claude Haiku) | ~$1–2 per full run |
| `openai` | yes | high (GPT-4o-mini) | ~$1–2 per full run |
| `llama_local` | no | medium (Qwen2.5-1.5B) | free, runs on Colab GPU |
| `skip` | no | – | – |

`llama_local` is the default for full reproducibility at zero cost.

## Dataset

The dataset originates from Mullick et al. (2022) and was curated and annotated as part of an earlier UG research effort.

- **Indian-Data** — 93 Indian Supreme Court / High Court judgments across four categories (Murder, Land Dispute, Robbery, Corruption). Intent phrases are manually annotated.
- **Australian-Data** — 59 Australian legal case reports across the same four categories. Intent phrases are extracted automatically by a JointBERT model trained on Indian-Data (transfer learning).

After extracting `legal_dataset.tar.gz`:

```
dataset/
├── indian_data/
│   ├── ind_text/         93 .txt files (1.txt to 93.txt)
│   ├── ind_phrases_2/    93 .txt files (intent phrase annotations)
│   └── ind_labels.csv
└── australian_data/
    ├── aus_text/         59 .txt files
    ├── aus_phrases/      59 .txt files
    └── aus_labels.csv
```

## Method

### Original Intent Metric (Mullick et al., 2022)

Binary similarity between intent phrase P and summary sentence O:

```
s_ij = 1 if P_i is a substring of O_j, else 0
```

Precision/recall/F1 are computed in the standard way.

### Semantic Intent Metric (SIM, ours)

Replaces the binary substring match with windowed cosine similarity over sentence-tuned MPNet embeddings:

```
s_ij = 1 if max_{w in windows(O_j, |P_i|)} cos(e(P_i), e(w)) >= τ, else 0
```

Two design choices matter:

1. **Windowed matching.** Comparing a short phrase to an entire long summary sentence dilutes the cosine score. Sliding a window of length ≈ phrase length across each sentence localises the comparison.
2. **Encoder choice.** We use `all-mpnet-base-v2` rather than Legal-BERT because mean-pooled token embeddings of legal phrases cluster too tightly to discriminate between matches and non-matches.

Threshold τ is selected by sweep over `[0.40, 0.80]`; for our experiments τ = 0.45.

### LLM-as-Judge

Structured prompt asking the LLM to rate intent preservation on a 1–5 Likert scale, returning JSON. Prompt design follows G-Eval (Liu et al., 2023) and Zheng et al. (2023).

## Reproducing the Figures

```bash
cd paper
python make_figures.py
```

Reads `../results/results_per_doc_indian.csv` and writes 4 figures (PDF + PNG) to `figures/`.

## Compiling the Paper

```bash
cd paper
pdflatex paper.tex
pdflatex paper.tex   # second pass for cross-references
```

Or open `paper.tex` in [Overleaf](https://overleaf.com).

## References

- Beltagy, I., Peters, M. E., and Cohan, A. (2020). Longformer: The long-document transformer. *arXiv:2004.05150*.
- Bhattacharya, P., Hiware, K., Rajgaria, S., Pochhi, N., Ghosh, K., and Ghosh, S. (2019). A comparative study of summarization algorithms applied to legal case judgments. *ECIR 2019*.
- Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., and Androutsopoulos, I. (2020). LEGAL-BERT: The Muppets straight out of Law School. *Findings of EMNLP 2020*.
- Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., and Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *EMNLP 2023*.
- Miller, D. (2019). Leveraging BERT for extractive text summarization on lectures. *arXiv:1906.04165*.
- Mullick, A., Nandy, A., Kapadnis, M. N., Patnaik, S., Raghav, R., and Kar, R. (2022). An evaluation framework for legal document summarization. *LREC 2022*.
- Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-Networks. *EMNLP 2019*.
- Song, K., Tan, X., Qin, T., Lu, J., and Liu, T.-Y. (2020). MPNet: Masked and permuted pre-training for language understanding. *NeurIPS 2020*.
- Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 Datasets and Benchmarks*.
