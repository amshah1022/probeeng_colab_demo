# 🔬 ProbeEng — Layer-Wise Probing for Mechanistic Interpretability

**Affiliation:** Long-term AI Safety Lab, Cornell University  
**Lead Developer:** Lionel Levine
**Goal:** A scalable pipeline for training and evaluating probes over LLM internal representations (layers, heads, token positions) with reproducible presets and artifact management.

> ProbeEng is the **interpretability pillar** of an AI Reliability stack:
> **Truth Layer** (evidence-grounded verification) ⇄ **ProbeEng** (mechanistic interpretability) ⇄ **Rubric Feedback** (human-feedback calibration).

---

## What ProbeEng Does

- **Representation extraction** from HuggingFace-compatible LLMs (e.g., Llama-2, GPT-2, Pythia).  
- **Probe training** (e.g., LR, DIM, LDA; unsupervised PCA/CCS/LAT) across **selected layers** and **token positions**.  
- **Evaluation** with cross-dataset generalization, rankings, and plots.  
- **Ensemble support** (e.g., Parallel Projection) for combining methods.  
- **Artifact I/O** to **S3** for sharing datasets, probes, and evaluation results across runs.

---

## Pipeline Architecture

**Pipelines** (pick one):
- **Standard:** Dataset Creation → Probe Training → Probe Evaluation  
- **Ensemble:** Dataset Creation → Ensemble Training → Ensemble Evaluation  
- **Comparison:** Dataset Creation → Probes + Ensemble → Comparison Evaluation

**Stages**:
1) **Activation Extraction**: forward hooks capture hidden states at chosen layers and tokens → serialized activation dataset (pickle + metadata).  
2) **Probe / Ensemble Training**: train per-layer probes or a learned ensemble across methods.  
3) **Evaluation**: compute metrics (accuracy, F1, MI), same-vs-cross dataset generalization, rankings, and plots.

---


---

## Quick Start

> **Requirements:** Python 3.10+, CUDA GPU recommended. For gated models (e.g., `meta-llama/Llama-2-7b-chat-hf`), add a Hugging Face token with access.

```bash
git clone https://github.com/SamuelS0/ProbeEng
cd ProbeEng
pip install -e .
```

## Run a Pipeline (Standard)

```bash
# List available presets
python experiments/run_pipeline.py standard --list

# Minimal end-to-end run
python experiments/run_pipeline.py standard debug

# Full standard preset (example)
python experiments/run_pipeline.py standard standard
```

## Other workflows 

```bash
# Ensemble pipeline
python experiments/run_pipeline.py ensemble comprehensive

# Comparison: probes vs ensemble
python experiments/run_pipeline.py comparison comprehensive
```

## MVP Preset 
```bash
# experiments/presets/dataset_creation.toml
[presets.alina_mvp]
tag = "alina_mvp_dataset"
llm_ids = ["meta-llama/Llama-2-7b-chat-hf"]
dataset_collections = ["got"]
num_tokens_from_end = 1
device = "cuda"
layers_start = 1
layers_end   = 19
layers_skip  = 2

# experiments/presets/probe_training.toml
[presets.alina_mvp]
tag = "alina_mvp_train"
llm_ids = ["meta-llama/Llama-2-7b-chat-hf"]
dataset_collections = ["got"]
probe_methods = ["lr"]        # logistic regression (or "dim", "lda", "pca", "ccs", "lat")
layers_start = 1
layers_end   = 19
layers_skip  = 2
token_idxs   = [0]            # because num_tokens_from_end=1

# experiments/presets/probe_evaluation.toml
[presets.alina_mvp]
tag = "alina_mvp_eval"
datasets = []                 # evaluate on collection below
dataset_collections = ["got"]
eval_splits = ["validation"]
output_formats = ["csv","json"]
generate_plots = true
advanced_plots = true
calculate_recovered_accuracy = true
generate_rankings = true
```

# Run
```bash
python experiments/run_pipeline.py standard alina_mvp --output-dir ./output/
```



