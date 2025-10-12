# ProbeEng: Meta-Probing Infrastructure for Reliable Interpretability (Alina Shah DEMO) 

**Role:** Research Assistant — Cornell Long-Term AI Safety Lab
**Affiliation:** Long-term AI Safety Lab, Cornell University  
**Keywords:** meta-probing, interpretability infrastructure, AI reliability, evaluation science

---

## Purpose 

ProbeEng is a research framework for meta-probing — building, evaluating, and orchestrating probe networks that measure what large language models know and how reliably they express it.
Instead of interpreting a single model directly, ProbeEng focuses on the infrastructure layer of interpretability:
Designing standardized probe pipelines to train, test, and ensemble thousands of probes across layers, models, and datasets
Developing a meta-probe system that learns to evaluate and calibrate other probes’ reliability
This transforms probing from an ad-hoc interpretability technique into a scientific instrument for reproducible model-understanding research. ProbeEng complements mechanistic interpretability by focusing on the measurement layer—quantifying how well interpretability itself works.

---

## Core capabilities  

- End-to-End Pipeline — Dataset creation → Probe training → Evaluation → Visualization
- Cross-Model Analysis — Train probes on one model (e.g., GPT-2) and test on another (e.g., Llama-2 or Phi-3)
- Probe Families — Logistic regression, dimensionality, LDA, PCA, CCS, LAT, and ensemble probes
- Meta-Probe Evaluation (In Development) — Probe-on-probe reliability scoring for interpretability consistency
- Layer-Wise Generalization Metrics — Identify where features stabilize or transfer between datasets
- Configurable Presets — .toml-based experiment presets for fast, reproducible workflows

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

### Run
```bash
python experiments/run_pipeline.py standard alina_mvp --output-dir ./output/
```

### Artifacts (Example S3 Layout)




