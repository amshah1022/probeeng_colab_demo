# ProbeEng: Meta-Probing Infrastructure for Reliable Interpretability (DEMO) 

**Role:** Research Assistant — Cornell Long-Term AI Safety Lab

**Affiliation:** Long-term AI Safety Lab, Cornell University  

**Keywords:** meta-probing, interpretability infrastructure, AI reliability, evaluation science

---

## Purpose 

ProbeEng is a research framework for meta-probing, building, evaluating, and orchestrating probe networks that measure what large language models know and how reliably they express it.
Instead of interpreting a single model directly, ProbeEng focuses on the infrastructure layer of interpretability:
- Designing standardized probe pipelines to train, test, and ensemble thousands of probes across layers, models, and datasets
- Developing a meta-probe system that learns to evaluate and calibrate other probes’ reliability

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

## Quick Start (Read-Only Demonstration) 

This repository is a demonstration of the ProbeEng pipeline, intended for inspection rather than execution.
Due to its reliance on private datasets and configurations from the Cornell Long-Term AI Safety Lab, the full pipeline cannot be reproduced externally.
If you wish to understand the workflow, you can review the Colab notebook and accompanying code to see the pipeline stages and structure.

### 1. Installation 

```bash
git clone https://github.com/SamuelS0/ProbeEng
cd ProbeEng
ip install -r requirements.txt
```

### 2. Pipeline Structure Overview
These commands illustrate the internal experiment flow used in the Cornell LAISR environment.
They are provided for educational transparency, not for direct execution in this colab. 

```bash
# List available presets
python experiments/run_pipeline.py standard --list

# Minimal end-to-end run
python experiments/run_pipeline.py standard debug

# Full standard preset (example)
python experiments/run_pipeline.py standard standard
```

### 3. Stage-by-Stage Breakdown (Conceptual Example)

```bash
# Activation dataset creation
python experiments/run_dataset_creation.py demo

# Probe training
python experiments/run_probe_training.py demo \
  --activation-dataset path/to/activations.pickle

# Probe evaluation
python experiments/run_probe_evaluation.py demo \
  --probes path/to/probes/
```


## Illustrative Results

**Sample Probe Evaluation Results (Llama-2-7b-chat-hf, Logistic Regression)**

| accuracy | n   | llm_id              | train_dataset | eval_dataset              | probe_method | layer | token_idx | split       | is_supervised | is_grouped | same_dataset | threshold | recovered_accuracy |
|:---------:|:----|:--------------------|:---------------|:---------------------------|:--------------|:------:|:-----------:|:-------------|:---------------|:-------------|:--------------|:-----------:|:--------------------:|
| 0.500000 | 100 | Llama-2-7b-chat-hf | got_cities | got_cities | lr | 1 | 0 | validation | True | False | True | 1.00 | 0.500000 |
| 0.523077 | 65  | Llama-2-7b-chat-hf | got_cities | got_sp_en_trans | lr | 1 | 0 | validation | True | False | False | 1.00 | 0.523077 |
| 0.520000 | 100 | Llama-2-7b-chat-hf | got_cities | got_cities_cities_conj | lr | 1 | 0 | validation | True | False | False | 0.97 | 0.536082 |
| 0.530000 | 100 | Llama-2-7b-chat-hf | got_cities | got_cities_cities_disj | lr | 1 | 0 | validation | True | False | False | 0.94 | 0.563830 |
| 0.550000 | 100 | Llama-2-7b-chat-hf | got_cities | got_larger_than | lr | 1 | 0 | validation | True | False | False | 1.00 | 0.550000 |
| 0.520000 | 100 | Llama-2-7b-chat-hf | got_cities | got_cities | lr | 3 | 0 | validation | True | False | True | 1.00 | 0.520000 |
| 0.523077 | 65  | Llama-2-7b-chat-hf | got_cities | got_sp_en_trans | lr | 3 | 0 | validation | True | False | False | 1.00 | 0.523077 |
| 0.530000 | 100 | Llama-2-7b-chat-hf | got_cities | got_cities_cities_conj | lr | 3 | 0 | validation | True | False | False | 0.97 | 0.546392 |
| 0.580000 | 100 | Llama-2-7b-chat-hf | got_cities | got_cities_cities_disj | lr | 3 | 0 | validation | True | False | False | 0.94 | 0.617021 |
| 0.530000 | 100 | Llama-2-7b-chat-hf | got_cities | got_larger_than | lr | 3 | 0 | validation | True | False | False | 1.00 | 0.530000 |

**Interpretation:**  
This table shows a subset of results from the ProbeEng demo pipeline, measuring how well logistic regression probes trained on one dataset (*got_cities*) generalize to others.  
Accuracy improves modestly between layers 1 and 3, with *got_larger_than* and *got_cities_disj* showing the strongest early transfer (≈0.55–0.62).  
Later layers (not shown here) typically achieve >0.8 accuracy and exhibit the most stable cross-dataset generalization.

---




## Architecture 
```bash
┌────────────────┐   ┌────────────────┐   ┌────────────────────┐
│ Dataset Loader │ → │ Probe Trainer  │ → │ Evaluation + Plots │
│  (HF datasets) │   │  (sklearn/Py)  │   │  (metrics, graphs) │
└────────────────┘   └────────────────┘   └────────────────────┘
```
Each probe acts as a diagnostic classifier over hidden states.
The meta-probe module (in progress) evaluates these diagnostic probes across layers, datasets, and runs to estimate interpretability reliability.

## Note on Scope and Reproducibilty 

This public demo reproduces the core architecture and experimental flow of the full ProbeEng framework,
but omits restricted datasets, configurations, and analysis modules from the
Long-Term AI Safety Lab at Cornell.
All results, scripts, and figures are representative,
intended to demonstrate methodology and structure, not replicate the full internal system. 
## Acknowledgements & Role 

This repository reflects collaborative work conducted within the Long-Term AI Safety Lab at Cornell.
ProbeEng is maintained by the lab’s Probe Engineering research team (LAISR).

My role (Alina Shah):
  - Assisted with cross-layer probe generalization experiments on Llama-2-7B
  - Supported MVP preset design and documentation for pipeline reproducibility
  - Helped analyze and visualize probe performance (layer accuracy, transfer gaps, plots)
  - Contributed to testing and feedback loops for future meta-probe integration
## Contact
Alina Miret Shah
Research Assistant — Long-Term AI Safety Lab, Cornell University

alina.shah1022@gmail.com  

[alinamshah](https://www.linkedin.com/in/alinamshah/)





