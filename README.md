# LAF-ECG

**Multi-Resolution Lead-Aware Fusion of Time-Series and Language Foundation Models for ECG Classification**

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="Task" src="https://img.shields.io/badge/Task-ECG%20Classification-0F766E">
  <img alt="Backbones" src="https://img.shields.io/badge/Backbones-MOMENT%20%2B%20Llama--2--7B-6B7280">
  <img alt="Model" src="https://img.shields.io/badge/Model-LAF--ECG-7C3AED">
</p>

LAF-ECG is an ECG classification framework that augments the **MedTsLLM** reprogramming pathway with representations from the pretrained **MOMENT** time-series foundation model. In the codebase, the proposed model is registered internally as `moment_medtsllm`.

> **What is new?** MOMENT itself is adopted from prior work and is not claimed as a contribution. LAF-ECG introduces the ECG-specific interface that preserves lead-wise temporal information, incorporates original-resolution temporal context, and adaptively fuses temporal foundation representations with MedTsLLM tokens.

---

## Proposed components

- **Lead-Aware Temporal Foundation Module (LTFM)**  
  Preserves unreduced lead-wise MOMENT patch tokens and learns token-dependent inter-lead relevance instead of averaging leads with fixed weights.

- **Multi-Resolution Temporal Context (MRTC)**  
  Encodes two overlapping 512-sample windows from the original 1,000-sample ECG and pools them into a complementary global temporal context representation.

- **Adaptive Residual Token Fusion (ARTF)**  
  Aligns temporal tokens with MedTsLLM tokens, projects them into the LLM hidden space, and selectively injects them using feature-wise residual gating.

- **Branch-Specific Auxiliary Supervision (BAS)**  
  Applies auxiliary classification heads to the pre-fusion MedTsLLM and projected temporal representations to keep both pathways discriminative.

- **Supervised Cross-Branch Alignment (SCA)**  
  Encourages class-consistent representations across the two pre-fusion branches in their shared LLM-dimensional latent space.

---

## Pipeline

1. The MedTsLLM pathway applies **RevIN**, patch embedding, and vocabulary-based reprogramming.
2. Frozen `AutonLab/MOMENT-1-base` produces unreduced lead-wise temporal tokens.
3. **LTFM** learns token-dependent lead attention and aggregates the lead dimension.
4. **MRTC** extracts original-resolution context from two overlapping 512-sample ECG windows.
5. A gated context module injects the MRTC representation into aligned LTFM tokens.
6. **ARTF** projects and adaptively fuses LTFM features with MedTsLLM-reprogrammed tokens.
7. Structured text prompts and fused ECG tokens are processed by the frozen LLM.
8. Attention pooling and a compact classification head produce a recording-level prediction.
9. **BAS** and **SCA** provide branch-level supervision and cross-branch representation alignment during training.

---

## Repository scope

The current public configuration documented below targets the repository's **five-class, single-label PTB-XL classification setup**.

The associated manuscript also evaluates LAF-ECG on Chapman-Shaoxing. A complete public Chapman-Shaoxing reproduction configuration should be added to the repository alongside the PTB-XL setup when that experiment is released.

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/kkianamm/LAF-ECG.git
cd LAF-ECG
```

### 2. Python environment

Python **3.11** is recommended by the checked-in environment specification.

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-full.txt
```

Optional GPU/performance dependencies are listed in `recommended.txt`:

```bash
python -m pip install -r recommended.txt
```

> `recommended.txt` includes packages such as `flash-attn`, `causal-conv1d`, and `mamba-ssm`. These are optional and may require a compatible CUDA/PyTorch/compiler environment. If installation fails, the core environment in `requirements-full.txt` should be installed first.

### 3. Hugging Face model access

The default configuration uses:

- `AutonLab/MOMENT-1-base`
- `meta-llama/Llama-2-7b-hf`

Both models are downloaded from Hugging Face on first initialization. Make sure the machine can access Hugging Face and that your account has permission to use the configured Llama-2 checkpoint.

---

## Datasets

### PTB-XL v1.0.3

Download the official **PTB-XL v1.0.3** release from PhysioNet:

**https://physionet.org/content/ptb-xl/1.0.3/**

Create the dataset directory:

```bash
mkdir -p data/ptbxl
```

Place/extract the PTB-XL files so that the repository contains:

```text
data/ptbxl/
├── ptbxl_database.csv
├── scp_statements.csv
├── records100/
├── records500/
└── ...
```

The paper-aligned implementation uses the **100 Hz** recordings (`filename_lr`), corresponding to 1,000 samples per lead for each 10-second ECG.

The PTB-XL loader uses the official patient-level folds:

- folds **1-8**: training
- fold **9**: validation
- fold **10**: test

For the five-class single-label setup, diagnostic SCP-code likelihoods are accumulated within the diagnostic superclasses **NORM, MI, STTC, CD, and HYP**, and the superclass with the largest accumulated score is used as the recording-level label.

On first use, each split is cached as:

```text
data/ptbxl/cache_100hz_{split}.npz
```

Delete these caches only when changing the underlying PTB-XL records or the target-construction logic.

---

## Run LAF-ECG

The paper-aligned PTB-XL configuration is:

```text
configs/datasets/ptbxl_moment_medtsllm.toml
```

Run training with:

```bash
python3 train.py configs/datasets/ptbxl_moment_medtsllm.toml
```

> **Publication-style test protocol:** the checked-in configuration should use validation macro-F1 for checkpoint selection and evaluate fold 10 only after selecting/restoring the best validation checkpoint. Before final paper runs, set `evaluate_test_each_epoch = false` in the training section if test-per-epoch evaluation is enabled.

---

## Principal frozen-backbone configuration

- MOMENT encoder: **frozen**
- MOMENT patch embedder: **frozen**
- Llama-2-7B: **frozen**
- MedTsLLM adaptation layers: **trainable**
- LTFM, MRTC, temporal-to-language projection, ARTF, pooling, BAS heads, and classifier: **trainable**
- SCA: **training alignment objective**
- Effective batch size: `4 × 4 = 16`
- Validation model selection: **PTB-XL fold 9 macro-F1**
- Final held-out test split: **PTB-XL fold 10**

---

## Reproducible training details

The paper-aligned PTB-XL setup uses:

```text
Optimizer                 AdamW
Learning rate             3e-4
Weight decay              0.01
Epochs                     20
Physical batch size       4
Gradient accumulation     4
Effective batch size      16
LR scheduler              constant
Gradient clipping         1.0
Class weighting           enabled
Label smoothing           0.05
Classifier dropout        0.20
Auxiliary-loss weight     0.20
Alignment-loss weight     0.05
Alignment temperature     0.10
```

Training-time ECG augmentation includes:

- per-lead amplitude scaling: ±5%
- Gaussian noise: `σ = 0.01`
- lead dropout: `p = 0.05`
- contiguous temporal masking: 2% of the signal

Validation and test data are not augmented.

---

## Notes on input construction

- The original PTB-XL ECG contains **1,000 samples per lead at 100 Hz**.
- The aligned MedTsLLM/LTFM pathway receives a full-record linear resampling to **512 samples**.
- MedTsLLM uses **8-sample patches with stride 8**.
- MRTC retains the standardized 1,000-sample recording and uses two overlapping 512-sample windows.
- The default MRTC windows correspond to samples `0:512` and `488:1000`.

---

## Citation

If you use this repository, please cite the LAF-ECG paper and the underlying MOMENT and MedTsLLM works.

```bibtex
@inproceedings{taktehrani2026lafecg,
  title     = {LAF-ECG: Multi-Resolution Lead-Aware Fusion of Time-Series and Language Foundation Models for ECG Classification},
  author    = {Taktehrani, Kiana and Hosseini, Maryam and Mohammadi, Arash},
  booktitle = {The 48th Conference of the Canadian Medical and Biological Engineering Society / APIBQ 2026},
  year      = {2026}
}
```

---

## Acknowledgements

LAF-ECG builds on the MedTsLLM framework and uses pretrained MOMENT and Llama-2 backbones. Please cite the corresponding original works when using this repository.
