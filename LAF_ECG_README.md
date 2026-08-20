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


### Proposed components

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

The implementation described here targets the repository's **five-class, single-label PTB-XL classification setup**.

The associated manuscript also evaluates LAF-ECG on Chapman-Shaoxing. The PTB-XL installation and training path below is the runnable configuration documented by this bundle; add the exact public Chapman configuration alongside it when releasing that experiment.

---

## Installation

Run the installer from this bundle. It is idempotent and creates one-time `.before-moment-medtsllm` backups of modified files.

```bash
python install_moment_medtsllm.py --repo /path/to/medtsllm2
cd /path/to/medtsllm2
```

Create a clean Python 3.11 environment and install the merged requirements:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-full-moment.txt
```

The original repository pins older NumPy and Transformers versions that are incompatible with the current `momentfm` package. Use `requirements-full-moment.txt`, or install the original requirements first and `requirements-moment.txt` last.

---

## PTB-XL data

The dataset implementation expects the following structure:

```text
data/ptbxl/
├── ptbxl_database.csv
├── scp_statements.csv
├── records100/
└── ...
```

On first use, each split is cached as:

```text
cache_100hz_{split}.npz
```

Delete these caches only when changing the underlying PTB-XL records or the target-construction logic.

---

## Train LAF-ECG

```bash
python train.py configs/datasets/ptbxl_moment_medtsllm.toml
```

### Principal frozen-backbone configuration

- MOMENT encoder: **frozen**
- MOMENT patch embedder: **frozen**
- LLM: **frozen**
- MedTsLLM adaptation layers: **trainable**
- LTFM, MRTC, temporal-to-language projection, ARTF, pooling, BAS heads, and classifier: **trainable**
- Effective batch size: `4 × 4 = 16`
- Validation model selection: **PTB-XL fold 9 macro-F1**
- Final test split: **PTB-XL fold 10**

For a clean publication protocol, evaluate the held-out test split only after validation-based checkpoint selection.

---

## Reproducible training details

The paper-aligned setup uses:

```text
Optimizer                 AdamW
Learning rate             3e-4
Weight decay              0.01
Epochs                     20
Physical batch size       4
Gradient accumulation     4
Effective batch size      16
Gradient clipping         1.0
Label smoothing           0.05
Classifier dropout        0.20
```

Training-time ECG augmentation includes:

- per-lead amplitude scaling: ±5%
- Gaussian noise: `σ = 0.01`
- lead dropout: `p = 0.05`
- contiguous temporal masking: 2% of the signal

Validation and test data are not augmented.

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

LAF-ECG builds on the ideas and open-source implementations of **MedTsLLM** and **MOMENT**. Please cite the original works when using those components.

---

## Notes for the public release

Before publishing the repository alongside the paper, verify that:

- the public config uses the same final test-evaluation protocol described in the paper;
- the exact Chapman-Shaoxing loader/config used in the manuscript is included if those results are reported;
- the README hyperparameters match the final released config;
- pretrained model IDs and dependency versions are pinned for reproducibility.
