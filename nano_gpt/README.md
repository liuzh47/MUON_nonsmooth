# Musec: Momentum Spectral Clipping for Stable Muon-Type Training

This repository provides the official implementation of **Soft Musec**, an optimizer that stabilizes Muon training by replacing spectral flattening with spectral clipping. Built on the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) codebase.

## Overview

Muon is a highly effective optimizer for LLM training but suffers from instability caused by its spectral flattening — all singular values of the momentum matrix are set to one, amplifying weakly represented directions and causing loss spikes and unbounded weight growth.

Soft Musec fixes this by **clipping** singular values that exceed a threshold *D* rather than flattening them all. This preserves the spectral structure of the momentum while constraining large spectral components. The clipping is approximated efficiently via coupled Newton–Schulz iterations, requiring no SVD.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

<details>
<summary><strong>Core dependencies</strong></summary>

```
numpy
tqdm
torch==2.10
huggingface-hub
kernels
setuptools
typing-extensions==4.15.0
datasets
tiktoken
```

</details>

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
# downloads only the first 900M training tokens to save time
python data/cached_fineweb10B.py 9
./run.sh
```

## Usage

### Quick start

```bash
# Train NanoGPT-Medium on FineWeb with Soft Musec (default settings)
export MODE=muon
export ORTH_METHOD=softclip
python train_gpt_medium.py
```

### Model configurations

| Config         | Layers | Heads | Head Dim | Model Dim | Params |
|----------------|--------|-------|----------|-----------|--------|
| NanoGPT-Small  | 11     | 6     | 128      | 768       | ~491M  |
| NanoGPT-Medium | 16     | 8     | 128      | 1,024     | ~613M  |
| NanoGPT-Wide   | 16     | 16    | 128      | 2,048     | ~1.63B |

### Training examples

```bash
# NanoGPT-Small 
export MODE=muon
export ORTH_METHOD=softclip
python train_gpt.py 

# NanoGPT-Medium 
export MODE=muon
export ORTH_METHOD=softclip
python train_gpt_medium.py 


# NanoGPT-Wide 
export MODE=muon
export ORTH_METHOD=softclip
python train_gpt_wide.py 
```

### Comparing with baselines

```bash
# Standard Muon
export MODE=muon
export ORTH_METHOD=vanilla_muon_no_triton
python train_gpt_medium.py

# MuonClip
export MODE=muon
export USE_MUONCLIP=true
export ORTH_METHOD=vanilla_muon_no_triton
python train_gpt_medium.py

# SPECTRA (SGDM + spectral clipping)
export MODE=muon
export ORTH_METHOD=spectra
python train_gpt_medium.py
```

## Repository structure

```
nano_gpt/
├── README.md
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
├── run.sh
├── train_gpt.py             # Training script for NanoGPT-Small
├── train_gpt_medium.py      # Training script for NanoGPT-Medium
├── train_gpt_wide.py        # Training script for NanoGPT-Wide
├── triton_kernels.py
├── plot_res/
│   ├── plot_bank_vs_steps.py      # Plot weight bank vs steps
│   ├── plot_erank.py              # Plot effective rank
│   ├── plot_final_val_loss_vs_lr.py  # Plot learning rate sweep result
│   └── ...
├── data/
    ├── cached_fineweb100B.py
    ├── cached_fineweb10B.py
    ├── cached_finewebedu10B.py
    ├── fineweb.py
    ├── requirements.txt
```

## Results

Soft Musec consistently improves training stability over existing Muon variants across learning rates and model sizes. It remains stable where Muon and MuonClip diverge, and matches their best-tuned performance.


## Acknowledgements

This codebase is built on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) by Keller Jordan et al.

## License

MIT License
