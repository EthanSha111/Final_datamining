## Relative Positional Encodings in ViT – Project Overview

This repository implements and compares several relative positional encoding (RPE)
mechanisms for small Vision Transformers (ViTs), following the Cayley-STRING
framework of Schenck et al. (2025). The main goal is to benchmark:

- **Baseline RoPE** (rotary positional embeddings)
- **Cayley-STRING with dense skew-symmetric S**
- **Reflection-based STRING** (rotations via 2D Householder reflections)
- **Sparse-S Cayley-STRING** (fraction \(f\) of upper-tri entries learnable)

on **MNIST** and **CIFAR‑10**, comparing **accuracy**, **training time**, and
**inference time**.

### Repository Structure

- **`positional_encodings.py`**  
  All RPE logic (RoPE, Cayley-STRING, reflection-based rotations, sparse-S):
  - `build_skew_symmetric`, `cayley_apply`, `apply_rope`
  - `householder_2d`, `apply_reflection_rope`
  - `make_sparse_mask`, `build_sparse_skew_symmetric`
  - `RoPEPositionalModule`, `CayleyStringPE`, `ReflectionStringPE`, `SparseCayleyStringPE`  
  (Currently contains signatures and detailed TODO comments for implementations.)

- **`vit_models.py`**  
  Vision Transformer components with pluggable positional modules:
  - `PatchEmbedding`: image → patch sequence
  - `MultiHeadSelfAttention`: calls `pos_module(q, k, pos)` before attention
  - `TransformerEncoderBlock`: MHA + MLP with residuals and LayerNorm
  - `VisionTransformer`: end-to-end ViT classifier using a chosen RPE module

- **`data_utils.py`**  
  Dataset and reproducibility utilities:
  - `set_seed(seed)`: set Python / NumPy / PyTorch seeds
  - `get_mnist_loaders(...)`: MNIST train/test `DataLoader`s
  - `get_cifar10_loaders(...)`: CIFAR‑10 train/test `DataLoader`s

- **`train_eval.py`**  
  Training, evaluation, timing, and experiment orchestration:
  - `train_one_epoch`, `eval_model`
  - `measure_epoch_time`, `measure_inference_time`
  - `ExperimentConfig`: config dataclass for a run
  - `run_experiment(config, device)`: full train + eval + timing for a variant

- **`mnist_experiments.ipynb`**  
  Notebook to run and compare RPE variants on MNIST:
  - RoPE baseline
  - Cayley-STRING (dense S)
  - Reflection-STRING
  - Sparse-S Cayley-STRING for several \(f\) values

- **`positional_encodings_math.md`**  
  Short derivation-style document explaining the key equations used in:
  - the ViT attention block in `vit_models.py`
  - the RPE mechanisms defined in `positional_encodings.py`.

### How to Use

- **Implement Part 1 (math/TODOs)**:  
  Fill in the implementations in `positional_encodings.py` following the TODO
  comments and the equations outlined in `positional_encodings_math.md`.

- **Run MNIST experiments**:  
  Open `mnist_experiments.ipynb` and run all cells to train and time:
  - RoPE, Cayley-STRING, Reflection-STRING, and Sparse-S variants.

- **Extend to CIFAR‑10** (optional):  
  - Use `get_cifar10_loaders` from `data_utils.py`.  
  - Reuse `ExperimentConfig` and `run_experiment` from `train_eval.py` in a
    separate notebook or script.

