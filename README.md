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

### ViT Attention Block (Math Overview)

Given an input sequence \(x \in \mathbb{R}^{B \times N \times D}\) with batch size \(B\),
sequence length \(N\), and model dimension \(D\):

\[
Q = x W_Q,\quad K = x W_K,\quad V = x W_V,\quad
W_Q, W_K, W_V \in \mathbb{R}^{D \times D}.
\]

Before splitting into heads, we apply a positional encoding module:

\[
(Q_{\text{pos}}, K_{\text{pos}}) = \text{pos\_module}(Q, K, \text{pos}),
\]

where `pos` encodes token positions (CLS token + patch indices).  
The module may be RoPE-only, Cayley-STRING, reflection-based STRING,
or sparse-S Cayley-STRING.

For multi-head attention, let \(D_h = D / h\) be the per-head dimension and
reshape \((Q_{\text{pos}}, K_{\text{pos}}, V)\) into \((B, h, N, D_h)\). For each head:

\[
\text{scores} = \frac{Q_{\text{pos}} K_{\text{pos}}^\top}{\sqrt{D_h}},\quad
\alpha = \text{softmax}(\text{scores}, \text{dim} = -1),\quad
\text{context} = \alpha V.
\]

Heads are concatenated and projected with \(W_O \in \mathbb{R}^{D \times D}\) to get the
final attention output.

The transformer encoder block with LayerNorm and an MLP block \(\text{MLP}(\cdot)\) is:

\[
x' = x + \text{MHA}(\text{LayerNorm}(x), \text{pos}),
\]
\[
y = x' + \text{MLP}(\text{LayerNorm}(x')).
\]

In the ViT, a learned CLS token is prepended, the encoder blocks are stacked, and the
final CLS representation is fed to a linear layer for classification.

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

