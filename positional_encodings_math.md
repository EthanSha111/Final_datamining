## Positional Encodings and ViT Attention – Key Equations

This document collects the core equations that correspond to the code in
`vit_models.py` and the planned implementations in `positional_encodings.py`.
It is meant as a concise reference while you implement and experiment.

---

### 1. ViT Attention Block (in `vit_models.py`)

Given an input sequence $x \in \mathbb{R}^{B \times N \times D}$
(batch size $B$, sequence length $N$, model dimension $D$):

- **Linear projections**

$$
Q = x W_Q, \quad
K = x W_K, \quad
V = x W_V, \quad
W_Q, W_K, W_V \in \mathbb{R}^{D \times D}.
$$

- **Positional module call**  
Before splitting into heads, we apply a positional encoding module:

$$
(Q_{\text{pos}}, K_{\text{pos}}) = \text{pos\_module}(Q, K, \text{pos}),
$$

where `pos` encodes token positions (CLS + patch indices). The module may be
RoPE-only, Cayley-STRING, reflection-based, or sparse-S Cayley-STRING.

- **Multi-head attention (per head)**  
Let $D_h = D / h$ be the per-head dimension and reshape
$Q_{\text{pos}}, K_{\text{pos}}, V$ into $(B, h, N, D_h)$. For each head:

$$
\text{scores} = \frac{Q_{\text{pos}} K_{\text{pos}}^\top}{\sqrt{D_h}}, \quad
\alpha = \text{softmax}(\text{scores}, \text{dim} = -1),
$$

$$
\text{context} = \alpha V,
$$

then heads are concatenated and projected with $W_O \in \mathbb{R}^{D \times D}$
to get the final attention output.

- **Transformer encoder block**  
With LayerNorm and an MLP block $\text{MLP}(\cdot)$,

$$
x' = x + \text{MHA}(\text{LayerNorm}(x), \text{pos}),
$$

$$
y = x' + \text{MLP}(\text{LayerNorm}(x')).
$$

In the ViT, a learned CLS token is prepended, the encoder blocks are stacked,
and the final CLS representation is fed to a linear head for classification.

---

### 2. Vanilla RoPE (Rotary Positional Embedding)

For an even $D$, split each token vector into 2D blocks along the last
dimension. Let

$$
q = [q_1, q_2, \dots, q_{D/2}] \in \mathbb{R}^{D}, \quad
q_j = (q_{2j-1}, q_{2j}),
$$

and let `freqs` be a vector $\omega \in \mathbb{R}^{D/2}$. For position
$p \in \mathbb{Z}$, the angle for block $j$ is

$$
\theta_j = p \cdot \omega_j.
$$

Define the 2D rotation matrix for block $j$ as

$$
R(\theta_j) =
\begin{bmatrix}
 \cos \theta_j & -\sin \theta_j \\
 \sin \theta_j & \cos \theta_j
\end{bmatrix}.
$$

RoPE applies this rotation to each 2D block:

$$
q'_j = R(\theta_j) \, q_j,
$$

and concatenates all blocks back to obtain the rotated vector $q' \in \mathbb{R}^D$.
In code (`apply_rope`), this is implemented with elementwise `cos`/`sin` and
reshaping into $(B, N, D/2, 2)$.

The **`RoPEPositionalModule`** applies this transformation independently to
both $Q$ and $K$:

$$
Q_{\text{pos}} = \text{RoPE}(Q, \text{pos}, \omega), \quad
K_{\text{pos}} = \text{RoPE}(K, \text{pos}, \omega).
$$

---

### 3. Cayley-STRING with Dense Skew-Symmetric $S$

#### 3.1 Skew-symmetric matrix from parameters

Let $D = d_{\text{model}}$ and let `param_vec` be a vector of length
$D(D-1)/2$, one parameter per $(i, j)$ with $i < j$. We construct
the skew-symmetric matrix $S \in \mathbb{R}^{D \times D}$ by:

- For each parameter index $k$ with associated pair $(i, j)$, set

$$
S_{ij} = \theta_k, \quad S_{ji} = -\theta_k,
$$

and $S_{ii} = 0$. This is what `build_skew_symmetric` will implement.

By construction, $S^\top = -S$.

#### 3.2 Cayley transform

Given a skew-symmetric $S$, the Cayley transform defines an orthogonal
matrix

$$
U = (I - S)(I + S)^{-1}.
$$

To apply $U$ to a vector $h \in \mathbb{R}^D$ (or batch of vectors),
we note that

$$
y = U h = (I - S)(I + S)^{-1} h,
$$

is equivalently obtained by first solving a linear system

$$
(I + S) z = h, \quad \text{then} \quad y = (I - S) z.
$$

In `cayley_apply(S, h)`, we implement this via a dense linear solve:
- compute $z$ from $(I + S) z = h$,
- then return $y = (I - S) z$.

#### 3.3 Cayley-STRING positional module

The **`CayleyStringPE`** module combines RoPE and the Cayley transform:

1. Build $S$ from `S_params` using `build_skew_symmetric`.
2. Apply RoPE to queries and keys:

$$
Q_{\text{rope}} = \text{RoPE}(Q, \text{pos}, \omega), \quad
K_{\text{rope}} = \text{RoPE}(K, \text{pos}, \omega).
$$

3. Apply the Cayley transform:

$$
Q_{\text{out}} = U Q_{\text{rope}}, \quad
K_{\text{out}} = U K_{\text{rope}},
$$

where $U$ is defined as above and applied to the last dimension of each token.

These $Q_{\text{out}}, K_{\text{out}}$ are then passed into the attention.

---

### 4. Reflection-Based STRING (Householder in 2D)

Instead of 2D Givens rotations, the reflection-based variant uses products of
2D Householder reflections.

#### 4.1 2D Householder reflection

For an angle $\alpha$, define the vector

$$
v =
\begin{bmatrix}
 \cos \alpha \\
 \sin \alpha
\end{bmatrix}
\in \mathbb{R}^2.
$$

The Householder reflection matrix is

$$
H(\alpha) = I - 2 \frac{v v^\top}{\|v\|^2}.
$$

Since $\|v\|^2 = \cos^2 \alpha + \sin^2 \alpha = 1$, this simplifies to

$$
H(\alpha) = I - 2 v v^\top.
$$

The function `householder_2d(alpha)` returns these $2 \times 2$ matrices
batched over the shape of `alpha`.

#### 4.2 Reflection-based RoPE

For each 2D block $q_j$ of a token, we define two angles

$$
\alpha_{1,j} = p \cdot \omega^{(1)}_j, \quad
\alpha_{2,j} = p \cdot \omega^{(2)}_j,
$$

where $\omega^{(1)}, \omega^{(2)} \in \mathbb{R}^{D/2}$ are learned frequency
vectors.

We then construct

$$
H_1 = H(\alpha_{1,j}), \quad
H_2 = H(\alpha_{2,j}),
$$

and define the effective rotation as the product

$$
R_j = H_2 H_1,
$$

which is an orthogonal matrix in $\mathbb{R}^{2 \times 2}$. Applying this to
each 2D block gives

$$
q'_j = R_j q_j.
$$

In `apply_reflection_rope`, this is implemented blockwise, similarly to RoPE
but using $R_j = H_2 H_1$ instead of a simple rotation matrix. The
**`ReflectionStringPE`** module applies this to $Q$ and $K$, optionally
followed by a Cayley transform in the same spirit as `CayleyStringPE`.

---

### 5. Sparse-S Cayley-STRING

The sparse-S variant keeps only a fraction $f$ of the strictly upper-triangular
entries of $S$ learnable and non-zero.

#### 5.1 Sparse mask

Let $\mathcal{U} = \{(i, j) : 0 \le i < j < D\}$ be all upper-triangular
index pairs. The function `make_sparse_mask(d_model, f)` randomly samples
approximately

$$
K \approx f \cdot \frac{D(D-1)}{2}
$$

pairs from $\mathcal{U}$ and returns them as `chosen_indices`.

#### 5.2 Building sparse-structured $S$

Given parameters $\theta \in \mathbb{R}^K$ and `chosen_indices`
$\{(i_k, j_k)\}_{k=1}^K$, `build_sparse_skew_symmetric` forms

$$
S_{i_k j_k} = \theta_k, \quad
S_{j_k i_k} = -\theta_k,
$$

and all other entries are zero. This $S$ is still skew-symmetric but sparse
in structure (though implemented as a dense tensor initially).

#### 5.3 Sparse Cayley-STRING module

The **`SparseCayleyStringPE`** module mirrors `CayleyStringPE`:

1. Use `make_sparse_mask` once in the constructor to fix the support of $S$.
2. Maintain parameters only for those selected entries.
3. In `forward`, call `build_sparse_skew_symmetric` to construct $S$.
4. Apply RoPE to $Q$ and $K$, then apply the Cayley transform with this
   sparse-structured $S$.

The resulting $Q_{\text{out}}, K_{\text{out}}$ are plugged into the same
attention equations as in Section 1. By varying $f$, you can explore
trade-offs between parameter count, runtime (especially with sparse solvers),
and final accuracy.
