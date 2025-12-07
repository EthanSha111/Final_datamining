"""
positional_encodings.py
------------------------

Part 1: Relative positional encoding (RPE) mechanisms.

NOTE: Per project instructions, this file currently contains ONLY function/class
signatures and high-level TODO comments, but NOT full implementations.
The implementations will be completed in a later phase.
"""



from __future__ import annotations
from typing import Sequence, Tuple
import torch
from torch import nn

##############################
# 1.1 Common utilities
##############################

def build_skew_symmetric(param_vec: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Build a dense skew-symmetric matrix S in R^{d x d} from a flat parameter vector.
    Maps parameters to the strictly upper triangle and negates them for the lower triangle.
    """
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    S = torch.zeros(d_model, d_model, device=param_vec.device, dtype=param_vec.dtype)
    rows, cols = torch.triu_indices(d_model, d_model, offset=1, device=param_vec.device)
    
    expected_size = (d_model * (d_model - 1)) // 2
    if param_vec.numel() != expected_size:
        raise ValueError(f"Expected param_vec size {expected_size}, got {param_vec.numel()}")

    S[rows, cols] = param_vec
    S[cols, rows] = -param_vec  # Ensure S^T = -S
    return S

def cayley_apply(S: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    r"""
    Apply the Cayley transform linear map U to vector(s) h:
        y = (I - S)(I + S)^{-1} h
    
    Implementation solves the linear system:
        1. Solve (I + S) z = h for z
        2. Compute y = (I - S) z
    """
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"S must be square, got shape {tuple(S.shape)}")
    d_model = S.shape[0]
    if h.shape[-1] != d_model:
        raise ValueError(f"Last dim of h ({h.shape[-1]}) must match S ({d_model})")
    I = torch.eye(d_model, device=S.device, dtype=S.dtype)
    
    # Prepare matrices
    M_plus = I + S   # (D, D)
    M_minus = I - S  # (D, D)
    
    # Reshape h to (Batch..., D, 1) for the solver
    h_reshaped = h.unsqueeze(-1)
    
    # --- MAC/MPS FIX STARTS HERE ---
    # torch.linalg.solve is not implemented for MPS; perform the solve on CPU and
    # move the result back. This keeps autograd intact but will be slower.
    if S.device.type == "mps":
        M_plus_cpu = M_plus.cpu()
        h_reshaped_cpu = h_reshaped.cpu()
        z_cpu = torch.linalg.solve(M_plus_cpu, h_reshaped_cpu)
        z = z_cpu.to(S.device)
    else:
        # Standard path for CUDA/CPU
        z = torch.linalg.solve(M_plus, h_reshaped)
    # --- MAC/MPS FIX ENDS HERE ---
    
    # 2. Compute y = (I - S) z
    y = M_minus @ z
    
    return y.squeeze(-1)

def apply_rope(q: torch.Tensor, pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    r"""
    Apply vanilla 1D RoPE to query/key vectors.
    """
    B, N, D = q.shape
    if D % 2 != 0:
        raise ValueError(f"RoPE requires even d_model; got {D}")
    q = q.contiguous()
    # Reshape q into pairs: (B, N, D/2, 2)
    q_pairs = q.reshape(B, N, D // 2, 2)
    
    # Prepare angles: theta = pos * freq
    # pos: (B, N) or (N,), freqs: (D/2,)
    if pos.ndim == 1:
        angles = pos.view(1, N, 1) * freqs.view(1, 1, -1) # (1, N, D/2)
    else:
        angles = pos.unsqueeze(-1) * freqs.view(1, 1, -1) # (B, N, D/2)
        
    cos = torch.cos(angles).unsqueeze(-1) # (..., 1)
    sin = torch.sin(angles).unsqueeze(-1)
    
    # Standard rotation: [x', y'] = [x cos - y sin, x sin + y cos]
    x = q_pairs[..., 0:1]
    y = q_pairs[..., 1:2]
    
    x_rot = x * cos - y * sin
    y_rot = x * sin + y * cos
    
    q_rot = torch.cat([x_rot, y_rot], dim=-1)
    return q_rot.view(B, N, D)

########################################
# 1.4 Variant 1: Reflection-based RoPE
########################################

def householder_2d(alpha: torch.Tensor) -> torch.Tensor:
    r"""
    Construct 2D Householder reflection matrices H = I - 2vv^T
    where v = [cos(alpha), sin(alpha)]
    """
    c = torch.cos(alpha)
    s = torch.sin(alpha)
    v = torch.stack([c, s], dim=-1) # (..., 2)
    
    # Outer product vv^T
    vvT = v.unsqueeze(-1) @ v.unsqueeze(-2) # (..., 2, 2)
    
    I = torch.eye(2, device=alpha.device, dtype=alpha.dtype)
    H = I - 2 * vvT
    return H

def apply_reflection_rope(
    q: torch.Tensor, pos: torch.Tensor, freq1: torch.Tensor, freq2: torch.Tensor
) -> torch.Tensor:
    r"""
    Apply reflection-based RoPE: R = H2 @ H1
    """
    B, N, D = q.shape
    if D % 2 != 0:
        raise ValueError(f"Reflection RoPE requires even d_model; got {D}")
    q = q.contiguous()
    
    # Reshape position for broadcasting
    if pos.ndim == 1:
        p = pos.view(1, N, 1)
    else:
        p = pos.unsqueeze(-1)
        
    # Calculate angles
    alpha1 = p * freq1.view(1, 1, -1)
    alpha2 = p * freq2.view(1, 1, -1)
    
    # Construct Householder matrices (B, N, D/2, 2, 2)
    H1 = householder_2d(alpha1)
    H2 = householder_2d(alpha2)
    
    # Effective rotation R = H2 @ H1
    R = H2 @ H1
    
    # Apply to q
    q_pairs = q.reshape(B, N, D // 2, 2, 1)
    q_out = R @ q_pairs
    
    return q_out.view(B, N, D)

#########################################
# 1.5 Variant 2: Sparse-S Cayley-STRING
#########################################

def make_sparse_mask(d_model: int, f: float) -> Sequence[Tuple[int, int]]:
    """
    Randomly sample a fraction f of upper-triangular indices.
    """
    rows, cols = torch.triu_indices(d_model, d_model, offset=1)
    num_total = rows.shape[0]
    num_keep = int(f * num_total)
    
    perm = torch.randperm(num_total)
    keep_indices = perm[:num_keep]
    
    selected_rows = rows[keep_indices].tolist()
    selected_cols = cols[keep_indices].tolist()
    
    return list(zip(selected_rows, selected_cols))

def build_sparse_skew_symmetric(
    param_vec: torch.Tensor,
    d_model: int,
    chosen_indices: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """
    Build sparse-structured S as a sparse COO tensor with zeros in unselected positions.
    """
    if not chosen_indices:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), device=param_vec.device, dtype=torch.long),
            torch.empty((0,), device=param_vec.device, dtype=param_vec.dtype),
            (d_model, d_model),
            device=param_vec.device,
            dtype=param_vec.dtype,
        )
        
    indices = torch.tensor(chosen_indices, device=param_vec.device, dtype=torch.long).T  # (2, K)
    rows, cols = indices[0], indices[1]

    # Build both upper (rows, cols) and lower (cols, rows) with negated values for skew-symmetry.
    all_indices = torch.cat([torch.stack([rows, cols]), torch.stack([cols, rows])], dim=1)  # (2, 2K)
    all_values = torch.cat([param_vec, -param_vec])

    S = torch.sparse_coo_tensor(
        all_indices,
        all_values,
        (d_model, d_model),
        device=param_vec.device,
        dtype=param_vec.dtype,
    ).coalesce()
    
    return S


#########################################
# Sparse Cayley solver (iterative, CG)
#########################################

def sparse_cayley_apply(
    S: torch.Tensor,
    h: torch.Tensor,
    max_iter: int = 20,
    tol: float = 1e-5,
) -> torch.Tensor:
    r"""
    Apply y = (I - S)(I + S)^{-1} h using a conjugate-gradient solve on the
    normal equations (I + S)^T (I + S) z = (I + S)^T h, then y = (I - S) z.

    S is assumed sparse (COO). h has shape (..., D).
    """
    if S.layout != torch.sparse_coo:
        raise ValueError("sparse_cayley_apply expects a sparse COO tensor for S.")

    d_model = S.shape[0]
    if S.shape[0] != S.shape[1]:
        raise ValueError(f"S must be square, got {tuple(S.shape)}")
    if h.shape[-1] != d_model:
        raise ValueError(f"Last dim of h ({h.shape[-1]}) must match S ({d_model})")

    # Flatten batch dims for simplicity: (B*, D)
    h_flat = h.reshape(-1, d_model)
    device = S.device
    dtype = S.dtype

    S_T = S.transpose(0, 1).coalesce()

    def sparse_mv(mat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # mat: sparse (D, D), x: (D,)
        return torch.sparse.mm(mat, x.unsqueeze(-1)).squeeze(-1)

    def apply_A(x: torch.Tensor) -> torch.Tensor:
        # A = (I + S)^T (I + S)
        tmp = x + sparse_mv(S, x)         # (I + S) x
        out = tmp + sparse_mv(S_T, tmp)   # (I + S)^T tmp
        return out

    def apply_b(h_vec: torch.Tensor) -> torch.Tensor:
        # b = (I + S)^T h
        return h_vec + sparse_mv(S_T, h_vec)

    outputs = []
    for h_vec in h_flat:
        b = apply_b(h_vec)
        z = torch.zeros_like(h_vec, device=device, dtype=dtype)
        r = b.clone()
        p = r.clone()
        rsold = torch.dot(r, r)

        for _ in range(max_iter):
            Ap = apply_A(p)
            denom = torch.dot(p, Ap) + 1e-12
            alpha = rsold / denom
            z = z + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)
            if torch.sqrt(rsnew) < tol:
                break
            p = r + (rsnew / (rsold + 1e-12)) * p
            rsold = rsnew

        # y = (I - S) z
        y = z - sparse_mv(S, z)
        outputs.append(y)

    out = torch.stack(outputs, dim=0).reshape_as(h)
    return out

#########################################
# Modules
#########################################

class RoPEPositionalModule(nn.Module):
    def __init__(self, d_model: int, learnable_freqs: bool = True) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        if learnable_freqs:
            self.freqs = nn.Parameter(inv_freq)
        else:
            self.register_buffer("freqs", inv_freq)

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return apply_rope(q, pos, self.freqs), apply_rope(k, pos, self.freqs)

class CayleyStringPE(nn.Module):
    def __init__(self, d_model: int, learnable_freqs: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        # RoPE params
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.freqs = nn.Parameter(inv_freq) if learnable_freqs else inv_freq
        if not learnable_freqs: self.register_buffer("freqs", inv_freq)
        
        # S params
        num_params = (d_model * (d_model - 1)) // 2
        self.s_params = nn.Parameter(torch.zeros(num_params)) # Init at 0

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        S = build_skew_symmetric(self.s_params, self.d_model)
        q_rope = apply_rope(q, pos, self.freqs)
        k_rope = apply_rope(k, pos, self.freqs)
        return cayley_apply(S, q_rope), cayley_apply(S, k_rope)

class ReflectionStringPE(nn.Module):
    def __init__(self, d_model: int, use_cayley: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_cayley = use_cayley
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.freq1 = nn.Parameter(inv_freq.clone())
        self.freq2 = nn.Parameter(inv_freq.clone() * 0.95)
        
        if use_cayley:
            num_params = (d_model * (d_model - 1)) // 2
            self.s_params = nn.Parameter(torch.zeros(num_params))
        else:
            self.s_params = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_ref = apply_reflection_rope(q, pos, self.freq1, self.freq2)
        k_ref = apply_reflection_rope(k, pos, self.freq1, self.freq2)
        
        if self.use_cayley and self.s_params is not None:
            S = build_skew_symmetric(self.s_params, self.d_model)
            return cayley_apply(S, q_ref), cayley_apply(S, k_ref)
        return q_ref, k_ref

class SparseCayleyStringPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        f: float,
        learnable_freqs: bool = True,
        use_sparse_solver: bool = True,
        cg_max_iter: int = 20,
        cg_tol: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_sparse_solver = use_sparse_solver
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.freqs = nn.Parameter(inv_freq) if learnable_freqs else inv_freq
        if not learnable_freqs: self.register_buffer("freqs", inv_freq)
        
        self.chosen_indices = make_sparse_mask(d_model, f)
        self.s_params = nn.Parameter(torch.zeros(len(self.chosen_indices)))

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        S_sparse = build_sparse_skew_symmetric(self.s_params, self.d_model, self.chosen_indices)
        q_rope = apply_rope(q, pos, self.freqs)
        k_rope = apply_rope(k, pos, self.freqs)

        if self.use_sparse_solver:
            q_out = sparse_cayley_apply(S_sparse, q_rope, max_iter=self.cg_max_iter, tol=self.cg_tol)
            k_out = sparse_cayley_apply(S_sparse, k_rope, max_iter=self.cg_max_iter, tol=self.cg_tol)
        else:
            # Fallback: convert to dense and reuse dense solver
            S_dense = S_sparse.to_dense()
            q_out = cayley_apply(S_dense, q_rope)
            k_out = cayley_apply(S_dense, k_rope)

        return q_out, k_out