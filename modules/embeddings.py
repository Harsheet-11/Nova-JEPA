import sys
import os
import math
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from config import VOCAB_SIZE, D_MODEL, D_HEAD

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

logger = logging.getLogger(__name__)

    # Token ID -> Meaningful Vectors
class TokenEmbedding(nn.Module):


    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL) -> None:
        # # Input: vocab_size=32000, d_model=256.
        # # Creates embedding matrix [32000, 256].
        # pass
        
        super().__init__()
        
        self._validate_init_args(vocab_size, d_model)
        
        self.vocab_size = vocab_size
        self.d_model = d_model   
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0,
        )
        
        self._init_weights()

    
# -------------- VALIDATIONS ------------------ [ignore to understand the process]  
                 
    #   Reject obviously wrong constructor arguments before any memory is allocated.
    def _validate_init_args(self, vocab_size: int, d_model: int) -> None:
     
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be a positive int, got {vocab_size!r}"
            )
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(
                f"d_model must be a positive int, got {d_model!r}"
            )
        if d_model % config.N_HEADS != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by "
                f"N_HEADS={config.N_HEADS} so attention heads are equal width. "
                f"d_head would be {d_model / config.N_HEADS} (not an integer)."
            )  
            
            
    def _validate_input_dtype(self, input_ids: Tensor) -> None:

        if input_ids.dtype not in (torch.long, torch.int32):
            raise TypeError(
                f"input_ids must be torch.long (int64) or torch.int32, "
                f"got {input_ids.dtype}. "
                "Common cause: passing a float tensor from a data pipeline "
                "that forgot .long() after loading from a numpy array."
            )
            
            
    def _validate_input_shape(self, input_ids: Tensor) -> None:
        
        if input_ids.dim() not in (1, 2):
            raise ValueError(
                f"input_ids must be 1-D [seq_len] or 2-D [B, seq_len], "
                f"got shape {list(input_ids.shape)} ({input_ids.dim()}-D). "
                "If you are passing an embedding tensor, the embedding "
                "step has already been done."
            )
            
            
#   Every ID must be a valid row index into the embedding table.
    def _validate_input_range(self, input_ids: Tensor) -> None:
      
        if input_ids.numel() == 0:
            return  # empty tensor is fine — nothing to look up

        lo = input_ids.min().item()
        hi = input_ids.max().item()

        if lo < 0:
            raise ValueError(
                f"input_ids contains negative value {lo}. "
                "All token IDs must be in [0, vocab_size-1]. "
                "Check tokenizer for negative ID emission."
            )
        if hi >= self.vocab_size:
            raise ValueError(
                f"input_ids max value {hi} ≥ vocab_size={self.vocab_size}. "
                "Likely cause: config.VOCAB_SIZE does not match the "
                "vocab_size the tokenizer was built with."
            )

            
    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        with torch.no_grad():
            self.embedding.weight[0].fill_(0.0)  


    def forward(self, input_ids):
        # Input : [B, 256] integer token IDs.
        # Output: [B, 256, 256] float embeddings.
        
        self._validate_input_shape(input_ids)
        self._validate_input_dtype(input_ids)
        self._validate_input_range(input_ids)

        if input_ids.dtype == torch.int32:
            input_ids = input_ids.long()

        return self.embedding(input_ids)

# RoPE: injects position information into attention Q/K vectors.


#    STAGE BREAKDOWN:
#         Stage 1  compute_inv_freq    θᵢ = 10000^(-2i/d)  one per dim-pair
#         Stage 2  build_angles        angle[p,i] = p × θᵢ  shape [max_seq, d/2]
#         Stage 3  build_cos_cache     cos(angle)            shape [max_seq, d/2]
#         Stage 4  build_sin_cache     sin(angle)            shape [max_seq, d/2]
#         Stage 5  get_cos_sin         slice caches for current positions
#         Stage 6  split_pairs         reshape x[...,d] → x[...,d/2,2]
#         Stage 7  rotate_half         (-x_odd, x_even) → interleaved shape [...,d]
#         Stage 8  apply_rotation      x*cos + rotate_half(x)*sin
#         Stage 9  apply_to_query      full pipeline for Q tensor
#         Stage 10 apply_to_key        full pipeline for K tenso
class RotaryEmbedding:
    def __init__(self, dim: int = D_HEAD, max_seq_len: int = 2048, base: int = 10000) -> None:
        
        self._validate_init_args(dim, max_seq_len, base)

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Stages 1 → 4: build caches at construction time
        self.inv_freq = self._compute_inv_freq()    # Stage 1  [dim/2]
        self.angles = self._build_angles()           # Stage 2  [max_seq, dim/2]
        self.cos_cache = self._build_cos_cache()     # Stage 3  [max_seq, dim/2]
        self.sin_cache = self._build_sin_cache()    # Stage 4  [max_seq, dim/2]
        




# -------------- VALIDATIONS ------------------ [ignore to understand the process]   
  
    def _validate_init_args(self, dim: int, max_seq_len: int, base: int) -> None:

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                f"dim must be a positive int, got {dim!r}. "
                "Set dim = config.D_HEAD = config.D_MODEL // config.N_HEADS."
            )
        if dim % 2 != 0:
            raise ValueError(
                f"dim={dim} must be even. "
                "RoPE rotates dimension PAIRS (d₀,d₁), (d₂,d₃), ... "
                "An odd dim leaves one dimension without a pair."
            )
        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be a positive int, got {max_seq_len!r}."
            )
        if not isinstance(base, int) or base <= 0:
            raise ValueError(
                f"base must be a positive int, got {base!r}. "
                "Standard value is 10000."
            )

    def _validate_positions(self, positions: Tensor) -> None:
        
        if positions.dtype not in (torch.long, torch.int32, torch.int):
            raise TypeError(
                f"positions must be an integer tensor for cache indexing, "
                f"got {positions.dtype}. Use torch.arange(seq_len).long()."
            )
        if positions.numel() == 0:
            logger.warning(
                "get_cos_sin received an empty positions tensor. "
                "Returning empty cos/sin — check upstream sequence length."
            )
            return
        lo = positions.min().item()
        hi = positions.max().item()
        if lo < 0:
            raise ValueError(
                f"positions contains negative value {lo}. "
                "All positions must be ≥ 0."
            )
        if hi >= self.max_seq_len:
            raise ValueError(
                f"positions max value {hi} ≥ max_seq_len={self.max_seq_len}. "
                "Either increase max_seq_len at RotaryEmbedding init "
                "or truncate the sequence before calling apply_to_query / apply_to_key."
            )
            
            
    def _validate_rotation_inputs(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> None:

        if x.dim() != 4:
            raise ValueError(
                f"x must be 4-D [B, N_HEADS, seq_len, dim], "
                f"got shape {list(x.shape)}. "
                "Reshape Q/K before calling apply_rotation."
            )
        if cos.dim() != 2 or sin.dim() != 2:
            raise ValueError(
                f"cos and sin must be 2-D [seq_len, dim/2], "
                f"got cos {list(cos.shape)}, sin {list(sin.shape)}. "
                "Use get_cos_sin() to produce them."
            )

        seq_x = x.shape[2]
        seq_c = cos.shape[0]
        if seq_x != seq_c:
            raise ValueError(
                f"Sequence length mismatch: x has seq_len={seq_x} "
                f"but cos/sin have seq_len={seq_c}. "
                "Pass matching positions to get_cos_sin."
            )

        dim_x = x.shape[3]
        dim_c_half = cos.shape[1]
        if dim_x != self.dim:
            raise ValueError(
                f"x last dim={dim_x} does not match RotaryEmbedding dim={self.dim}. "
                "Check D_HEAD = D_MODEL // N_HEADS."
            )
        if dim_c_half != self.dim // 2:
            raise ValueError(
                f"cos last dim={dim_c_half} should be dim/2={self.dim // 2}. "
                "Do not expand cos/sin before passing to apply_rotation — "
                "it does the expansion internally."
            )



    def _validate_qk_shape(self, x: Tensor, name: str) -> None:

        if x.dim() != 4:
            raise ValueError(
                f"{name} must be 4-D [B, N_HEADS, seq_len, D_HEAD], "
                f"got shape {list(x.shape)}. "
                "Split the projected tensor into heads before calling "
                "apply_to_query / apply_to_key."
            )
        d = x.shape[-1]
        if d != self.dim:
            raise ValueError(
                f"{name} last dim={d} does not match "
                f"RotaryEmbedding dim={self.dim}. "
                f"Expected D_HEAD = D_MODEL // N_HEADS = "
                f"{config.D_MODEL} // {config.N_HEADS} = {config.D_HEAD}."
            )

    # --------------------------------------------------
    # STAGE 1
    # θ_i = 10000^(-2i/d)
    # --------------------------------------------------

    def _compute_inv_freq(self) -> Tensor:
        
        exponents = torch.arange(0, self.dim, 2).float() / self.dim
        
        return 1.0 / (self.base ** exponents)  # [dim/2]


    # --------------------------------------------------
    # STAGE 2
    # angle = position * frequency
    # --------------------------------------------------

    def _build_angles(self) -> Tensor:

        positions = torch.arange(self.max_seq_len).float()  
        
        return positions[:, None] * self.inv_freq[None, :]

    # --------------------------------------------------
    # STAGE 3
    # cos(angle)
    # --------------------------------------------------

    def _build_cos_cache(self) -> Tensor:

         return torch.cos(self.angles) 
    # --------------------------------------------------
    # STAGE 4
    # sin(angle)
    # --------------------------------------------------

    def _build_sin_cache(self):
        
        return torch.sin(self.angles)

#        Move all internal tensors to `device` for CPU calcutation as my GPU not powerful enough 
    def to(self, device: torch.device) -> "RotaryEmbedding":

        self.inv_freq = self.inv_freq.to(device)
        self.angles = self.angles.to(device)
        self.cos_cache = self.cos_cache.to(device)
        self.sin_cache = self.sin_cache.to(device)
        return self


    # --------------------------------------------------
    # STAGE 5
    # retrieve values for current positions
    # --------------------------------------------------

    def get_cos_sin(self, positions: Tensor):
        
        self._validate_positions(positions)

        cos = self.cos_cache[positions]  # [seq_len, dim/2]
        sin = self.sin_cache[positions]  # [seq_len, dim/2]

        return cos, sin

    # --------------------------------------------------
    # STAGE 6
    # split into dimension pairs
    # [2,1,4,3]
    # ->
    # [[2,1],
    #  [4,3]]
    # --------------------------------------------------

    def split_pairs(self, x: Tensor):
        
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"Last dimension of x must be even for pairing, "
                f"got shape {list(x.shape)}."
            )
        return x.reshape(*x.shape[:-1], -1, 2)

    # --------------------------------------------------
    # STAGE 7
    # (-y,x)
    # --------------------------------------------------

    def rotate_half(self, x: Tensor):
        
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"Last dimension must be even, got shape {list(x.shape)}."
            )

        x_even = x[..., ::2]    # [..., d/2]  dimensions 0, 2, 4, ...
        x_odd = x[..., 1::2]    # [..., d/2]  dimensions 1, 3, 5, ...

        rotated = torch.stack((-x_odd, x_even), dim=-1)  # [..., d/2, 2]
        return rotated.flatten(-2)

    # --------------------------------------------------
    # STAGE 8
    # actual rope rotation
    # x*cos + rotate_half(x)*sin
    # --------------------------------------------------

    def apply_rotation(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:

        self._validate_rotation_inputs(x, cos, sin)

        cos_full = torch.repeat_interleave(cos, 2, dim=-1)   # [seq_len, dim]
        sin_full = torch.repeat_interleave(sin, 2, dim=-1)   # [seq_len, dim]


        cos_bc = cos_full.unsqueeze(0).unsqueeze(0)   # [1, 1, seq_len, dim]
        sin_bc = sin_full.unsqueeze(0).unsqueeze(0)   # [1, 1, seq_len, dim]

        cos_bc = cos_bc.to(dtype=x.dtype)
        sin_bc = sin_bc.to(dtype=x.dtype)

        return x * cos_bc + self.rotate_half(x) * sin_bc
    
    

    # --------------------------------------------------
    # STAGE 9
    # apply to query
    # --------------------------------------------------

    def apply_to_query(self, q: Tensor, positions: Tensor) -> Tensor:

        self._validate_qk_shape(q, "q")        
        cos, sin = self.get_cos_sin(positions)
        return self.apply_rotation(q, cos, sin)

    # --------------------------------------------------
    # STAGE 10
    # apply to key
    # --------------------------------------------------

    def apply_to_key(self, k: Tensor, positions: Tensor) -> Tensor:

        self._validate_qk_shape(k, "k")
        cos, sin = self.get_cos_sin(positions)
        return self.apply_rotation(k, cos, sin)
    

