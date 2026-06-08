import torch
import torch.nn as nn

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ═══════════════════════════════════════════════════════════════
# RoPE (Rotary Positional Embedding)
# Based on: Su et al., "RoFormer / RoPE"
#
# IDEA:
# Instead of adding position embeddings,
# we ROTATE token vectors in embedding space.
#
# BENEFIT:
# - Encodes relative position naturally
# - Works better for long context than absolute embeddings
# ═══════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    
    def __init__ (self, dim: int, max_len: int = 2048 ):
        
        
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError(
                f"dim must be even for RoPE.\n"
                f"  Got: {dim}\n"
                f"  Fix: use dim=256, 512, 768 etc."
            )
            
        self.dim = dim
        self.max_seq_len = max_len
        
        
        # ─────────────────────────────────────────────
        # 1. Frequency setup (RoPE formula)
        # θ_i = 1 / 10000^(2i/dim)
        # ─────────────────────────────────────────────
        
        half_dim = dim // 2
        i = torch.arange(half_dim, dtype=torch.float32)
        theta = 1.0 / (10000.0 ** (2 * i / dim))
        
        # ─────────────────────────────────────────────
        # 2. Position indices  Φ = pos . θi​
        # ─────────────────────────────────────────────
        positions = torch.arange(max_len, dtype=torch.float32)
        angle = torch.outer(positions, theta) # Φ = pos . θi​
        
        
        # ─────────────────────────────────────────────
        # 3. Precompute trig tables 
        # ─────────────────────────────────────────────
        
        self.cos_table: torch.Tensor  # due to Pylance Warning
        self.sin_table: torch.Tensor  # due to Pylance Warning
        
        self.register_buffer("cos_table", torch.cos(angle)) 
        self.register_buffer("sin_table", torch.sin(angle))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        batch_size, seq_len, dim = x.shape
        
        # Get precomputed trig values for sequence length
        cos = self.cos_table[:seq_len].unsqueeze(0)  
        sin = self.sin_table[:seq_len].unsqueeze(0)
        
        # Split into even/odd dimensions (2D rotation pairs)
        x_even = x[:, :, 0::2]
        x_odd  = x[:, :, 1::2]
        
        # Apply 2D rotation:
        # x' = x cosθ - y sinθ
        # y' = x sinθ + y cosθ
        new_even = x_even * cos - x_odd * sin
        new_odd  = x_even * sin + x_odd * cos
        
        # Merge back into original shape
        x_rotated = torch.stack([new_even, new_odd], dim=-1) 
        x_rotated = x_rotated.flatten(start_dim=-2)
        
        return x_rotated
    
    
# ═══════════════════════════════════════════════════════════════
# Token Embedding Block
#
# Pipeline:
#   Token IDs → Embedding → RoPE → LayerNorm → Dropout
#
# Output:
#   [B, T, D] contextualized token vectors
# ═══════════════════════════════════════════════════════════════
    
class TokenEmbedding(nn.Module):
    
    def __init__(
        self,
        vocab_size : int = config.VOCAB_SIZE,
        embed_dim  : int = config.D_MODEL,
        max_len    : int = config.MAX_SEQ_LEN,
        dropout    : float = config.DROPOUT,
    ):
        
        super().__init__()
        # Step 1: Token lookup table  
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embed_dim,
            padding_idx    = 0,
        )
        
        # Step 2: Positional information via rotation
        self.rope = RotaryEmbedding(
            dim     = embed_dim,
            max_len = max_len * 4,
        )
        
        # Step 3: Stabilization layer (feature normalization)   
        self.norm = nn.LayerNorm(embed_dim)
        
        # Step 4: Regularization
        self.dropout = nn.Dropout(dropout)
        
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        
        
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TextEmbedding ready:")
        print(f"  vocab_size  : {vocab_size:,}")
        print(f"  embed_dim   : {embed_dim}")
        print(f"  max_len     : {max_len}  (RoPE table: {max_len * 4})")
        print(f"  dropout     : {dropout}")
        print(f"  params      : {n_params:,}")
        
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            
            # Step 1: Token lookup → dense vectors
            x = self.embedding(input_ids)
            
            # Step 2: Inject positional structure (RoPE rotation)
            x = self.rope(x)
            
            # Step 3: Normalize feature space (training stability)
            x = self.norm(x)
            
            # Step 4: Regularization (prevents overfitting)
            x = self.dropout(x) 
            
            return x
        
