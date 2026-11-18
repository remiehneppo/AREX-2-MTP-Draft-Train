import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import gguf


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        variance = x_float.square().mean(-1, keepdim=True)
        return (x_float * torch.rsqrt(variance + self.eps) * self.weight).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding matching llama.cpp / Qwen3.5.
    Partial RoPE rotates the first `rotary_dim` (64) dimensions of `head_dim` (256).
    """
    def __init__(self, rotary_dim: int = 64, max_seq_len: int = 32768, theta: float = 10000000.0):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, pos_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch_size, seq_len, num_heads, head_dim]
        pos_ids: [batch_size, seq_len] or None (defaults to 0..seq_len-1)
        """
        b_sz, seq_len, n_heads, head_dim = x.shape
        rot_dim = min(self.rotary_dim, head_dim)

        if pos_ids is None:
            pos_ids = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(0).expand(b_sz, -1)
        else:
            pos_ids = pos_ids.float()

        # [B, T, rot_dim // 2]
        freqs = torch.einsum("bt,d->btd", pos_ids, self.inv_freq)
        # For text, all mRoPE position axes are equal. llama.cpp rotates
        # each pair across the two halves of the rotary dimensions.
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(2).to(x.dtype)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(2).to(x.dtype)

        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]

        first, second = x_rot.chunk(2, dim=-1)
        x_rot_pair = torch.cat([-second, first], dim=-1)
        x_rotated = (x_rot * cos) + (x_rot_pair * sin)

        if x_pass.numel() > 0:
            return torch.cat([x_rotated, x_pass], dim=-1)
        return x_rotated


class Qwen35MTPBlock(nn.Module):
    """
    Multi-Token Prediction (MTP) Block for Qwen 3.5 / AREX-2 architecture.
    Faithfully mirrors llama.cpp/src/models/qwen35.cpp `graph_mtp`:
      - eh_proj: Concatenates enorm(token_embd) and hnorm(h_trunk)
      - Interleaved Q and Gate projection per head
      - RMSNorm on Q and K heads
      - Partial RoPE (n_rot=64 out of head_dim=256)
      - Sigmoid gating on self-attention output
      - SwiGLU FFN
      - Final shared_head_norm
    """
    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 17408,
        num_heads: int = 24,
        num_kv_heads: int = 4,
        head_dim: int = 256,
        rotary_dim: int = 64,
        rope_theta: float = 10000000.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.q_heads_dim = num_heads * head_dim       # 6144
        self.kv_heads_dim = num_kv_heads * head_dim   # 1024

        # 1. NextN Input projection
        self.hnorm = RMSNorm(hidden_size, eps=eps)
        self.enorm = RMSNorm(hidden_size, eps=eps)
        self.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # 2. Self-Attention Block
        self.attn_norm = RMSNorm(hidden_size, eps=eps)
        # q_proj projects to 2 * q_heads_dim (interleaved Q and Gate heads)
        self.q_proj = nn.Linear(hidden_size, self.q_heads_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_heads_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_heads_dim, bias=False)

        self.attn_q_norm = RMSNorm(head_dim, eps=eps) # 256
        self.attn_k_norm = RMSNorm(head_dim, eps=eps) # 256
        self.rotary_emb = RotaryEmbedding(rotary_dim=rotary_dim, theta=rope_theta)
        self.o_proj = nn.Linear(self.q_heads_dim, hidden_size, bias=False)

        # 3. Feed-Forward SwiGLU Block
        self.attn_post_norm = RMSNorm(hidden_size, eps=eps)
        self.ffn_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.ffn_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.ffn_down = nn.Linear(intermediate_size, hidden_size, bias=False)

        # 4. Shared Head Output Norm
        self.shared_head_norm = RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        h_63: torch.Tensor,
        token_embd: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass matching graph_mtp in llama.cpp:
          - h_63: [batch_size, seq_len, 5120] (hidden state from trunk model layer 63)
          - token_embd: [batch_size, seq_len, 5120] (embedding of token t+1)
        Returns:
          - mtp_out: [batch_size, seq_len, 5120] (representation ready for lm_head)
        """
        batch_size, seq_len, _ = h_63.shape

        # Step 1: Norm and Concat
        h_norm = self.hnorm(h_63)
        e_norm = self.enorm(token_embd)
        concat = torch.cat([e_norm, h_norm], dim=-1) # [B, T, 10240]
        cur = self.eh_proj(concat)                    # [B, T, 5120]
        inp_sa = cur

        # Step 2: Attention Norm & Gated QKV
        cur = self.attn_norm(cur)

        # q_full has shape [B, T, num_heads * 2 * head_dim]
        # In llama.cpp / Qwen3.5, each head h stores [Q_head, Gate_head] sequentially
        q_full = self.q_proj(cur).view(batch_size, seq_len, self.num_heads, 2, self.head_dim)
        q = q_full[..., 0, :]   # [B, T, num_heads, head_dim]
        gate = q_full[..., 1, :].reshape(batch_size, seq_len, self.q_heads_dim) # [B, T, q_heads_dim]

        k = self.k_proj(cur).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(cur).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Per-head RMSNorm on Q and K
        q = self.attn_q_norm(q) # [B, T, num_heads, head_dim]
        k = self.attn_k_norm(k) # [B, T, num_kv_heads, head_dim]

        # RoPE on Q and K
        q = self.rotary_emb(q, pos_ids=pos_ids)
        k = self.rotary_emb(k, pos_ids=pos_ids)

        # Transpose to [B, num_heads, T, head_dim] for PyTorch SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand KV heads for Grouped-Query Attention (24 heads / 4 kv_heads = 6)
        rep_factor = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(rep_factor, dim=1)
        v = v.repeat_interleave(rep_factor, dim=1)

        # Scaled Dot-Product Attention with causal mask
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=(attention_mask is None)
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.q_heads_dim)

        # Gated output: attn_out = attn_out * sigmoid(gate)
        attn_out = attn_out * torch.sigmoid(gate)
        attn_out = self.o_proj(attn_out)

        # Residual 1
        cur = inp_sa + attn_out

        # Step 3: SwiGLU FFN
        ffn_res = cur
        cur = self.attn_post_norm(cur)
        cur = self.ffn_down(F.silu(self.ffn_gate(cur)) * self.ffn_up(cur))

        # Residual 2
        cur = cur + ffn_res

        # Step 4: Output Norm before LM Head
        cur = self.shared_head_norm(cur)
        return cur


def load_warmstart_from_gguf(model: Qwen35MTPBlock, gguf_path: str):
    """
    Extracts and dequantizes blk.64 weights from a base MTP GGUF file and loads them into Qwen35MTPBlock.
    """
    print(f"Loading and dequantizing warm-start weights from GGUF: {gguf_path}")
    reader = gguf.GGUFReader(gguf_path)
    tensor_map = {t.name: t for t in reader.tensors}

    def get_tensor(name):
        if name not in tensor_map:
            raise KeyError(f"Tensor {name} not found in GGUF")
        t = tensor_map[name]
        data_f32 = gguf.quants.dequantize(t.data, t.tensor_type)
        return torch.from_numpy(data_f32.astype("float32"))

    with torch.no_grad():
        # Norms & Input Projection
        model.hnorm.weight.copy_(get_tensor("blk.64.nextn.hnorm.weight"))
        model.enorm.weight.copy_(get_tensor("blk.64.nextn.enorm.weight"))
        model.eh_proj.weight.copy_(get_tensor("blk.64.nextn.eh_proj.weight"))

        # Attention
        model.attn_norm.weight.copy_(get_tensor("blk.64.attn_norm.weight"))
        model.q_proj.weight.copy_(get_tensor("blk.64.attn_q.weight"))
        model.k_proj.weight.copy_(get_tensor("blk.64.attn_k.weight"))
        model.v_proj.weight.copy_(get_tensor("blk.64.attn_v.weight"))
        model.o_proj.weight.copy_(get_tensor("blk.64.attn_output.weight"))

        # Q and K norms (256 dims)
        model.attn_q_norm.weight.copy_(get_tensor("blk.64.attn_q_norm.weight"))
        model.attn_k_norm.weight.copy_(get_tensor("blk.64.attn_k_norm.weight"))

        # SwiGLU FFN
        model.attn_post_norm.weight.copy_(get_tensor("blk.64.post_attention_norm.weight"))
        model.ffn_gate.weight.copy_(get_tensor("blk.64.ffn_gate.weight"))
        model.ffn_up.weight.copy_(get_tensor("blk.64.ffn_up.weight"))
        model.ffn_down.weight.copy_(get_tensor("blk.64.ffn_down.weight"))

        # Shared Head Norm
        model.shared_head_norm.weight.copy_(get_tensor("blk.64.nextn.shared_head_norm.weight"))

    print(">>> ĐÃ NẠP THÀNH CÔNG TOÀN BỘ 15 CORE MTP TENSORS VÀO PYTORCH MODULE!")
