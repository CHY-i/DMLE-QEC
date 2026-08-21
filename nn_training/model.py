import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SyndromeEmbedding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.val_emb = nn.Linear(1, d_model)
        self.pos_emb = nn.Linear(2, d_model)
        self.type_emb = nn.Embedding(2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t: torch.Tensor, coords: torch.Tensor, det_types: torch.Tensor) -> torch.Tensor:
        val_emb = self.val_emb(x_t.unsqueeze(-1).float())
        pos_emb = self.pos_emb(coords.float()).unsqueeze(0)
        type_emb = self.type_emb(det_types.long()).unsqueeze(0)
        return self.dropout(val_emb + pos_emb + type_emb)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if self.head_dim % 4 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be divisible by 4 for 2D RoPE")

        self.rope_base = rope_base
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _rope_cos_sin(self, positions: torch.Tensor, axis_dim: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_base ** (torch.arange(0, axis_dim, 2, device=positions.device, dtype=torch.float32) / axis_dim)
        )
        angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1).to(dtype=dtype)
        sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1).to(dtype=dtype)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def _apply_2d_rope(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        axis_dim = self.head_dim // 2
        x_part = x[..., :axis_dim]
        y_part = x[..., axis_dim:]

        cos_x, sin_x = self._rope_cos_sin(coords[:, 0], axis_dim, x.dtype)
        cos_y, sin_y = self._rope_cos_sin(coords[:, 1], axis_dim, x.dtype)

        x_part = x_part * cos_x + self._rotate_half(x_part) * sin_x
        y_part = y_part * cos_y + self._rotate_half(y_part) * sin_y
        return torch.cat([x_part, y_part], dim=-1)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        q = self._apply_2d_rope(q, coords)
        k = self._apply_2d_rope(k, coords)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.d_model)
        return self.out_dropout(self.out_proj(attn_out))


class SyndromeTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model=d_model, nhead=nhead, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, src: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        src = src + self.self_attn(self.norm1(src), coords)
        src = src + self.ffn(self.norm2(src))
        return src


class SyndromeTransformerCell(nn.Module):
    def __init__(
        self,
        d_model: int = 192,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        use_conv: bool = False,
    ):
        super().__init__()
        del use_conv
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.fusion = nn.Linear(2 * d_model, d_model)
        self.fusion_norm = RMSNorm(d_model)
        self.layers = nn.ModuleList(
            [
                SyndromeTransformerLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

    def forward(self, x_t: torch.Tensor, coords: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros_like(x_t)

        combined_input = torch.cat([x_t, h_prev], dim=-1)
        combined_input = self.fusion_norm(F.gelu(self.fusion(combined_input)))

        h_next = combined_input
        for layer in self.layers:
            h_next = layer(h_next, coords)
        return h_next


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            self.norm_q(query),
            self.norm_kv(context),
            self.norm_kv(context),
            need_weights=False,
        )
        query = query + self.dropout(attn_out)
        query = query + self.ffn(self.norm_ffn(query))
        return query


class ReadoutHead(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.logical_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        )
        self.mlp = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        pooled = h_t.mean(dim=1, keepdim=True)
        query = pooled + self.logical_emb.expand(h_t.shape[0], -1, -1)
        for block in self.cross_blocks:
            query = block(query, h_t)
        return self.mlp(query).squeeze(-1).squeeze(-1)


class AlphaQubits(nn.Module):
    def __init__(
        self,
        d_model: int = 192,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        use_conv: bool = False,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.embedding = SyndromeEmbedding(d_model=d_model, dropout=dropout)
        self.cell = SyndromeTransformerCell(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_conv=use_conv,
        )
        self.readout = ReadoutHead(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    @staticmethod
    def _prepare_metadata(coords: torch.Tensor, det_types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if coords.dim() == 3:
            coords = coords[0]
        if det_types.dim() == 2:
            det_types = det_types[0]
        return coords, det_types

    def forward(
        self,
        x_seq: torch.Tensor,
        coords: torch.Tensor,
        det_types: torch.Tensor,
        h_0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_seq.dim() != 3:
            raise ValueError(f"x_seq should be [Batch, Time, N_det], got shape={tuple(x_seq.shape)}")

        coords, det_types = self._prepare_metadata(coords, det_types)
        coords = coords.to(device=x_seq.device)
        det_types = det_types.to(device=x_seq.device)

        batch_size, seq_len, _ = x_seq.shape
        h_t = h_0
        outputs = []

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            x_emb = self.embedding(x_t, coords, det_types)
            h_t = self.cell(x_emb, coords, h_prev=h_t)
            outputs.append(self.readout(h_t))

        return torch.stack(outputs, dim=1).view(batch_size, seq_len)
