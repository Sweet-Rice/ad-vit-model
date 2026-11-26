"""ViT encoder over the parcel grid: patch embedding, pre-norm encoder, output heads.

Follows architecture.md (Forward pass, Patch embedding, Position embedding, Encoder block,
Output heads, Code skeleton). Shapes at the default config (B = batch):

    bits         B x 512 x 4 x 4    float, +1/-1; bit k of patch (r, c) is parcel bit 4c + k
    mask         B x 512 x 4        bool, True = hidden patch (Stage A)
    hidden       B x 2048 x d       token 4r + c is row r, column c
    field_type   B x 512 x 4 x 4 x 5   per-bit logits, aligned with bits
    grouping     B x 512 x 4 x 4 x 16  per-bit grouping vectors, aligned with bits
    start        B x 512            start logit per row
    length       B x 512 x 4        logits for 1, 2, 3, 4 parcels per row
    masked_patch B x 2048 x 16      logits over the 16 patch values (value = sum_k bit_k << k)
    instr_embed  N x 128            unit vectors, one per (batch index, start row)

Heads only run when asked for, so a Stage A step builds no graph through the Stage B heads
and a Stage B step none through the masked-patch head.

Attention implementation (``ViTEncoder(cfg, attention=...)`` or ``set_attention``): a run-time
choice, not part of ``ViTConfig``; every mode uses the same modules and parameters, so the state
dict is the same and a checkpoint from one mode loads in any other.
    "mha"   nn.TransformerEncoder's own forward (nn.MultiheadAttention). The default.
    "sdpa"  a hand-written pre-norm block over each nn.TransformerEncoderLayer's parameters
            (norm1, self_attn.in_proj_weight / in_proj_bias, self_attn.out_proj, norm2, linear1,
            linear2) computing what the layer computes with dropout 0, attention by
            F.scaled_dot_product_attention.
``compile_encoder()`` compiles the embedding and encoder blocks (``torch.compile``,
``dynamic=False``) for training forward passes; the heads, and forward passes without
gradients, stay eager (see ``compile_encoder``). The model object, its parameters and its state dict
are unchanged by compilation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION = ("mha", "sdpa")

FIELD_TYPE = "field_type"
GROUPING = "grouping"
START_LENGTH = "start_length"
INSTR_EMBED = "instr_embed"
MASKED_PATCH = "masked_patch"
ALL_HEADS = (FIELD_TYPE, GROUPING, START_LENGTH, INSTR_EMBED, MASKED_PATCH)
STAGE_A_HEADS = (MASKED_PATCH,)
STAGE_B_HEADS = (FIELD_TYPE, GROUPING, START_LENGTH, INSTR_EMBED)


@dataclass(frozen=True)
class ViTConfig:
    rows: int = 512
    cols: int = 4
    patch_bits: int = 4
    d: int = 256
    blocks: int = 8
    heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    field_types: int = 5          # fixed, register, immediate, other variable, not-code
    group_width: int = 16
    lengths: int = 4              # 1 to 4 parcels
    instr_width: int = 128
    masked_patch_head: bool = True  # Stage A only; Stage B builds the model without it

    @property
    def tokens(self) -> int:
        return self.rows * self.cols

    @property
    def patch_values(self) -> int:
        return 1 << self.patch_bits


class PatchEmbed(nn.Module):
    """e = W_bit b + beta per patch; mask token for hidden patches; add row and column tables."""

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.patch_bits, cfg.d)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d))
        self.row = nn.Embedding(cfg.rows, cfg.d)
        self.col = nn.Embedding(cfg.cols, cfg.d)

    def content(self, bits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Patch vectors before position embeddings: B x rows x cols x d."""
        x = self.proj(bits)
        if mask is not None:
            x = torch.where(mask[..., None], self.mask_token.to(x.dtype), x)
        return x

    def forward(self, bits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.content(bits, mask)
        x = x + self.row.weight[:, None, :] + self.col.weight[None, :, :]
        return x.flatten(1, 2)


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


def encoder_block(layer: nn.TransformerEncoderLayer, x: torch.Tensor, attention: str) -> torch.Tensor:
    """What ``layer`` (norm_first=True, batch_first=True, dropout 0) computes on x (B x N x d),
    written out over its own parameters, with attention by F.scaled_dot_product_attention
    ("sdpa")."""
    attn = layer.self_attn
    b, n, d = x.shape
    h = attn.num_heads
    qkv = F.linear(layer.norm1(x), attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = qkv.view(b, n, 3, h, d // h).permute(2, 0, 3, 1, 4)      # each B x h x N x d/h
    if attention != "sdpa":
        raise ValueError(f"attention {attention!r} has no hand-written block")
    o = F.scaled_dot_product_attention(q, k, v)
    x = x + attn.out_proj(o.transpose(1, 2).reshape(b, n, d))
    return x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))


class ViTEncoder(nn.Module):
    def __init__(self, cfg: ViTConfig = ViTConfig(), attention: str = "mha"):
        super().__init__()
        self.cfg = cfg
        self.attention = "mha"
        self._compiled_encode = None
        self.set_attention(attention)
        d = cfg.d
        self.embed = PatchEmbed(cfg)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.heads, dim_feedforward=cfg.mlp_ratio * d, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        # enable_nested_tensor only matters for padding masks, which we never pass; turning it
        # off silences the warning norm_first=True triggers.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.blocks, norm=nn.LayerNorm(d), enable_nested_tensor=False)

        self.field_type_head = nn.Linear(d, cfg.patch_bits * cfg.field_types)
        self.grouping_head = nn.Linear(d, cfg.patch_bits * cfg.group_width)
        self.start_length_head = _mlp(d, d, 1 + cfg.lengths)
        self.instr_head = _mlp(d, d, cfg.instr_width)
        self.masked_patch_head = (
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, cfg.patch_values))
            if cfg.masked_patch_head else None)
        self.reset_parameters()

    # Initialization. The docs say only "random init". We use the ViT/BERT convention:
    # every Linear weight in the blocks and heads (and the fused attention in-projection)
    # from a normal with std 0.02, truncated at 2 std, biases zero; LayerNorm at 1 / 0.
    # The row and column tables and the mask token also get std 0.02 (nn.Embedding's default
    # N(0, 1) would make position swamp content). The 4 -> d patch projection gets Xavier
    # uniform, as MAE does for its patch projection: with only 4 inputs, std 0.02 would make
    # content vectors (norm ~0.6) about as small as the position vectors (~0.45), while
    # Xavier gives each patch vector a norm of about 2.8, so content dominates at init.
    # Under pre-norm the residual stream's scale does not change what the blocks see.
    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.trunc_normal_(m.in_proj_weight, std=0.02, a=-0.04, b=0.04)
                nn.init.zeros_(m.in_proj_bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.xavier_uniform_(self.embed.proj.weight)
        nn.init.zeros_(self.embed.proj.bias)
        nn.init.trunc_normal_(self.embed.mask_token, std=0.02, a=-0.04, b=0.04)

    def discard_masked_patch_head(self) -> None:
        """Stage B: drop the pretraining head so it has no parameters in the optimizer."""
        self.masked_patch_head = None

    def set_attention(self, attention: str) -> None:
        """Choose the attention implementation (one of ATTENTION; see the module docstring)."""
        if attention not in ATTENTION:
            raise ValueError(f"attention {attention!r} not in {ATTENTION}")
        if attention != "mha" and self.cfg.dropout:
            raise ValueError(f"attention {attention!r} has no dropout; the model has dropout "
                             f"{self.cfg.dropout}")
        self.attention = attention

    def compile_encoder(self) -> None:
        """Compile the embedding and encoder blocks with torch.compile(dynamic=False), for
        forward passes that build a graph (training); each new input shape compiles once.

        On the RX 7800 XT (ROCm 7.2.4, torch 2.9.1, bf16) the compiled graph for mask=None
        gave run-to-run different outputs with NaN in some windows (in every attention mode,
        with and without gradients, at batch 16 and 64), while the same graph with a mask
        matched eager and repeated exactly in every check. So the compiled path always passes
        a mask (all False when none is given, which computes the same). Forward passes without
        gradients (evaluation) stay uncompiled, where the failure was first seen; they are a
        small part of a run."""
        self._compiled_encode = torch.compile(self._encode, dynamic=False)

    def encode(self, bits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """B x rows x cols x patch_bits (+ optional B x rows x cols mask) -> B x tokens x d."""
        if self._compiled_encode is not None and torch.is_grad_enabled():
            if mask is None:
                mask = torch.zeros(bits.shape[:3], dtype=torch.bool, device=bits.device)
            return self._compiled_encode(bits, mask)
        return self._encode(bits, mask)

    def _encode(self, bits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = self.embed(bits, mask)
        if self.attention == "mha":
            return self.encoder(x)
        for layer in self.encoder.layers:
            x = encoder_block(layer, x, self.attention)
        return self.encoder.norm(x)

    def row_vectors(self, hidden: torch.Tensor) -> torch.Tensor:
        """Mean of each row's patch vectors: B x tokens x d -> B x rows x d."""
        cfg = self.cfg
        return hidden.view(hidden.shape[0], cfg.rows, cfg.cols, cfg.d).mean(dim=2)

    def field_type(self, hidden: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        return self.field_type_head(hidden).view(
            hidden.shape[0], cfg.rows, cfg.cols, cfg.patch_bits, cfg.field_types)

    def grouping(self, hidden: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        return self.grouping_head(hidden).view(
            hidden.shape[0], cfg.rows, cfg.cols, cfg.patch_bits, cfg.group_width)

    def start_length(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.start_length_head(self.row_vectors(hidden))
        return out[..., 0], out[..., 1:]

    def masked_patch(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.masked_patch_head is None:
            raise RuntimeError("masked-patch head was discarded")
        return self.masked_patch_head(hidden)

    def instr_embed(
        self, hidden: torch.Tensor, instr_row: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool each instruction's patch vectors, MLP d -> d -> instr_width, L2-normalize.

        instr_row: B x rows int64, the start row of the instruction owning each row, -1 for
        rows owned by none. Any assignment works (true or decoded instructions); rows sharing
        (batch index, start row) are pooled together. Each row has the same number of patches,
        so the mean of row means equals the mean over the instruction's patches.

        Returns (embeddings N x instr_width, batch index N, start row N), sorted by
        (batch index, start row). N = 0 gives empty tensors.
        """
        rows = self.row_vectors(hidden)                          # B x R x d
        valid = instr_row >= 0
        b_idx, r_idx = valid.nonzero(as_tuple=True)
        keys = torch.stack([b_idx, instr_row[b_idx, r_idx]], dim=1)
        if keys.shape[0] == 0:
            empty = torch.zeros(0, dtype=torch.long, device=hidden.device)
            return hidden.new_zeros(0, self.cfg.instr_width), empty, empty
        uniq, inverse = torch.unique(keys, dim=0, return_inverse=True)
        n = uniq.shape[0]
        vecs = rows[b_idx, r_idx]                                # M x d
        sums = vecs.new_zeros(n, vecs.shape[1]).index_add_(0, inverse, vecs)
        counts = vecs.new_zeros(n).index_add_(0, inverse, vecs.new_ones(inverse.shape[0]))
        pooled = sums / counts[:, None]
        emb = F.normalize(self.instr_head(pooled), dim=-1)
        return emb, uniq[:, 0], uniq[:, 1]

    def forward(
        self,
        bits: torch.Tensor,
        mask: torch.Tensor | None = None,
        heads: tuple[str, ...] | list[str] = (),
        instr_row: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode, then run only the requested heads (names from ALL_HEADS).

        Always returns "hidden". Head keys: "field_type", "grouping", "start", "length",
        "masked_patch", and for "instr_embed": "instr_embed", "instr_batch", "instr_start"
        (needs instr_row).
        """
        unknown = set(heads) - set(ALL_HEADS)
        if unknown:
            raise ValueError(f"unknown heads: {sorted(unknown)}")
        hidden = self.encode(bits, mask)
        out = {"hidden": hidden}
        if FIELD_TYPE in heads:
            out["field_type"] = self.field_type(hidden)
        if GROUPING in heads:
            out["grouping"] = self.grouping(hidden)
        if START_LENGTH in heads:
            out["start"], out["length"] = self.start_length(hidden)
        if INSTR_EMBED in heads:
            if instr_row is None:
                raise ValueError("instr_embed head needs instr_row")
            out["instr_embed"], out["instr_batch"], out["instr_start"] = self.instr_embed(
                hidden, instr_row)
        if MASKED_PATCH in heads:
            out["masked_patch"] = self.masked_patch(hidden)
        return out


def param_groups(model: nn.Module, weight_decay: float = 0.01) -> list[dict]:
    """AdamW groups: decay on matrices; none on LayerNorm, biases, embedding tables.

    Every 1-D parameter is a bias, a LayerNorm gain/shift, or the mask token (treated as an
    embedding table); every nn.Embedding weight (row and column tables) is an embedding table.
    """
    table_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or id(p) in table_ids else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _count(module: nn.Module | None) -> int:
    return 0 if module is None else sum(p.numel() for p in module.parameters())


def count_parameters(model: ViTEncoder) -> dict[str, int]:
    """Parameter counts per component, matching the rows of the doc's parameter budget."""
    e = model.embed
    blocks = model.encoder.layers
    counts = {
        "patch_embed_and_mask_token": _count(e.proj) + e.mask_token.numel(),
        "position_tables": _count(e.row) + _count(e.col),
        "attention_per_block": _count(blocks[0].self_attn),
        "mlp_per_block": _count(blocks[0].linear1) + _count(blocks[0].linear2),
        "block": _count(blocks[0]),
        "blocks": _count(blocks),
        "final_layernorm": _count(model.encoder.norm),
        "field_type_head": _count(model.field_type_head),
        "grouping_head": _count(model.grouping_head),
        "start_length_head": _count(model.start_length_head),
        "instr_embed_head": _count(model.instr_head),
        "masked_patch_head": _count(model.masked_patch_head),
    }
    counts["finetuning_heads"] = sum(counts[k] for k in (
        "field_type_head", "grouping_head", "start_length_head", "instr_embed_head"))
    counts["total"] = _count(model)
    counts["total_without_masked_patch"] = counts["total"] - counts["masked_patch_head"]
    return counts
