"""Frozen timm ViTs with a controllable attention-edge mask.

Every timm ``Attention`` module in ``model.blocks`` is patched so that, while a
mask is set on the ``EdgeController``, attention logits for removed edges are
set to -inf before the softmax (remaining attention renormalizes), or --
ablation ``renorm=False`` -- zeroed after the softmax without renormalizing.
The controller can also record per-layer attention homophily and last-block
token norms from the unmasked pass.
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field

import timm
import torch
import torch.nn as nn

REGISTRY = {
    # key: (timm id, head type, extra create kwargs)
    "deit_s": ("deit_small_patch16_224.fb_in1k", "native", {}),
    "deit3_s": ("deit3_small_patch16_224.fb_in1k", "native", {}),
    "vit_s_augreg": ("vit_small_patch16_224.augreg_in21k_ft_in1k", "native", {}),
    "dino_s": ("vit_small_patch16_224.dino", "probe", {}),
    "dinov2_s": ("vit_small_patch14_dinov2.lvd142m", "probe", {"img_size": 224}),
    "dinov2_s_reg": ("vit_small_patch14_reg4_dinov2.lvd142m", "probe", {"img_size": 224}),
    "clip_b": ("vit_base_patch16_clip_224.openai", "zeroshot", {}),
    # ViT-Base scale-up (final run, plan §18)
    "deit_b": ("deit_base_patch16_224.fb_in1k", "native", {}),
    "dinov2_b": ("vit_base_patch14_dinov2.lvd142m", "probe", {"img_size": 224}),
    "dinov2_b_reg": ("vit_base_patch14_reg4_dinov2.lvd142m", "probe", {"img_size": 224}),
}


@dataclass
class EdgeController:
    keep: torch.Tensor | None = None       # (B, T, T) bool, True = edge kept
    layers: set[int] | None = None         # blocks the mask applies to (None = all)
    renorm: bool = True
    labels: torch.Tensor | None = None     # (B, T) bool, set when recording
    record: bool = False
    att_homophily: dict[int, torch.Tensor] = field(default_factory=dict)

    def active(self, layer: int) -> bool:
        return self.keep is not None and (self.layers is None or layer in self.layers)


def _patched_forward(self, x, attn_mask=None, **kwargs):
    ctrl: EdgeController = self._edge_ctrl
    layer = self._edge_layer
    # Mirrors timm 1.0.x Attention.forward (non-fused path) with the edge mask added.
    B, T, _ = x.shape
    gate = self.gate(x).sigmoid() if getattr(self, "gate", None) is not None else None
    qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    logits = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, T, T)
    if attn_mask is not None:
        logits = (logits.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool
                  else logits + attn_mask)
    use = ctrl.active(layer)
    if use and ctrl.renorm:
        logits = logits.masked_fill(~ctrl.keep[:, None], float("-inf"))
    attn = logits.softmax(-1)
    if use and not ctrl.renorm:
        attn = attn * ctrl.keep[:, None]
    if ctrl.record and ctrl.labels is not None:
        same = (ctrl.labels[:, :, None] == ctrl.labels[:, None, :]).float()  # (B,T,T)
        # mean over heads and query tokens of attention mass on same-label keys
        ctrl.att_homophily[layer] = (attn.mean(1) * same).sum(-1).mean(-1).detach()
    attn = self.attn_drop(attn)
    out = (attn @ v).transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
    out = self.norm(out)
    if gate is not None:
        out = out * gate
    out = self.proj(out)
    return self.proj_drop(out)


class ProbedViT(nn.Module):
    def __init__(self, key: str, pretrained: bool = True):
        super().__init__()
        timm_id, self.head_type, kw = REGISTRY[key]
        self.key = key
        self.model = timm.create_model(timm_id, pretrained=pretrained, **kw).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.ctrl = EdgeController()
        for i, blk in enumerate(self.model.blocks):
            attn = blk.attn
            attn._edge_ctrl, attn._edge_layer = self.ctrl, i
            attn.forward = types.MethodType(_patched_forward, attn)
        self.n_prefix = self.model.num_prefix_tokens  # CLS + registers
        self.patch = self.model.patch_embed.patch_size[0]
        self.n_layers = len(self.model.blocks)
        cfg = self.model.pretrained_cfg
        self.mean, self.std = cfg["mean"], cfg["std"]
        self._last_tokens = None
        self.model.blocks[-1].register_forward_hook(self._grab_last)

    def _grab_last(self, _m, _i, out):
        self._last_tokens = out

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled pre-logit features (CLS readout for every registered model)."""
        f = self.model.forward_features(x)
        return self.model.forward_head(f, pre_logits=True)

    @torch.no_grad()
    def logits_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.model.get_classifier()(feats)

    def last_token_norms(self) -> torch.Tensor:
        return self._last_tokens.norm(dim=-1)  # (B, T)
