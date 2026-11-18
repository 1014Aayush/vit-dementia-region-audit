"""Edge-removal masks for ViT attention graphs.

A ViT with T tokens is a complete directed graph. Following the GNN paper, we
operate on *unordered* token pairs (i < j): removing a pair blocks attention in
both directions (i reads j and j reads i). Removing only one arc would let
information keep flowing through the other, silently changing what "sparsity"
means -- the same pitfall as PyG's two-arc storage of undirected edges.

Self-edges (i attends to i) are never removed; together with the residual
stream they play the role of GCNConv's self-loops / the "ego path".

Node labels: True = "object side", False = "background side".
Prefix tokens (CLS, registers) are assigned to a side by ``readout_side``.
The CLS token must be on the object side for "isolate the object from the
background" to mean anything: otherwise background information reaches the
readout directly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

STRATEGIES = ("random", "cross_first", "same_first")


def patch_labels(fg_mask: torch.Tensor, patch: int, thresh: float = 0.5) -> torch.Tensor:
    """fg_mask: (B, H, W) in [0, 1] aligned with the model input.
    Returns (B, N) bool: patch is object if its foreground fraction > thresh."""
    frac = F.avg_pool2d(fg_mask[:, None].float(), patch)  # (B,1,h,w)
    return frac.flatten(1) > thresh


def token_labels(patch_fg: torch.Tensor, n_prefix: int, readout_side: str = "fg") -> torch.Tensor:
    """Prepend labels for the prefix tokens (CLS [+ registers])."""
    B = patch_fg.shape[0]
    side = {"fg": True, "bg": False}[readout_side]
    prefix = torch.full((B, n_prefix), side, dtype=torch.bool, device=patch_fg.device)
    return torch.cat([prefix, patch_fg], dim=1)


def structural_homophily(labels: torch.Tensor) -> torch.Tensor:
    """Fraction of unordered token pairs with equal labels, per image. (B,)"""
    T = labels.shape[1]
    n_fg = labels.sum(1).float()
    n_bg = T - n_fg
    same = n_fg * (n_fg - 1) / 2 + n_bg * (n_bg - 1) / 2
    return same / (T * (T - 1) / 2)


_TRIU_CACHE: dict = {}


def _triu(T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (T, str(device))
    if key not in _TRIU_CACHE:
        _TRIU_CACHE[key] = tuple(torch.triu_indices(T, T, offset=1, device=device))
    return _TRIU_CACHE[key]


def edge_keep_mask(
    labels: torch.Tensor,
    sparsity: float | None,
    strategy: str,
    generator: torch.Generator | None = None,
    n_remove: torch.Tensor | None = None,
    direction: str = "both",
    generators: list[torch.Generator] | None = None,
) -> torch.Tensor:
    """Build a (B, T, T) bool mask; True = query i may attend to key j.
    Vectorized over the batch; runs on ``labels.device``.

    sparsity: fraction of unordered non-self pairs to remove (ignored if
        ``n_remove`` gives a per-image count).
    strategy: 'random' | 'cross_first' (heterophilic-first oracle) |
        'same_first' (homophilic-first inverse oracle).
    direction: 'both' removes both arcs of each chosen pair (default, the
        paper's semantics). 'into_fg' blocks only arcs where an object-side
        query reads a background-side key; 'into_bg' the reverse. Same-label
        pairs always lose both arcs.
    generators: one CPU generator per image (preferred: makes each image's
        mask independent of batch composition); else ``generator`` draws the
        whole (B, E) block.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    if direction not in ("both", "into_fg", "into_bg"):
        raise ValueError(direction)
    B, T = labels.shape
    dev = labels.device
    iu, ju = _triu(T, dev)
    E = iu.numel()
    if generators is not None:
        r = torch.stack([torch.rand(E, generator=g) for g in generators])
    else:
        r = torch.rand(B, E, generator=generator)
    r = r.to(dev)
    li, lj = labels[:, iu], labels[:, ju]
    cross = li != lj
    if strategy == "cross_first":
        r = r + (~cross).float()
    elif strategy == "same_first":
        r = r + cross.float()
    if n_remove is None:
        k = torch.full((B,), int(round(sparsity * E)), device=dev)
    else:
        k = torch.as_tensor(n_remove, device=dev).long().view(B)
    order = torch.argsort(r, dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(E, device=dev).expand(B, E))
    removed = rank < k[:, None]
    if direction == "both":
        block_ij = block_ji = removed
    else:
        # arc (i -> j) means query i reads key j
        i_is_fg = li if direction == "into_fg" else ~li
        block_ij = removed & (~cross | i_is_fg)
        block_ji = removed & (~cross | ~i_is_fg)
    keep = torch.ones(B, T, T, dtype=torch.bool, device=dev)
    keep[:, iu, ju] = ~block_ij
    keep[:, ju, iu] = ~block_ji
    return keep


def isolate_mask(labels: torch.Tensor, keep_bg_tokens: torch.Tensor | None = None) -> torch.Tensor:
    """All cross-label pairs removed (= cross_first at s = 1 - h, exactly).

    keep_bg_tokens: optional (B, T) bool of background-side tokens whose edges
        to the object side are *retained* (used for the high-norm "artifact
        token" reconnection test and its random-token control)."""
    keep = labels[:, :, None] == labels[:, None, :]
    if keep_bg_tokens is not None:
        k = keep_bg_tokens
        keep = keep | k[:, None, :] | k[:, :, None]
    return keep


def n_cross_pairs(labels: torch.Tensor) -> torch.Tensor:
    n_fg = labels.sum(1)
    return n_fg * (labels.shape[1] - n_fg)


def edge_keep_mask_reference(
    labels: torch.Tensor,
    sparsity: float | None,
    strategy: str,
    generator: torch.Generator | None = None,
    n_remove: torch.Tensor | None = None,
    direction: str = "both",
) -> torch.Tensor:
    """Original per-image loop, kept as the reference the vectorized version is tested against.

    Build a (B, T, T) bool mask; True = query i may attend to key j.

    sparsity: fraction of unordered non-self pairs to remove (ignored if
        ``n_remove`` gives a per-image count).
    strategy: 'random' | 'cross_first' (heterophilic-first oracle) |
        'same_first' (homophilic-first inverse oracle).
    direction: 'both' removes both arcs of each chosen pair (default, the
        paper's semantics). 'into_fg' blocks only arcs where an object-side
        query reads a background-side key; 'into_bg' the reverse. Same-label
        pairs always lose both arcs.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    B, T = labels.shape
    dev = labels.device
    iu, ju = torch.triu_indices(T, T, offset=1, device=dev)
    E = iu.numel()
    keep = torch.ones(B, T, T, dtype=torch.bool, device=dev)
    for b in range(B):
        k = int(n_remove[b]) if n_remove is not None else int(round(sparsity * E))
        if k <= 0:
            continue
        lab = labels[b]
        cross = lab[iu] != lab[ju]
        r = torch.rand(E, generator=generator, device="cpu").to(dev)
        if strategy == "cross_first":
            r = r + (~cross).float()  # cross pairs sort first
        elif strategy == "same_first":
            r = r + cross.float()
        idx = torch.argsort(r)[:k]
        i, j = iu[idx], ju[idx]
        if direction == "both":
            keep[b, i, j] = False
            keep[b, j, i] = False
        else:
            c = cross[idx]
            # orient each removed cross pair as (fg_tok, bg_tok)
            fg_tok = torch.where(lab[i], i, j)
            bg_tok = torch.where(lab[i], j, i)
            if direction == "into_fg":
                keep[b, fg_tok[c], bg_tok[c]] = False
            elif direction == "into_bg":
                keep[b, bg_tok[c], fg_tok[c]] = False
            else:
                raise ValueError(direction)
            keep[b, i[~c], j[~c]] = False
            keep[b, j[~c], i[~c]] = False
    return keep
