"""Pooling and readout strategies for class-conditional matching losses.

The matching losses (mean/cov) need a single vector per sample. *How* you pool
token activations into that vector matters: a model that homogenizes the
last-token chat-template position can still leak class information at earlier
positions or under different pooling rules. To make matching robust against any
single readout, we use ``mixed_batch_readout``: per sample, randomly pick one
of {last-token, random-subset mean, chat-template-position mean}, with the
remaining mass falling through to a uniform mean over attended tokens.

All functions take ``h`` of shape ``[B, T, D]`` (hidden states) and
``attention_mask`` of shape ``[B, T]`` (right-padded, 1 for real tokens).
"""
from __future__ import annotations

import torch


def pooled_last_hidden(h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean of hidden states over attended (non-pad) positions: ``[B, D]``."""
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp_min(1)


def last_token_hidden(h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden state at each sample's final non-pad position (right-padding): ``[B, D]``."""
    last_idx = (attention_mask.sum(-1) - 1).clamp_min(0)
    B, _, D = h.shape
    idx = last_idx.view(B, 1, 1).expand(-1, 1, D)
    return h.gather(1, idx).squeeze(1)


def chat_template_pool_hidden(
    h: torch.Tensor,
    attention_mask: torch.Tensor,
    positions_str: str = "-5,-4,-3,-2,-1",
) -> torch.Tensor:
    """Mean of hidden states at fixed offsets relative to each sample's last non-pad token.

    Negative offsets are interpreted relative to the last real token: ``-1`` is the
    last real token itself, ``-2`` the one before, etc. (Right-padding only.)
    """
    positions = [int(x) for x in positions_str.split(",")]
    last_idx = attention_mask.sum(-1) - 1
    out = []
    for b in range(h.shape[0]):
        idxs = [int((last_idx[b] + 1 + p).clamp_min(0).item()) for p in positions]
        out.append(h[b, idxs].mean(0))
    return torch.stack(out, 0)


def mixed_batch_readout(
    h: torch.Tensor,
    attention_mask: torch.Tensor,
    random_pool_ratio: float,
    random_pool_token_coverage: float,
    last_token_ratio: float = 0.0,
    chat_template_pool_ratio: float = 0.0,
    chat_template_positions: str = "-5,-4,-3,-2,-1",
) -> torch.Tensor:
    """Per-sample random assignment to one of four pooling strategies.

    With probability ``last_token_ratio`` use ``last_token_hidden``; with
    ``random_pool_ratio`` use a Gumbel-top-k random subset mean covering a
    ``random_pool_token_coverage`` fraction of attended tokens; with
    ``chat_template_pool_ratio`` use ``chat_template_pool_hidden`` at the given
    positions; the remainder fall through to a uniform mean over all attended
    tokens (``pooled_last_hidden``).

    The three ratios must sum to at most 1.0 (any remainder goes to mean-pool).
    """
    mean_pooled = pooled_last_hidden(h, attention_mask)
    if random_pool_ratio == 0.0 and last_token_ratio == 0.0 and chat_template_pool_ratio == 0.0:
        return mean_pooled

    B, T, _ = h.shape
    rand = torch.rand(B, device=h.device)
    use_last = rand < last_token_ratio
    use_random = (rand >= last_token_ratio) & (rand < last_token_ratio + random_pool_ratio)
    use_chat = ((rand >= last_token_ratio + random_pool_ratio)
                & (rand < last_token_ratio + random_pool_ratio + chat_template_pool_ratio))

    result = mean_pooled.clone()
    if use_last.any():
        result[use_last] = last_token_hidden(h[use_last], attention_mask[use_last])
    if use_random.any():
        hs_r = h[use_random]
        m_r = attention_mask[use_random]
        n_r = hs_r.size(0)
        n_valid = m_r.float().sum(1)
        k = (random_pool_token_coverage * n_valid).ceil().clamp(min=1).long()
        k_max = int(k.max().item())
        # Gumbel-top-k sampling: argmax of (log p_i + Gumbel(0,1)) yields a
        # uniform sample from the attended positions, and top-k gives k samples
        # without replacement in one shot.
        gn = -torch.log(-torch.log(torch.rand(n_r, T, device=h.device) + 1e-10) + 1e-10)
        gn = gn * m_r.float() - (1 - m_r.float()) * 1e9
        _, top_idx = gn.topk(k_max, dim=1)
        sm = torch.zeros(n_r, T, device=h.device)
        for i in range(n_r):
            sm[i, top_idx[i, :k[i]]] = 1.0
        sm = sm.unsqueeze(-1)
        rp = (hs_r.float() * sm).sum(1) / sm.sum(1).clamp(min=1.0)
        result[use_random] = rp.to(h.dtype)
    if use_chat.any():
        result[use_chat] = chat_template_pool_hidden(
            h[use_chat], attention_mask[use_chat], chat_template_positions
        ).to(h.dtype)
    return result
