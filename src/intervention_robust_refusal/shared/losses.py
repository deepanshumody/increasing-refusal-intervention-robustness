"""Class-conditional matching losses and refusal-KD.

Mean and covariance matching equalize the first- and second-order statistics
of class-conditional pooled representations. By Belrose et al. 2023 Theorem
3.1, equal class-conditional means imply a linear probe cannot exceed chance
on the population — so penalizing the empirical mean gap is a soft, finite-
sample analogue of that constraint. Covariance matching pushes redistribution
further by equalizing the *spread* of the two classes, which empirically
forces refusal information across more directions.

The KD term matches the student's next-token distribution at the last input
position to a frozen Instruct teacher's, preserving baseline refusal surface
behavior while the matching loss reshapes the residual stream underneath.
"""
from __future__ import annotations

from typing import Callable, Literal, Sequence

import torch
import torch.nn.functional as F


PenaltyType = Literal["l1", "l2"]
PenaltyKind = Literal["mean", "cov"]
PoolFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def compute_mean_penalty_with_type(
    pooled: torch.Tensor, labels: torch.Tensor, mean_penalty_type: PenaltyType = "l2"
) -> torch.Tensor:
    """Class-conditional mean-difference penalty.

    ``pooled``: ``[B, H]`` pooled representations. ``labels``: ``[B]`` in ``{0, 1}``.
    ``l1`` returns ``mean(|Δμ|)``; ``l2`` returns ``mean(Δμ²)``. Returns 0 if a
    batch lacks one class.
    """
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return pooled.new_tensor(0.0)
    pos_mean = pooled[pos].float().mean(dim=0)
    neg_mean = pooled[neg].float().mean(dim=0)
    delta = pos_mean - neg_mean
    if mean_penalty_type == "l1":
        return torch.mean(torch.abs(delta))
    return torch.mean(delta ** 2)


def compute_cov_penalty(
    pooled: torch.Tensor, labels: torch.Tensor, cov_penalty_type: PenaltyType = "l2"
) -> torch.Tensor:
    """Class-conditional covariance-difference penalty, normalised by H².

    ``l1`` returns ``|ΔΣ|.sum() / H²``; ``l2`` returns ``(ΔΣ²).sum() / H²``
    (squared Frobenius). The H² normalization keeps the penalty's magnitude
    comparable to the mean penalty across hidden sizes.
    """
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos < 2 or n_neg < 2:
        return pooled.new_tensor(0.0)
    pf = pooled.float()
    p, q = pf[pos], pf[neg]
    p_c = p - p.mean(0, keepdim=True)
    q_c = q - q.mean(0, keepdim=True)
    cov_p = (p_c.T @ p_c) / (n_pos - 1)
    cov_n = (q_c.T @ q_c) / (n_neg - 1)
    delta = cov_p - cov_n
    H = delta.shape[0]
    if cov_penalty_type == "l1":
        return torch.abs(delta).sum() / (H * H)
    return (delta ** 2).sum() / (H * H)


def per_layer_penalty(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    pool_fn: PoolFn,
    penalty_kind: PenaltyKind,
    penalty_type: PenaltyType,
    multi_layer: bool,
) -> torch.Tensor:
    """Aggregate the chosen penalty across transformer-block outputs.

    HuggingFace's ``hidden_states`` tuple is ``(embeddings, block_1_out, ...,
    block_L_out)``; we always skip the embedding layer (index 0). With
    ``multi_layer=True`` we average the penalty over every block output; with
    ``False`` we apply it only at the final block. The paper finds multi-layer
    matching materially helps — without it, several configs *increase* probe
    accuracy as the model concentrates the signal in unconstrained layers.
    """
    layer_states = hidden_states[1:] if multi_layer else (hidden_states[-1],)
    penalties = []
    for hs in layer_states:
        pooled = pool_fn(hs, attention_mask)
        if penalty_kind == "mean":
            penalties.append(compute_mean_penalty_with_type(pooled, labels, penalty_type))
        else:
            penalties.append(compute_cov_penalty(pooled, labels, penalty_type))
    return torch.stack(penalties).mean()


def kd_kl_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 1.0
) -> torch.Tensor:
    """Temperature-scaled KL(teacher || student) at one already-selected position.

    Inputs are ``[B, V]`` logits. Returns ``T² · KL`` so the gradient magnitude
    is invariant to ``T`` (Hinton et al. 2015). With ``T=1`` this is just the
    ordinary KL between the teacher's and student's next-token distributions.
    """
    s_logp = F.log_softmax(student_logits / T, dim=-1)
    t_p = F.softmax(teacher_logits / T, dim=-1)
    t_logp = F.log_softmax(teacher_logits / T, dim=-1)
    kl = (t_p * (t_logp - s_logp)).sum(-1).mean()
    return T * T * kl
