"""Class-conditional matching losses and KD.

Defaults match the existing pipeline: mean penalty L2 (mean of (Δμ)^2), cov
penalty L2 (squared Frobenius normalized by H*H), KD KL(teacher || student)
at the last non-pad position with default temperature T=1.0.
"""
import torch
import torch.nn.functional as F


def compute_mean_penalty_with_type(pooled, labels, mean_penalty_type="l2"):
    """pooled: [B, H], labels: [B] in {0, 1}. l1 → mean(|Δ|); l2 → mean(Δ²)."""
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


def compute_cov_penalty(pooled, labels, cov_penalty_type="l2"):
    """Centered class-conditional cov diff. l1 → |Δ|.sum()/H²; l2 → (Δ²).sum()/H²."""
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos < 2 or n_neg < 2:
        return pooled.new_tensor(0.0)
    pf = pooled.float()
    p = pf[pos]; q = pf[neg]
    p_c = p - p.mean(0, keepdim=True)
    q_c = q - q.mean(0, keepdim=True)
    cov_p = (p_c.T @ p_c) / (n_pos - 1)
    cov_n = (q_c.T @ q_c) / (n_neg - 1)
    delta = cov_p - cov_n
    H = delta.shape[0]
    if cov_penalty_type == "l1":
        return torch.abs(delta).sum() / (H * H)
    return (delta ** 2).sum() / (H * H)


def _per_layer_penalty(hidden_states, attention_mask, labels, pool_fn,
                       penalty_kind, penalty_type, multi_layer):
    """Apply penalty across hidden_states[1:] (transformer block outputs only) when
    multi_layer, else only the last block.
    """
    layer_states = hidden_states[1:] if multi_layer else (hidden_states[-1],)
    penalties = []
    for hs in layer_states:
        pooled = pool_fn(hs, attention_mask)
        if penalty_kind == "mean":
            penalties.append(compute_mean_penalty_with_type(pooled, labels, penalty_type))
        else:
            penalties.append(compute_cov_penalty(pooled, labels, penalty_type))
    if not penalties:
        return hidden_states[-1].new_tensor(0.0)
    return torch.stack(penalties).mean()


def kd_kl_loss(student_logits, teacher_logits, T=1.0):
    """T² · KL(p_T || p_S) at one position (already-selected logits, [B, V])."""
    s_logp = F.log_softmax(student_logits / T, dim=-1)
    t_p = F.softmax(teacher_logits / T, dim=-1)
    t_logp = F.log_softmax(teacher_logits / T, dim=-1)
    kl = (t_p * (t_logp - s_logp)).sum(-1).mean()
    return T * T * kl
