"""Unit tests for the class-conditional matching losses and refusal-KD.

All tests run on CPU with small synthetic tensors — no model or dataset
downloads — so they are fast and deterministic enough for CI.
"""
import math

import torch

from intervention_robust_refusal.shared.losses import (
    compute_cov_penalty,
    compute_mean_penalty_with_type,
    kd_kl_loss,
    per_layer_penalty,
)


def test_mean_penalty_hand_computed():
    # pos row [3, 3], neg row [1, 1] -> delta = [2, 2]
    pooled = torch.tensor([[3.0, 3.0], [1.0, 1.0]])
    labels = torch.tensor([1, 0])
    # l1 = mean(|delta|) = 2.0 ; l2 = mean(delta**2) = 4.0
    assert compute_mean_penalty_with_type(pooled, labels, "l1").item() == 2.0
    assert compute_mean_penalty_with_type(pooled, labels, "l2").item() == 4.0


def test_mean_penalty_equal_means_is_zero():
    pooled = torch.tensor([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]])
    labels = torch.tensor([1, 1, 0])  # pos mean [1,1] == neg mean [1,1]
    assert compute_mean_penalty_with_type(pooled, labels, "l2").item() == 0.0


def test_mean_penalty_missing_class_returns_zero():
    pooled = torch.randn(4, 5)
    labels = torch.tensor([1, 1, 1, 1])  # no negatives
    assert compute_mean_penalty_with_type(pooled, labels).item() == 0.0


def test_cov_penalty_same_spread_different_mean_is_zero():
    # Covariance is mean-invariant: identical spread, shifted mean -> ~0.
    pos = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    neg = torch.tensor([[10.0, 0.0], [12.0, 0.0]])  # same centered structure
    pooled = torch.cat([pos, neg], dim=0)
    labels = torch.tensor([1, 1, 0, 0])
    assert compute_cov_penalty(pooled, labels, "l2").item() < 1e-10


def test_cov_penalty_hand_computed():
    # pos cov = [[2,0],[0,0]] (centered [-1,0],[1,0]); neg cov = 0.
    pooled = torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 5.0], [5.0, 5.0]])
    labels = torch.tensor([1, 1, 0, 0])
    # delta = [[2,0],[0,0]]; H=2 ; l2 = 4 / 4 = 1.0 ; l1 = 2 / 4 = 0.5
    assert math.isclose(compute_cov_penalty(pooled, labels, "l2").item(), 1.0, abs_tol=1e-6)
    assert math.isclose(compute_cov_penalty(pooled, labels, "l1").item(), 0.5, abs_tol=1e-6)


def test_cov_penalty_too_few_samples_returns_zero():
    pooled = torch.randn(3, 4)
    labels = torch.tensor([1, 0, 0])  # only one positive -> n_pos < 2
    assert compute_cov_penalty(pooled, labels).item() == 0.0


def test_kd_loss_identical_logits_is_zero():
    logits = torch.randn(3, 7)
    for T in (0.5, 1.0, 2.0):
        assert kd_kl_loss(logits, logits, T=T).item() < 1e-6


def test_kd_loss_hand_computed():
    # teacher = uniform over 2 classes; student logits [1, 0]; T=1.
    teacher = torch.tensor([[0.0, 0.0]])
    student = torch.tensor([[1.0, 0.0]])
    # KL(unif || softmax([1,0])) computed by hand = 0.1201...
    assert math.isclose(kd_kl_loss(student, teacher, T=1.0).item(), 0.1201, abs_tol=2e-3)


def test_kd_loss_temperature_scaling():
    # Distinct logits so temperature genuinely matters (identical logits give 0
    # for any T and cannot validate the T**2 factor / softening).
    teacher = torch.tensor([[0.0, 0.0]])
    student = torch.tensor([[1.0, 0.0]])
    v1 = kd_kl_loss(student, teacher, T=1.0).item()
    v2 = kd_kl_loss(student, teacher, T=2.0).item()
    # T=2 value = 4 * KL(softmax([0,0]/2) || softmax([1,0]/2)) = 0.1237...
    assert math.isclose(v2, 0.1237, abs_tol=2e-3)
    assert not math.isclose(v1, v2, abs_tol=1e-3)  # temperature has a real effect


def test_kd_loss_nonnegative():
    torch.manual_seed(0)
    s, t = torch.randn(4, 10), torch.randn(4, 10)
    assert kd_kl_loss(s, t).item() >= 0.0


def _pool_first_token(h, _mask):
    return h[:, 0, :]


def test_per_layer_penalty_multi_vs_single_layer():
    # hidden_states = (embeddings, block1, block2); embedding layer is skipped.
    emb = torch.zeros(2, 1, 2)
    block1 = torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]])  # delta=[1,1] -> l2 mean 1.0
    block2 = torch.tensor([[[2.0, 2.0]], [[0.0, 0.0]]])  # delta=[2,2] -> l2 mean 4.0
    hidden = (emb, block1, block2)
    mask = torch.ones(2, 1)
    labels = torch.tensor([1, 0])

    multi = per_layer_penalty(hidden, mask, labels, _pool_first_token, "mean", "l2", True)
    single = per_layer_penalty(hidden, mask, labels, _pool_first_token, "mean", "l2", False)
    assert math.isclose(multi.item(), 2.5, abs_tol=1e-6)  # mean(1.0, 4.0)
    assert math.isclose(single.item(), 4.0, abs_tol=1e-6)  # last block only
