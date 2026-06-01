"""Unit tests for residual-stream capture and directional ablation hooks.

The projection math (``prepare_directions`` orthonormalisation and the
``a - (a @ D^T) @ D`` projector) is the core of the Arditi-style attack the
defense has to survive, so it is tested directly on synthetic tensors.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from intervention_robust_refusal.shared.hooks import (
    _project_out,
    ablation_context,
    ablation_context_from_list,
    get_blocks,
    prepare_directions,
)


def test_prepare_directions_single_is_unit_norm():
    d = torch.tensor([[3.0, 4.0]]).numpy()  # norm 5
    out = prepare_directions(d, device="cpu")
    assert out.shape == (1, 2)
    assert out.dtype == torch.float32
    assert torch.allclose(out.norm(dim=1), torch.ones(1), atol=1e-6)


def test_prepare_directions_multi_is_orthonormal():
    torch.manual_seed(0)
    raw = torch.randn(3, 8).numpy()
    out = prepare_directions(raw, device="cpu")
    gram = out @ out.T
    assert torch.allclose(gram, torch.eye(3), atol=1e-5)


def test_project_out_is_idempotent_and_orthogonal():
    # Ablate the first two coordinate axes from random activations.
    dirs = torch.eye(4)[:2]  # orthonormal rows e0, e1
    acts = torch.randn(5, 4)
    once = _project_out(acts, dirs)
    twice = _project_out(once, dirs)
    assert torch.allclose(once, twice, atol=1e-6)  # idempotent
    # Components along ablated directions are removed; others untouched.
    assert torch.allclose(once[:, :2], torch.zeros(5, 2), atol=1e-6)
    assert torch.allclose(once[:, 2:], acts[:, 2:], atol=1e-6)


def test_prepare_directions_makes_projector_idempotent():
    # The whole point of prepare_directions: raw mean-diff vectors are neither
    # unit-norm nor mutually orthogonal, so projecting them out is NOT a true
    # (idempotent) projector. After QR-orthonormalisation it is, and the span
    # of the original directions is preserved.
    torch.manual_seed(0)
    raw = torch.randn(3, 8)
    acts = torch.randn(5, 8)

    # Raw directions: projecting twice differs from once -> not a projector.
    raw_once = _project_out(acts, raw)
    raw_twice = _project_out(raw_once, raw)
    assert not torch.allclose(raw_once, raw_twice, atol=1e-4)

    # Orthonormalised directions: a genuine idempotent projector...
    orth = prepare_directions(raw.numpy(), device="cpu")
    once = _project_out(acts, orth)
    twice = _project_out(once, orth)
    assert torch.allclose(once, twice, atol=1e-5)

    # ...whose range still spans the original directions (so ablation removes
    # exactly the intended subspace): projecting the raw directions out is ~0.
    assert torch.allclose(_project_out(raw, orth), torch.zeros(3, 8), atol=1e-5)


def test_get_blocks_dispatch_and_error():
    llama = SimpleNamespace(model=SimpleNamespace(layers=[1, 2, 3]))
    gpt2 = SimpleNamespace(transformer=SimpleNamespace(h=[1, 2]))
    assert get_blocks(llama) == ([1, 2, 3], "llama")
    assert get_blocks(gpt2) == ([1, 2], "gpt2")
    with pytest.raises(ValueError):
        get_blocks(SimpleNamespace())


def test_ablation_context_from_list_empty_is_noop():
    ctx = ablation_context_from_list(model=None, direction_list=[], device="cpu")
    assert isinstance(ctx, nullcontext)
    with ctx:  # must not raise even with model=None
        pass


def test_ablation_context_rejects_non_llama():
    gpt2 = SimpleNamespace(transformer=SimpleNamespace(h=[1, 2]))
    with pytest.raises(ValueError):
        with ablation_context(gpt2, torch.eye(2)):
            pass
