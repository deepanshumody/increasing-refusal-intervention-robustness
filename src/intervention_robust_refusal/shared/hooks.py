"""Residual-stream capture and Arditi-style directional ablation hooks.

The Arditi et al. 2024 single-direction ablation projects out a (set of)
direction(s) from the residual stream at *three* sites per transformer layer:

    1. the residual-stream INPUT to the block (forward-pre-hook on the layer)
    2. the self_attn submodule's output
    3. the mlp submodule's output

Why all three: ablating only the block input leaves attention and MLP free to
re-write the direction back into the stream within the same layer. Hooking
the two submodule outputs as well guarantees the direction is suppressed at
every write site.

The projection is the standard ``a' = a - (a @ D^T) @ D`` for an orthonormal
direction matrix ``D``. ``prepare_directions`` QR-orthonormalises a stack of
raw direction vectors so multi-direction ablation stays a true projector.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext

import numpy as np
import torch


def get_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "llama"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h, "gpt2"
    raise ValueError("Unsupported model architecture")


class ResidualCapture:
    """Capture residual stream at requested transformer-block outputs."""
    def __init__(self, model, layers=None):
        self.model = model
        self.requested = layers
        self.activations = {}
        self.handles = []

    def __enter__(self):
        blocks, _ = get_blocks(self.model)
        idxs = self.requested if self.requested is not None else list(range(len(blocks)))
        for i in idxs:
            self.handles.append(blocks[i].register_forward_hook(self._make_hook(i)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make_hook(self, idx):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.activations[idx] = h
        return hook


def prepare_directions(directions_np, device):
    """Return float32 (K, D) tensor with mutually orthonormal rows.

    For K==1 just unit-normalise. For K>1, QR-orthonormalize via QR on D^T
    and align signs to the originals (for debugging readability).
    """
    if isinstance(directions_np, np.ndarray):
        directions = torch.tensor(directions_np, dtype=torch.float32, device=device)
    else:
        directions = directions_np.to(dtype=torch.float32, device=device)
    K = directions.shape[0]
    if K == 1:
        return directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
    Q, _ = torch.linalg.qr(directions.T)
    orth = Q.T
    signs = torch.sign((orth * directions).sum(dim=1, keepdim=True))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return orth * signs


def _project_out(activations, directions):
    dirs = directions.to(dtype=activations.dtype)
    return activations - (activations @ dirs.T) @ dirs


@contextmanager
def ablation_context(model, directions):
    """Install Arditi-style ablation hooks at every layer (Llama only).

    `directions`: (K, D) tensor with orthonormal rows (use prepare_directions).
    """
    blocks, kind = get_blocks(model)
    if kind != "llama":
        raise ValueError("ablation_context currently supports Llama-style models only")
    handles: list = []

    def make_pre_hook(dirs):
        def hook(module, inp, dirs=dirs):
            h = inp[0]
            return (_project_out(h, dirs),) + inp[1:]
        return hook

    def make_attn_hook(dirs):
        def hook(module, inp, out, dirs=dirs):
            if isinstance(out, tuple):
                return (_project_out(out[0], dirs),) + out[1:]
            return _project_out(out, dirs)
        return hook

    def make_mlp_hook(dirs):
        def hook(module, inp, out, dirs=dirs):
            if isinstance(out, tuple):
                return (_project_out(out[0], dirs),) + out[1:]
            return _project_out(out, dirs)
        return hook

    for layer in blocks:
        handles.append(layer.register_forward_pre_hook(make_pre_hook(directions)))
        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(directions)))
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(directions)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def ablation_context_from_list(model, direction_list, device):
    """Wrap a Python list of raw directions into an ablation_context (or no-op)."""
    if not direction_list:
        return nullcontext()
    arr = np.stack([np.asarray(d) for d in direction_list], axis=0)
    dirs = prepare_directions(arr, device)
    return ablation_context(model, dirs)


@contextmanager
def add_block_input_addition_hook(model, direction, layer_idx, coeff=1.0):
    """Add `coeff * direction` to the residual-stream input of `layer_idx` via a
    forward-pre-hook. Used for Arditi's induce-refusal scoring.
    """
    blocks, _ = get_blocks(model)
    v = torch.as_tensor(direction)

    def hook(module, inp):
        h = inp[0] if isinstance(inp, tuple) else inp
        h = h + (coeff * v).to(dtype=h.dtype, device=h.device)
        if isinstance(inp, tuple):
            return (h,) + inp[1:]
        return h

    handle = blocks[layer_idx].register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()
