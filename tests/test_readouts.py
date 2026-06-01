"""Unit tests for the pooling / readout strategies.

These verify that each pooling rule selects the right positions under
right-padding, and that ``mixed_batch_readout`` reduces to a specific rule
when its probability mass is concentrated on that rule. Fixtures are chosen so
that mean-pool, last-token and chat-template pooling give *distinct* answers —
otherwise a test could pass even if the readout silently fell through to the
mean-pool fallback.
"""
import torch

from intervention_robust_refusal.shared.readouts import (
    chat_template_pool_hidden,
    last_token_hidden,
    masked_mean_pool,
    mixed_batch_readout,
)

# Three real tokens, no padding. The three readouts are all different here:
#   mean over {1,2,4} = 2.333 ; last token = 4.0 ; chat("-2,-1") = mean{2,4} = 3.0
H = torch.tensor([[[1.0], [2.0], [4.0]]])
MASK = torch.tensor([[1, 1, 1]])

# A right-padded fixture (third token is padding) for pad-handling checks.
H_PAD = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
MASK_PAD = torch.tensor([[1, 1, 0]])


def test_masked_mean_pool_ignores_padding():
    out = masked_mean_pool(H_PAD, MASK_PAD)  # mean of the two real tokens
    assert torch.allclose(out, torch.tensor([[2.0, 2.0]]), atol=1e-6)


def test_last_token_hidden_picks_final_real_token():
    out = last_token_hidden(H_PAD, MASK_PAD)  # index 1 (last non-pad)
    assert torch.allclose(out, torch.tensor([[3.0, 3.0]]), atol=1e-6)


def test_chat_template_pool_hidden_offsets():
    out = chat_template_pool_hidden(H, MASK, positions_str="-2,-1")  # mean{2,4} = 3.0
    assert torch.allclose(out, torch.tensor([[3.0]]), atol=1e-6)
    # Must differ from a plain mean-pool, or the test proves nothing.
    assert not torch.allclose(out, masked_mean_pool(H, MASK), atol=1e-3)


def test_mixed_readout_defaults_to_mean_pool():
    torch.manual_seed(0)
    out = mixed_batch_readout(H, MASK, random_pool_ratio=0.0, random_pool_token_coverage=0.5)
    assert torch.allclose(out, masked_mean_pool(H, MASK), atol=1e-6)


def test_mixed_readout_all_last_token():
    torch.manual_seed(0)
    out = mixed_batch_readout(
        H, MASK, random_pool_ratio=0.0, random_pool_token_coverage=0.5, last_token_ratio=1.0
    )
    assert torch.allclose(out, last_token_hidden(H, MASK), atol=1e-6)
    assert not torch.allclose(out, masked_mean_pool(H, MASK), atol=1e-3)


def test_mixed_readout_all_chat_template():
    torch.manual_seed(0)
    out = mixed_batch_readout(
        H,
        MASK,
        random_pool_ratio=0.0,
        random_pool_token_coverage=0.5,
        chat_template_pool_ratio=1.0,
        chat_template_positions="-2,-1",
    )
    assert torch.allclose(out, chat_template_pool_hidden(H, MASK, "-2,-1"), atol=1e-6)
    # Proves the chat dispatch ran rather than falling through to mean-pool.
    assert not torch.allclose(out, masked_mean_pool(H, MASK), atol=1e-3)


def test_mixed_readout_random_pool_full_coverage_equals_mean():
    # With full token coverage, the Gumbel-top-k subset is every attended token,
    # so the random-pool branch must reduce exactly to the masked mean (and must
    # exclude the padded position).
    torch.manual_seed(0)
    out = mixed_batch_readout(
        H_PAD, MASK_PAD, random_pool_ratio=1.0, random_pool_token_coverage=1.0
    )
    assert torch.allclose(out, masked_mean_pool(H_PAD, MASK_PAD), atol=1e-6)


def test_mixed_readout_shape():
    torch.manual_seed(0)
    h = torch.randn(4, 6, 8)
    mask = torch.ones(4, 6, dtype=torch.long)
    out = mixed_batch_readout(
        h, mask, random_pool_ratio=0.34, random_pool_token_coverage=0.5,
        last_token_ratio=0.33, chat_template_pool_ratio=0.33,
    )
    assert out.shape == (4, 8)
