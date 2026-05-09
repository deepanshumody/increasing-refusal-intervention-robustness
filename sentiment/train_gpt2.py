"""Fine-tune GPT-2 small on IMDB with optional class-conditional matching loss.

Matches the existing pipeline:
  - tokenize the review text directly (no chat template)
  - LM loss over the entire input (HF default, no masking)
  - mean penalty default L2 (mean of (Δμ)²)
  - cov penalty default L2 ((Δ²).sum()/H²)
  - mixed batch readout: per-sample random assignment to last/random/chat with
    `last_token_ratio`, `random_pool_ratio`, `chat_template_pool_ratio`
  - multi-layer matching uses every transformer block output (hidden_states[1:])
"""
import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup

from paper_code.shared.losses import (
    compute_mean_penalty_with_type, compute_cov_penalty, _per_layer_penalty)


def pooled_last_hidden(h, attn_mask):
    m = attn_mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp_min(1)


def last_token_hidden(h, attn_mask):
    last_idx = (attn_mask.sum(-1) - 1).clamp_min(0)
    B, T, D = h.shape
    idx = last_idx.view(B, 1, 1).expand(-1, 1, D)
    return h.gather(1, idx).squeeze(1)


def chat_template_pool_hidden(h, attn_mask, positions_str="-5,-4,-3,-2,-1"):
    """Average representations at the given (negative) chat-template positions
    (positions are relative to the last *non-pad* token per sample for right-padding)."""
    positions = [int(x) for x in positions_str.split(",")]
    last_idx = attn_mask.sum(-1) - 1
    B, T, D = h.shape
    out = []
    for b in range(B):
        idxs = [int((last_idx[b] + 1 + p).clamp_min(0).item()) for p in positions]
        out.append(h[b, idxs].mean(0))
    return torch.stack(out, 0)


def mixed_batch_readout(h, attn_mask, random_pool_ratio, random_pool_token_coverage,
                        last_token_ratio=0.0, chat_template_pool_ratio=0.0,
                        chat_template_positions="-5,-4,-3,-2,-1"):
    mean_pooled = pooled_last_hidden(h, attn_mask)
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
        result[use_last] = last_token_hidden(h[use_last], attn_mask[use_last])
    if use_random.any():
        hs_r = h[use_random]; m_r = attn_mask[use_random]
        n_r = hs_r.size(0)
        n_valid = m_r.float().sum(1)
        k = (random_pool_token_coverage * n_valid).ceil().clamp(min=1).long()
        k_max = int(k.max().item())
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
            h[use_chat], attn_mask[use_chat], chat_template_positions).to(h.dtype)
    return result


def make_loader(tok, batch_size, max_len=256, max_samples=None, split="train"):
    ds = load_dataset("imdb", split=split)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    def encode(ex):
        e = tok(ex["text"], truncation=True, max_length=max_len, padding="max_length")
        e["label"] = ex["label"]
        return e

    ds = ds.map(encode, batched=False)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", choices=["none", "mean", "cov"], default="none")
    p.add_argument("--mean_penalty_type", choices=["l1", "l2"], default="l2")
    p.add_argument("--cov_penalty_type", choices=["l1", "l2"], default="l2")
    p.add_argument("--lambda_mean", type=float, default=100.0)
    p.add_argument("--lambda_cov", type=float, default=0.0)
    p.add_argument("--multi_layer", type=int, default=1)
    p.add_argument("--last_token_ratio", type=float, default=0.0)
    p.add_argument("--random_pool_ratio", type=float, default=0.0)
    p.add_argument("--random_pool_token_coverage", type=float, default=0.5)
    p.add_argument("--chat_template_pool_ratio", type=float, default=0.0)
    p.add_argument("--chat_template_positions", type=str, default="-5,-4,-3,-2,-1")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    loader = make_loader(tok, args.batch_size, max_samples=args.max_train, split="train")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

    multi_layer = bool(args.multi_layer)

    def pool_fn(hs, am):
        return mixed_batch_readout(
            hs, am,
            random_pool_ratio=args.random_pool_ratio,
            random_pool_token_coverage=args.random_pool_token_coverage,
            last_token_ratio=args.last_token_ratio,
            chat_template_pool_ratio=args.chat_template_pool_ratio,
            chat_template_positions=args.chat_template_positions,
        )

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask,
                        labels=input_ids, output_hidden_states=True)
            lm_loss = out.loss
            loss = lm_loss

            mean_p = torch.tensor(0.0, device=device)
            cov_p = torch.tensor(0.0, device=device)
            if args.match == "mean":
                mean_p = _per_layer_penalty(out.hidden_states, attn_mask, labels, pool_fn,
                                            "mean", args.mean_penalty_type, multi_layer)
                loss = loss + args.lambda_mean * mean_p
            elif args.match == "cov":
                cov_p = _per_layer_penalty(out.hidden_states, attn_mask, labels, pool_fn,
                                           "cov", args.cov_penalty_type, multi_layer)
                loss = loss + args.lambda_cov * cov_p

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            if step % 50 == 0:
                print(f"ep{epoch} step{step}/{len(loader)} lm={lm_loss.item():.4f} "
                      f"mean={float(mean_p):.4f} cov={float(cov_p):.4f}")

    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
