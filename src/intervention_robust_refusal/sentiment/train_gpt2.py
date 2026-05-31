"""Fine-tune GPT-2 small on IMDB with an optional class-conditional matching loss.

Sentiment proof-of-concept for §4.1. The loss is the standard causal-LM loss
optionally augmented with a mean- or covariance-matching penalty over class-
conditional pooled hidden states (positive vs. negative reviews). Tokenization
is over raw review text — no chat template — and the LM loss covers the entire
input (HuggingFace's default behavior with ``labels=input_ids``).
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup

from intervention_robust_refusal.shared.losses import per_layer_penalty
from intervention_robust_refusal.shared.readouts import mixed_batch_readout


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
                mean_p = per_layer_penalty(out.hidden_states, attn_mask, labels, pool_fn,
                                           "mean", args.mean_penalty_type, multi_layer)
                loss = loss + args.lambda_mean * mean_p
            elif args.match == "cov":
                cov_p = per_layer_penalty(out.hidden_states, attn_mask, labels, pool_fn,
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
