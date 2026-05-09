"""Fine-tune Llama-3.2-1B-Instruct with mean/cov matching + optional KD.

Matches the existing pipeline:
  - Tokenize PROMPT-ONLY with chat template + add_generation_prompt=True (no
    response in the input).
  - LM loss over the entire prompt sequence (HF default, no masking).
  - Per-layer matching across every transformer block (hidden_states[1:]).
  - Pool: mixed_batch_readout (last/random/chat ratios; remainder mean-pool).
  - KD: T² · KL(p_T || p_S) at the last non-pad position; default T=1.0.
"""
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from paper_code.shared.losses import (
    compute_mean_penalty_with_type, compute_cov_penalty, _per_layer_penalty, kd_kl_loss)
from paper_code.sentiment.train_gpt2 import mixed_batch_readout


class RefusalDataset(Dataset):
    def __init__(self, df, tok, max_len=512):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        prompt = str(row["prompt"])
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, add_special_tokens=False, truncation=True,
                       max_length=self.max_len)["input_ids"]
        label = 1 if row["prompt_harm_label"] == "harmful" else 0
        return {"ids": ids, "label": label}


def collate(batch, pad_id):
    L = max(len(b["ids"]) for b in batch)
    input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(batch), L), dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    cls = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        input_ids[i, :n] = torch.tensor(b["ids"])
        attn_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(b["ids"])  # LM loss over the entire prompt
        cls[i] = b["label"]
    return {"input_ids": input_ids, "attention_mask": attn_mask,
            "labels": labels, "cls": cls}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--source_model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--match", choices=["none", "mean", "cov"], default="none")
    p.add_argument("--mean_penalty_type", choices=["l1", "l2"], default="l2")
    p.add_argument("--cov_penalty_type", choices=["l1", "l2"], default="l2")
    p.add_argument("--lambda_mean", type=float, default=100.0)
    p.add_argument("--lambda_cov", type=float, default=0.0)
    p.add_argument("--multi_layer", type=int, default=1)
    p.add_argument("--last_token_ratio", type=float, default=0.3333)
    p.add_argument("--random_pool_ratio", type=float, default=0.3333)
    p.add_argument("--random_pool_token_coverage", type=float, default=0.5)
    p.add_argument("--chat_template_pool_ratio", type=float, default=0.0)
    p.add_argument("--chat_template_positions", type=str, default="-5,-4,-3,-2,-1")
    p.add_argument("--kd_lambda", type=float, default=0.0)
    p.add_argument("--kd_T", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--micro_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.source_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    student = AutoModelForCausalLM.from_pretrained(args.source_model, torch_dtype=torch.bfloat16).to(device)
    teacher = None
    if args.kd_lambda > 0:
        teacher = AutoModelForCausalLM.from_pretrained(args.source_model, torch_dtype=torch.bfloat16).to(device)
        teacher.eval()
        for q in teacher.parameters(): q.requires_grad_(False)

    train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    ds = RefusalDataset(train_df, tok, max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.micro_batch, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    multi_layer = bool(args.multi_layer)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max((len(loader) // args.grad_accum) * args.epochs, 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

    def pool_fn(hs, am):
        return mixed_batch_readout(
            hs, am,
            random_pool_ratio=args.random_pool_ratio,
            random_pool_token_coverage=args.random_pool_token_coverage,
            last_token_ratio=args.last_token_ratio,
            chat_template_pool_ratio=args.chat_template_pool_ratio,
            chat_template_positions=args.chat_template_positions,
        )

    student.train()
    step = 0
    opt.zero_grad()
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            cls = batch["cls"].to(device)

            out = student(input_ids=input_ids, attention_mask=am,
                          labels=labels, output_hidden_states=True)
            lm_loss = out.loss
            loss = lm_loss

            mean_p = torch.tensor(0.0, device=device)
            cov_p = torch.tensor(0.0, device=device)
            if args.match == "mean":
                mean_p = _per_layer_penalty(out.hidden_states, am, cls, pool_fn,
                                            "mean", args.mean_penalty_type, multi_layer)
                loss = loss + args.lambda_mean * mean_p
            elif args.match == "cov":
                cov_p = _per_layer_penalty(out.hidden_states, am, cls, pool_fn,
                                           "cov", args.cov_penalty_type, multi_layer)
                loss = loss + args.lambda_cov * cov_p

            kd = torch.tensor(0.0, device=device)
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(input_ids=input_ids, attention_mask=am)
                last_idx = (am.sum(-1) - 1).clamp_min(0)
                arange = torch.arange(input_ids.size(0), device=device)
                s_logits = out.logits[arange, last_idx]
                t_logits = t_out.logits[arange, last_idx]
                kd = kd_kl_loss(s_logits, t_logits, T=args.kd_T)
                loss = loss + args.kd_lambda * kd

            (loss / args.grad_accum).backward()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"ep{epoch} step{step} lm={lm_loss.item():.4f} "
                          f"mean={float(mean_p):.4f} cov={float(cov_p):.4f} kd={float(kd):.4f}")

    student.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
