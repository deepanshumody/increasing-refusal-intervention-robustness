"""Evaluate a sentiment fine-tune: perplexity, mean gap, linear/MLP probe accuracy.

If --baseline is given, also reports LEACE and INLP applied to that baseline's
frozen embeddings (the upper bound on linear erasure).
"""
import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from intervention_robust_refusal.shared.erasure import inlp_apply, inlp_fit, leace_apply, leace_fit
from intervention_robust_refusal.shared.hooks import ResidualCapture
from intervention_robust_refusal.shared.probes import linear_probe, mlp_probe


@torch.no_grad()
def extract_features(model, tok, texts, layer, device, max_len=256, batch_size=8):
    """Mean-pool over all attended token positions at `layer`."""
    model.eval()
    feats = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with ResidualCapture(model, layers=[layer]) as cap:
            _ = model(**enc)
        h = cap.activations[layer]
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats, 0)


@torch.no_grad()
def perplexity(model, tok, texts, device, max_len=256, batch_size=4):
    model.eval()
    tot_loss, tot = 0.0, 0
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        out = model(**enc, labels=enc["input_ids"])
        n = enc["attention_mask"].sum().item()
        tot_loss += out.loss.item() * n
        tot += n
    return float(np.exp(tot_loss / max(tot, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--max_eval", type=int, default=2000)
    p.add_argument("--baseline", default=None,
                   help="Optional baseline model name for LEACE/INLP comparison")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = GPT2TokenizerFast.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)

    train = load_dataset("imdb", split="train").shuffle(seed=42).select(range(args.max_eval))
    test = load_dataset("imdb", split="test").shuffle(seed=42).select(range(args.max_eval))
    ytr = np.array(train["label"])
    yte = np.array(test["label"])

    Xtr = extract_features(model, tok, train["text"], args.layer, device)
    Xte = extract_features(model, tok, test["text"], args.layer, device)

    mu_p = Xtr[ytr == 1].mean(0)
    mu_n = Xtr[ytr == 0].mean(0)
    print(f"=== {args.model} (layer {args.layer}) ===")
    print(f"L2 mean gap: {np.linalg.norm(mu_p - mu_n):.4f}")
    print(f"Linear probe acc: {linear_probe(Xtr, ytr, Xte, yte):.4f}")
    print(f"MLP probe acc:    {mlp_probe(Xtr, ytr, Xte, yte):.4f}")
    print(f"Perplexity (500 test): {perplexity(model, tok, list(test['text'])[:500], device):.4f}")

    if args.baseline:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bt = GPT2TokenizerFast.from_pretrained(args.baseline)
        bt.pad_token = bt.eos_token
        bm = GPT2LMHeadModel.from_pretrained(args.baseline).to(device)
        Xtr_b = extract_features(bm, bt, train["text"], args.layer, device)
        Xte_b = extract_features(bm, bt, test["text"], args.layer, device)
        mu_b_p = Xtr_b[ytr == 1].mean(0)
        mu_b_n = Xtr_b[ytr == 0].mean(0)
        print(f"\n=== Baseline {args.baseline} frozen embeddings (layer {args.layer}) ===")
        print(f"L2 mean gap: {np.linalg.norm(mu_b_p - mu_b_n):.4f}")
        print(f"Linear probe acc: {linear_probe(Xtr_b, ytr, Xte_b, yte):.4f}")

        mu, P = leace_fit(Xtr_b, ytr)
        leace_acc = linear_probe(leace_apply(Xtr_b, mu, P), ytr, leace_apply(Xte_b, mu, P), yte)
        print(f"LEACE linear probe acc: {leace_acc:.4f}")

        Pin = inlp_fit(Xtr_b, ytr)
        print(f"INLP  linear probe acc: {linear_probe(inlp_apply(Xtr_b, Pin), ytr, inlp_apply(Xte_b, Pin), yte):.4f}")


if __name__ == "__main__":
    main()
