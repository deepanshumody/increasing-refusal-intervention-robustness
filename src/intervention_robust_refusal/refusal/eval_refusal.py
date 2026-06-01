"""K=0 evaluation: greedy generation + WildGuard refusal rate, plus linear/MLP
probes on last-token chat-template hidden states at the auto-selected probe layer.

Probe layer default = int(0.65 * num_hidden_layers) (matches existing pipeline).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intervention_robust_refusal.shared.hooks import ResidualCapture
from intervention_robust_refusal.shared.probes import linear_probe, mlp_probe
from intervention_robust_refusal.shared.wildguard import load_classifier, score_refusals


@torch.no_grad()
def generate(model, tok, prompts, batch_size=8, max_new_tokens=256, device="cuda"):
    model.eval()
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        tail = gen[:, enc["input_ids"].size(1):]
        out.extend(tok.batch_decode(tail, skip_special_tokens=True))
    return out


@torch.no_grad()
def last_token_features(model, tok, prompts, layer, device, batch_size=8, max_len=2048):
    """Hidden state at the last (= T-1) position with left-padding, at `layer`."""
    model.eval()
    feats = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        with ResidualCapture(model, layers=[layer]) as cap:
            _ = model(**enc)
        h = cap.activations[layer]
        feats.append(h[:, -1, :].float().cpu().numpy())
    return np.concatenate(feats, 0)


def split_by_label(df):
    harm = df[df["prompt_harm_label"] == "harmful"].reset_index(drop=True)
    un = df[df["prompt_harm_label"] == "unharmful"].reset_index(drop=True)
    return harm, un


def evaluate_refusal(model, tok, holdout_df, classifier, device="cuda", gen_batch=8):
    harm, un = split_by_label(holdout_df)
    harm_resp = generate(model, tok, harm["prompt"].tolist(), batch_size=gen_batch, device=device)
    un_resp = generate(model, tok, un["prompt"].tolist(), batch_size=gen_batch, device=device)
    harm_ref = score_refusals(classifier, harm["prompt"].tolist(), harm_resp)
    un_ref = score_refusals(classifier, un["prompt"].tolist(), un_resp)
    return {
        "refusal_harmful": float(np.mean(harm_ref)) if harm_ref else 0.0,
        "refusal_harmless": float(np.mean(un_ref)) if un_ref else 0.0,
        "harm_responses": harm_resp,
        "harmless_responses": un_resp,
        "harm_refused": list(map(bool, harm_ref)),
        "harmless_refused": list(map(bool, un_ref)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--probe_layer", type=int, default=-1,
                   help="-1 → auto = int(0.65 * num_hidden_layers)")
    p.add_argument("--gen_batch", type=int, default=8)
    p.add_argument("--probe_train_n", type=int, default=2000)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)

    holdout = pd.read_parquet(os.path.join(args.data_dir, "holdout.parquet"))
    train = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    if args.probe_train_n and len(train) > args.probe_train_n:
        train = train.sample(args.probe_train_n, random_state=42).reset_index(drop=True)

    layer = args.probe_layer
    if layer < 0:
        layer = int(0.65 * model.config.num_hidden_layers)

    print(f"Extracting features at layer {layer}")
    Xtr = last_token_features(model, tok, train["prompt"].tolist(), layer, device)
    ytr = (train["prompt_harm_label"] == "harmful").astype(int).values
    Xte = last_token_features(model, tok, holdout["prompt"].tolist(), layer, device)
    yte = (holdout["prompt_harm_label"] == "harmful").astype(int).values

    print("Loading WildGuard")
    classifier = load_classifier()
    print("Generating + scoring on holdout")
    eval_out = evaluate_refusal(model, tok, holdout, classifier, device=device, gen_batch=args.gen_batch)
    eval_out["linear_probe_acc"] = linear_probe(Xtr, ytr, Xte, yte)
    eval_out["mlp_probe_acc"] = mlp_probe(Xtr, ytr, Xte, yte)
    eval_out["probe_layer"] = layer
    eval_out["model"] = args.model

    print(f"Refusal harmful : {eval_out['refusal_harmful']:.3f}")
    print(f"Refusal harmless: {eval_out['refusal_harmless']:.3f}")
    print(f"Linear probe (layer {layer}): {eval_out['linear_probe_acc']:.4f}")
    print(f"MLP probe   (layer {layer}): {eval_out['mlp_probe_acc']:.4f}")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(eval_out, f, indent=2)


if __name__ == "__main__":
    main()
