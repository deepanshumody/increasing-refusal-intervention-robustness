"""Iterated single-direction ablation matching the existing pipeline exactly.

Per iteration k:
  1. With current ablation directions installed (orthonormalized via QR),
     forward-pass extraction prompts and capture residuals at every (layer,
     position) candidate. Layers are pruned by `--prune_layer_percentage`
     (Arditi default 0.2 → drop the FINAL 20%).
  2. Build candidates as raw mean-diff vectors d^(l,p). Skip if ||d|| ~= 0.
  3. Score each candidate with the existing-directions baseline:
       refusal = mean of [log p(refusal|harmful) - log(1 - p(refusal|harmful))]
                 with `existing + candidate` ablated.
       kl       = KL(baseline || ablated) on harmless first-token probs.
       steering = log-odds of refusal on harmless when the RAW mean-diff is
                  added at the source layer via a forward-pre-hook (coeff=1).
  4. Filter survivors: kl <= kl_threshold AND steering >= induce_refusal_threshold.
     Choose argmin(refusal). Fallback (no survivors) = argmin(refusal + kl_alpha*kl).
  5. Append the unit-normalized chosen direction. Stop on norm collapse.
  6. After every k, generate on the holdout under the current ablation and score
     with WildGuard-7B for the per-K refusal rate.

Refusal-token set defaults to just "I" (Arditi's Llama-3 choice, token 40).
"""
import argparse
import math
import os
import json
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from paper_code.shared.hooks import (
    ResidualCapture, prepare_directions, ablation_context,
    ablation_context_from_list, add_block_input_addition_hook, get_blocks)
from paper_code.shared.wildguard import load_classifier
from paper_code.refusal.eval_refusal import evaluate_refusal


def resolve_refusal_token_ids(tokenizer, token_strings=("I",)):
    ids = []
    for t in token_strings:
        out = tokenizer(t, add_special_tokens=False).input_ids
        if out:
            ids.append(out[0])
    return ids


def encode_chat(tok, prompts, device, max_len=512):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                      tokenize=False, add_generation_prompt=True) for p in prompts]
    return tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)


@torch.no_grad()
def extract_per_layer_position(model, tok, prompts, device, layers, positions,
                               batch_size=32, max_len=512):
    """Returns dict[(l, p)] -> (N, D) numpy array of activations.

    Tokenizer is left-padded so the last real token sits at index T-1.
    """
    model.eval()
    out = {(l, p): [] for l in layers for p in positions}
    for start in range(0, len(prompts), batch_size):
        enc = encode_chat(tok, prompts[start:start + batch_size], device, max_len=max_len)
        T = enc["attention_mask"].size(1)
        with ResidualCapture(model, layers=layers) as cap:
            _ = model(**enc)
        for l in layers:
            h = cap.activations[l]
            for p in positions:
                idx = max(T + p, 0)  # p in {-5..-1}, gives T-5..T-1
                out[(l, p)].append(h[:, idx, :].float().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


@torch.no_grad()
def harmless_baseline_logp(model, tok, prompts, existing_dirs, device, batch_size=8):
    """Next-token log-probs at the last position under the current ablation."""
    model.eval()
    logps = []
    ctx = ablation_context_from_list(model, existing_dirs, device)
    with ctx:
        for start in range(0, len(prompts), batch_size):
            enc = encode_chat(tok, prompts[start:start + batch_size], device)
            out = model(**enc)
            logp = torch.log_softmax(out.logits[:, -1, :], dim=-1)
            logps.append(logp.float().cpu())
    return torch.cat(logps, dim=0)


@torch.no_grad()
def score_candidate_arditi(model, tok, candidate_unit, existing_dirs,
                           prompts_harm, prompts_un, baseline_un_logp,
                           refusal_ids, device, batch_size=8, skip_kl=False):
    """Returns (mean refusal log-odds on harm, mean KL(base||abl) on un)."""
    model.eval()
    dirs = list(existing_dirs)
    if candidate_unit is not None:
        dirs.append(candidate_unit)
    eps = 1e-8
    refusal = []
    kl_sum, n_un = 0.0, 0
    ref_t = torch.tensor(refusal_ids, dtype=torch.long, device=device)

    with ablation_context_from_list(model, dirs, device):
        for start in range(0, len(prompts_harm), batch_size):
            enc = encode_chat(tok, prompts_harm[start:start + batch_size], device)
            out = model(**enc)
            probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
            p_ref = probs[:, ref_t].sum(-1)
            p_non = (1 - p_ref).clamp_min(eps)
            refusal.append((torch.log(p_ref.clamp_min(eps)) - torch.log(p_non)).cpu())

        if not skip_kl:
            idx = 0
            for start in range(0, len(prompts_un), batch_size):
                enc = encode_chat(tok, prompts_un[start:start + batch_size], device)
                out = model(**enc)
                logp = torch.log_softmax(out.logits[:, -1, :], dim=-1).float().cpu()
                B = logp.shape[0]
                base = baseline_un_logp[idx:idx + B]
                p_base = base.exp()
                kl = (p_base * (base - logp)).sum(-1)
                kl_sum += float(kl.sum().item())
                n_un += B
                idx += B

    mean_refusal = float(torch.cat(refusal).mean().item()) if refusal else float("nan")
    mean_kl = kl_sum / max(n_un, 1)
    return mean_refusal, mean_kl


@torch.no_grad()
def steering_score(model, tok, raw_delta, source_layer, existing_dirs,
                   prompts_un, refusal_ids, device, batch_size=8, coeff=1.0):
    """Add coeff * raw_delta at source_layer's input via pre-hook on harmless.
    Score = mean log p(refusal) - log(1 - p(refusal)) at last token. Higher = more induced refusal.
    """
    model.eval()
    eps = 1e-8
    ref_t = torch.tensor(refusal_ids, dtype=torch.long, device=device)
    delta = torch.tensor(raw_delta, dtype=torch.float32, device=device)
    scores = []
    with ablation_context_from_list(model, existing_dirs, device), \
         add_block_input_addition_hook(model, delta, source_layer, coeff=coeff):
        for start in range(0, len(prompts_un), batch_size):
            enc = encode_chat(tok, prompts_un[start:start + batch_size], device)
            out = model(**enc)
            probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
            p_ref = probs[:, ref_t].sum(-1)
            p_non = (1 - p_ref).clamp_min(eps)
            scores.append((torch.log(p_ref.clamp_min(eps)) - torch.log(p_non)).cpu())
    return float(torch.cat(scores).mean().item()) if scores else float("nan")


def run_iterated(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    train = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    holdout = pd.read_parquet(os.path.join(args.data_dir, "holdout.parquet"))

    # Score sets: balanced subset of TRAIN
    rng_seed = args.seed
    train_h = train[train["prompt_harm_label"] == "harmful"]
    train_u = train[train["prompt_harm_label"] == "unharmful"]
    train_h = train_h.sample(min(args.score_n, len(train_h)), random_state=rng_seed)
    train_u = train_u.sample(min(args.score_n, len(train_u)), random_state=rng_seed)
    extract_stmts = pd.concat([train_h, train_u]).reset_index(drop=True)
    extract_labels = (extract_stmts["prompt_harm_label"] == "harmful").astype(int).values
    score_harm = train_h["prompt"].tolist()
    score_un = train_u["prompt"].tolist()

    refusal_ids = resolve_refusal_token_ids(tok, ("I",))
    print(f"refusal_token_ids: {refusal_ids}")

    blocks, _ = get_blocks(model)
    n_layers = len(blocks)
    cutoff = int(n_layers * (1.0 - args.prune_layer_percentage))
    candidate_layers = list(range(cutoff))
    candidate_positions = [int(x) for x in args.scan_positions.split(",")]
    print(f"Layer pruning: keep layer < {cutoff} (drop final {args.prune_layer_percentage*100:.0f}%); "
          f"scanning {len(candidate_layers)} layers × {len(candidate_positions)} positions")

    print("Loading WildGuard")
    classifier = load_classifier()

    # K=0 baseline behavioral eval
    print("=== K=0 ===")
    k0 = evaluate_refusal(model, tok, holdout, classifier, device=device, gen_batch=args.gen_batch)
    log = [{"K": 0, "refusal_harmful": k0["refusal_harmful"], "refusal_harmless": k0["refusal_harmless"]}]
    print(f"K=0 harmful={k0['refusal_harmful']:.3f} harmless={k0['refusal_harmless']:.3f}")

    directions = []
    chosen_candidates = []
    norms = []
    d1_norm = None

    for k in range(args.K_max):
        print(f"--- iteration {k+1} ---")
        # Phase 1: re-extract residuals under current ablation
        prompts_for_extract = [str(s) for s in extract_stmts["prompt"].tolist()]
        with ablation_context_from_list(model, directions, device):
            acts = extract_per_layer_position(
                model, tok, prompts_for_extract,
                device, candidate_layers, candidate_positions,
                batch_size=args.extract_batch)
        candidates = []
        for (lyr, pos), arr in acts.items():
            mu_h = arr[extract_labels == 1].mean(0)
            mu_u = arr[extract_labels == 0].mean(0)
            delta = mu_h - mu_u
            n = float(np.linalg.norm(delta))
            candidates.append((lyr, pos, delta, n))

        # Phase 2: baseline logp on harmless under current ablation (no candidate)
        baseline_un_logp = harmless_baseline_logp(
            model, tok, score_un, directions, device, batch_size=args.gen_batch)

        scored = []
        for (lyr, pos, delta, n) in candidates:
            if n < 1e-12:
                continue
            unit = delta / n
            refusal, kl = score_candidate_arditi(
                model, tok, unit, directions, score_harm, score_un,
                baseline_un_logp, refusal_ids, device, batch_size=args.gen_batch)
            steer = steering_score(
                model, tok, delta, lyr, directions, score_un, refusal_ids, device,
                batch_size=args.gen_batch, coeff=args.steering_coeff)
            print(f"  layer={lyr:2d} pos={pos:2d} norm={n:.4f} refusal={refusal:+.4f} "
                  f"kl={kl:.4f} steering={steer:+.4f}")
            scored.append((refusal, kl, steer, lyr, pos, delta, n))

        def finite(c):
            return not (math.isnan(c[0]) or math.isnan(c[1]) or math.isnan(c[2]))

        survivors = [c for c in scored
                     if finite(c) and c[1] <= args.kl_threshold and c[2] >= args.induce_refusal_threshold]
        if survivors:
            chosen = min(survivors, key=lambda c: c[0])
            print(f"  Selected (KL<={args.kl_threshold}, steer>={args.induce_refusal_threshold}): "
                  f"layer={chosen[3]} pos={chosen[4]} refusal={chosen[0]:+.4f} "
                  f"kl={chosen[1]:.4f} steer={chosen[2]:+.4f}")
        elif [c for c in scored if finite(c)]:
            chosen = min([c for c in scored if finite(c)],
                         key=lambda c: c[0] + args.kl_alpha * c[1])
            print(f"  Fallback argmin(refusal + {args.kl_alpha}*kl): "
                  f"layer={chosen[3]} pos={chosen[4]} refusal={chosen[0]:+.4f} kl={chosen[1]:.4f}")
        else:
            print("  No candidate available; stopping.")
            break

        _, _, _, best_lyr, best_pos, best_delta, best_norm = chosen
        if best_delta is None or best_norm is None:
            print(f"  Stopping at iter {k+1}: no usable candidate")
            break
        if k == 0:
            d1_norm = best_norm

        norms.append(best_norm)
        chosen_candidates.append((best_lyr, best_pos))
        print(f"  Direction {k+1}: layer={best_lyr} pos={best_pos} norm={best_norm:.4f} "
              f"({100*best_norm/d1_norm:.1f}% of d1)")

        if best_norm < args.norm_threshold * d1_norm and k > 0:
            print(f"  Stopping: norm collapse {best_norm:.6f} < {args.norm_threshold} * {d1_norm:.6f}")
            break
        if best_norm < 1e-12:
            print("  Stopping: zero-norm direction")
            break

        directions.append(best_delta / best_norm)

        # Per-K behavioural eval under current ablation
        with ablation_context_from_list(model, directions, device):
            kK = evaluate_refusal(model, tok, holdout, classifier,
                                  device=device, gen_batch=args.gen_batch)
        print(f"K={k+1} harmful={kK['refusal_harmful']:.3f} harmless={kK['refusal_harmless']:.3f}")
        log.append({
            "K": k + 1,
            "refusal_harmful": kK["refusal_harmful"],
            "refusal_harmless": kK["refusal_harmless"],
            "selected_layer": int(best_lyr),
            "selected_position": int(best_pos),
            "norm": float(best_norm),
        })

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"model": args.model, "log": log,
                   "chosen_candidates": chosen_candidates,
                   "norms": norms}, f, indent=2)
    if directions:
        torch.save(torch.tensor(np.stack([np.asarray(d) for d in directions], 0)),
                   args.out_json.replace(".json", "_directions.pt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--K_max", type=int, default=20)
    p.add_argument("--score_n", type=int, default=64,
                   help="Per-class size of extraction + scoring set")
    p.add_argument("--extract_batch", type=int, default=32)
    p.add_argument("--gen_batch", type=int, default=8)
    p.add_argument("--scan_positions", type=str, default="-5,-4,-3,-2,-1")
    p.add_argument("--prune_layer_percentage", type=float, default=0.2,
                   help="Drop final X fraction of layers from candidates (Arditi default 0.2)")
    p.add_argument("--kl_threshold", type=float, default=0.1)
    p.add_argument("--kl_alpha", type=float, default=1.0)
    p.add_argument("--induce_refusal_threshold", type=float, default=0.0)
    p.add_argument("--steering_coeff", type=float, default=1.0)
    p.add_argument("--norm_threshold", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_iterated(args)


if __name__ == "__main__":
    main()
