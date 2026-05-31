"""WildGuardMix → train/val split.

Steps (matches existing pipeline):
  1. Load allenai/wildguardmix wildguardtrain. Keep non-adversarial rows with
     valid prompt_harm_label.
  2. Generate fresh responses with the source model (Llama-3.2-1B-Instruct).
  3. Score each (prompt, response) with WildGuard-7B → predicted_refusal.
  4. (Optional, --filter_train) drop rows where the generated behaviour
     disagrees with prompt_harm_label.
  5. Stratified 80/20 train/val split (seed 42).

Each output row carries both `prompt_harm_label` (gold) and
`wildguard_reference_label` (WildGuardMix's response_refusal_label) plus the
freshly generated `response` and `predicted_refusal`. Downstream consumers can
choose which column to use as the active refusal label.
"""
import argparse
import os
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from intervention_robust_refusal.shared.wildguard import load_classifier, score_full


@torch.no_grad()
def generate_responses(model_name, prompts, batch_size=8, max_new_tokens=256, device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        tail = gen[:, enc["input_ids"].size(1):]
        out.extend(tok.batch_decode(tail, skip_special_tokens=True))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def stratified_train_val_split(df, label_col, train_frac=0.8, seed=42):
    """Per-class shuffled split, then concatenate. Matches refusal_ablation/datasets.py."""
    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed)
    df = df.iloc[rng.values].reset_index(drop=True)
    train_parts, val_parts = [], []
    for cls in df[label_col].unique():
        sub = df[df[label_col] == cls]
        n_train = max(1, round(len(sub) * train_frac))
        train_parts.append(sub.iloc[:n_train])
        val_parts.append(sub.iloc[n_train:])
    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--gen_batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--filter_train", action="store_true",
                   help="Drop rows where predicted_refusal disagrees with prompt_harm_label.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    df = ds.to_pandas()
    if "adversarial" in df.columns:
        df = df[df["adversarial"] == False]
    df = df.dropna(subset=["prompt_harm_label"])
    df = df[df["prompt_harm_label"].isin(["harmful", "unharmful"])].reset_index(drop=True)
    df = df.rename(columns={"response_refusal_label": "wildguard_reference_label"})
    if "response" in df.columns:
        df = df.drop(columns=["response"])

    n = min((df["prompt_harm_label"] == "harmful").sum(),
            (df["prompt_harm_label"] == "unharmful").sum())
    if args.max_samples:
        n = min(n, args.max_samples // 2)
    df_h = df[df["prompt_harm_label"] == "harmful"].sample(n, random_state=args.seed)
    df_u = df[df["prompt_harm_label"] == "unharmful"].sample(n, random_state=args.seed)
    df = pd.concat([df_h, df_u]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"Generating responses on {len(df)} prompts with {args.source_model}")

    df["response"] = generate_responses(args.source_model, df["prompt"].tolist(),
                                        batch_size=args.gen_batch, device=device)

    print("Scoring responses with WildGuard-7B")
    classifier = load_classifier()
    full = score_full(classifier, df["prompt"].tolist(), df["response"].tolist())
    df["predicted_refusal"] = [
        (str(r.get("response_refusal", "")).lower() == "refusal")
        if not r.get("is_parsing_error") else False
        for r in full
    ]
    df["predicted_prompt_harmfulness"] = [r.get("prompt_harmfulness") for r in full]
    df["predicted_response_harmfulness"] = [r.get("response_harmfulness") for r in full]
    df["wg_parse_error"] = [bool(r.get("is_parsing_error")) for r in full]

    if args.filter_train:
        agrees = ((df["prompt_harm_label"] == "harmful") & df["predicted_refusal"]) | \
                 ((df["prompt_harm_label"] == "unharmful") & ~df["predicted_refusal"])
        df = df[agrees].reset_index(drop=True)
        print(f"After --filter_train: {len(df)} rows")

    train, val = stratified_train_val_split(df, label_col="prompt_harm_label",
                                            train_frac=args.train_frac, seed=args.seed)
    train.to_parquet(os.path.join(args.out_dir, "train.parquet"))
    val.to_parquet(os.path.join(args.out_dir, "holdout.parquet"))
    print(f"Train: {len(train)} ({(train['prompt_harm_label']=='harmful').mean():.4f} harmful)")
    print(f"Val:   {len(val)} ({(val['prompt_harm_label']=='harmful').mean():.4f} harmful)")


if __name__ == "__main__":
    main()
