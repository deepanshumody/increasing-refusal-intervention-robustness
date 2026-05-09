# Paper code: Increasing the Intervention-Robustness of Refusal

Minimal from-scratch code that runs every experiment in icml2026.pdf, with
implementation choices matched to the existing pipeline:

- Hooks: pre-hook on layer input + self_attn output + mlp output (Arditi).
- Mean penalty default = L2 (mean of (Δμ)²); cov penalty default = L2
  (Frobenius² / H²).
- LM loss is over the entire input. For refusal training the input is
  prompt-only (chat template, `add_generation_prompt=True`) — no response
  tokens.
- Pool: `mixed_batch_readout` (per-sample random assignment to last / random /
  chat-template positions; remainder mean-pool).
- KD default T=1.0, KL(teacher‖student) at the last non-pad position.
- Probes: sklearn StandardScaler+LogReg(liblinear, balanced) and
  MLPClassifier(256,128, early_stopping). Probe layer auto =
  `int(0.65 * num_hidden_layers)`.
- Iterated ablation: drop final 20% of layers, scan `-5..-1` positions, score
  with refusal-log-odds on harmful + KL(base‖abl) on harmless +
  induce-refusal log-odds via raw-delta pre-hook (coeff 1.0), choose
  argmin(refusal) among survivors of (`KL ≤ 0.1`, `steering ≥ 0`); fallback
  argmin(refusal + 1·KL); stop on norm-collapse below 1% of `d1_norm`.
- WildGuard scoring uses the `wildguard` pip package (vLLM-backed).

Run from the repo root so `paper_code` is on the path.

## Sentiment proof-of-concept (Section 4.1)

```bash
# Baseline
python -m paper_code.sentiment.train_gpt2 --match none --out_dir ckpt/gpt2_baseline

# Mean L2 (paper headline λ=100, multi-layer, mixed readout)
python -m paper_code.sentiment.train_gpt2 \
    --match mean --mean_penalty_type l2 --lambda_mean 100 \
    --multi_layer 1 \
    --last_token_ratio 0.3333 --random_pool_ratio 0.3333 \
    --out_dir ckpt/gpt2_mean_l2_lam100

# Mean L1 (paper "L1" condition)
python -m paper_code.sentiment.train_gpt2 \
    --match mean --mean_penalty_type l1 --lambda_mean 100 \
    --multi_layer 1 \
    --last_token_ratio 0.3333 --random_pool_ratio 0.3333 \
    --out_dir ckpt/gpt2_mean_l1_lam100

# Cov L2 (Frobenius² / H²)
python -m paper_code.sentiment.train_gpt2 \
    --match cov --cov_penalty_type l2 --lambda_cov 100 \
    --multi_layer 1 \
    --last_token_ratio 0.3333 --random_pool_ratio 0.3333 \
    --out_dir ckpt/gpt2_cov_l2_lam100

# Multi-layer-off variant
python -m paper_code.sentiment.train_gpt2 \
    --match mean --mean_penalty_type l1 --lambda_mean 100 \
    --multi_layer 0 \
    --last_token_ratio 0.3333 --random_pool_ratio 0.3333 \
    --out_dir ckpt/gpt2_mean_l1_singlelayer
```

λ-sweep `{10, 50, 100, 500}` is the same command with `--lambda_mean` /
`--lambda_cov` varied.

Evaluate (perplexity + mean gap + linear/MLP probe):

```bash
python -m paper_code.sentiment.eval_sentiment --model ckpt/gpt2_mean_l2_lam100

# Baseline + LEACE + INLP comparison (paper: 0.79, 0.57, 0.62)
python -m paper_code.sentiment.eval_sentiment --model gpt2 --baseline gpt2
```

## Refusal main results (Sections 4.2, 4.3)

### 1. Build the dataset

```bash
# Without filter (default, matches existing default)
python -m paper_code.refusal.data_pipeline \
    --source_model meta-llama/Llama-3.2-1B-Instruct \
    --out_dir data/refusal

# With agreement filter (drop rows where predicted_refusal disagrees with prompt_harm_label)
python -m paper_code.refusal.data_pipeline \
    --source_model meta-llama/Llama-3.2-1B-Instruct \
    --out_dir data/refusal --filter_train
```

Produces `train.parquet` and `holdout.parquet` (80/20 stratified split, seed 42).
Each row carries `prompt`, `prompt_harm_label`, `wildguard_reference_label`,
`response`, `predicted_refusal`.

### 2. Train the four refusal conditions

```bash
# L1 (mean matching with L1 norm, sentence-pool, no KD)
python -m paper_code.refusal.train_llama \
    --data_dir data/refusal \
    --match mean --mean_penalty_type l1 --lambda_mean 100 \
    --multi_layer 1 \
    --last_token_ratio 0.3333 --random_pool_ratio 0.3333 \
    --out_dir ckpt/llama_l1

# Cov L2 (covariance matching with Frobenius, sentence-pool, no KD)
python -m paper_code.refusal.train_llama \
    --data_dir data/refusal \
    --match cov --cov_penalty_type l2 --lambda_cov 100 \
    --multi_layer 1 \
    --last_token_ratio 0.3333 --random_pool_ratio 0.3333 \
    --out_dir ckpt/llama_cov

# L1 + KD (chat-template pool, KD on)
python -m paper_code.refusal.train_llama \
    --data_dir data/refusal \
    --match mean --mean_penalty_type l1 --lambda_mean 100 \
    --multi_layer 1 \
    --chat_template_pool_ratio 1.0 \
    --kd_lambda 1.0 --kd_T 1.0 \
    --out_dir ckpt/llama_l1_kd

# Cov L2 + KD
python -m paper_code.refusal.train_llama \
    --data_dir data/refusal \
    --match cov --cov_penalty_type l2 --lambda_cov 100 \
    --multi_layer 1 \
    --chat_template_pool_ratio 1.0 \
    --kd_lambda 1.0 --kd_T 1.0 \
    --out_dir ckpt/llama_cov_kd
```

Hyperparameter defaults: `lr=2e-5`, `weight_decay=0.01`, `micro_batch=2 ×
grad_accum=64`, `epochs=20`, bf16, multi-layer matching across every block.

### 3. K=0 evaluation (Section 4.2)

```bash
for M in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-1B \
         ckpt/llama_l1 ckpt/llama_cov ckpt/llama_l1_kd ckpt/llama_cov_kd; do
  python -m paper_code.refusal.eval_refusal \
      --model "$M" --data_dir data/refusal \
      --out_json out/k0_$(basename "$M").json
done
```

Probe layer defaults to `int(0.65 * num_hidden_layers)`.

### 4. Iterated single-direction ablation (Section 4.3)

```bash
for M in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-1B \
         ckpt/llama_l1_kd ckpt/llama_cov_kd; do
  python -m paper_code.refusal.iterated_ablation \
      --model "$M" --data_dir data/refusal --K_max 16 \
      --out_json out/ablate_$(basename "$M").json
done
```

Defaults match the existing pipeline: `prune_layer_percentage=0.2`,
`scan_positions=-5,-4,-3,-2,-1`, `kl_threshold=0.1`, `kl_alpha=1.0`,
`induce_refusal_threshold=0.0`, `steering_coeff=1.0`,
`norm_threshold=0.01`, refusal-token set = `{ "I" }`.


