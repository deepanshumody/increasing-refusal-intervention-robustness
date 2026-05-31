# Increasing the Intervention-Robustness of Refusal in Open-Weight LLMs

Code for the ICML 2025 submission *Increasing the Intervention-Robustness of
Refusal in Open-Weight Language Models*.

**Authors.** Deepanshu Mody, Acey Vogelstein, Jonathan Merchan.
All three contributed as students at New York University — Deepanshu Mody
(M.S. Data Science), Acey Vogelstein (M.S. Data Science), Jonathan Merchan
(Ph.D. Linguistics). Correspondence: `deepanshumody26@gmail.com`.

## What this is about

[Arditi et al. (2024)](https://arxiv.org/abs/2406.11717) showed that refusal in
instruction-tuned LLMs is mediated by a *single* residual-stream direction:
project that one direction out at every layer and refusal collapses across a
wide range of harmful prompts, with general capability essentially untouched.
For open-weight models this is a vulnerability — anyone with the weights can
disable safety with a one-line intervention, so heavy investment in
post-training alignment is largely wasted.

This project asks a constructive question: can a **training-time** objective
*redistribute* the refusal signal across many directions so that low-rank
linear ablation no longer works? Refusal stops being a single auditable
feature and becomes a higher-rank subspace that an attacker has to find and
suppress in full.

Concretely, we add two regularisers to the standard causal-LM loss during
fine-tuning of `meta-llama/Llama-3.2-1B-Instruct` on WildGuardMix:

| Term | What it does | Why |
| --- | --- | --- |
| `L_match` — mean (L1) or covariance (L2 Frobenius) matching of class-conditional pooled hidden states at every layer | Equalises the first- or first+second-order statistics of `harmful` vs. `harmless` representations | Belrose et al. (2023) Thm 3.1: equal class-conditional means ⇒ no linear probe beats chance on the population. Covariance matching pushes the redistribution past first-order. |
| `L_ref` — temperature-scaled KL distillation from a frozen Instruct teacher at the last input position | Locks the next-token distribution to the original Instruct model's | Without it, the matching loss "cheats" by destroying surface refusal (saying *yes* to harmful prompts) instead of reshaping the underlying geometry |

The **headline result**: training with `Cov L2 + KD` survives **≥16 iterations
of Arditi's mean-difference ablation** without dropping below the 30% harmful-
refusal suppression threshold, vs. **K=1** for unmodified Instruct — at least
a 16× increase in the rank of the linear attack required to disable refusal,
while preserving the K=0 refusal behaviour of the original Instruct model.

## Repository layout

```
src/intervention_robust_refusal/
├── shared/
│   ├── hooks.py        # ResidualCapture, Arditi-style 3-site ablation hooks
│   ├── losses.py       # mean / cov matching penalties + KD
│   ├── readouts.py     # pooling strategies + mixed_batch_readout
│   ├── probes.py       # sklearn linear / MLP probes
│   ├── erasure.py      # LEACE + INLP (post-hoc baselines, sentiment only)
│   └── wildguard.py    # thin wrapper over the wildguard package
├── sentiment/
│   ├── train_gpt2.py     # GPT-2 + matching loss on IMDB (proof of concept)
│   └── eval_sentiment.py # perplexity, mean gap, probe accuracy, LEACE/INLP
└── refusal/
    ├── data_pipeline.py     # WildGuardMix → train/holdout parquet
    ├── train_llama.py       # Llama-3.2-1B-Instruct + matching + KD
    ├── eval_refusal.py      # K=0 generation + WildGuard + probes
    └── iterated_ablation.py # Arditi-style K≥1 ablation attack curve
```

## Install

```bash
pip install -e .

# WildGuard pulls a heavy vLLM stack; only needed for refusal eval/data prep.
pip install -e ".[wildguard]"
```

Tested on Python 3.10+, PyTorch 2.1+, single A100 (refusal training) /
single L4 (sentiment).

## Sentiment proof-of-concept

A lightweight sanity check: same losses, smaller model (GPT-2, 124M),
unambiguous labels (IMDB pos/neg). Establishes that mean matching closes
the empirical mean gap by >90% but still leaves probes well above chance,
which motivates the additional covariance term and the chat-token pool used
in the refusal experiments.

```bash
# Baseline (no matching loss)
python -m intervention_robust_refusal.sentiment.train_gpt2 --match none --out_dir ckpt/gpt2_baseline

# Mean L2, multi-layer, mixed readout (headline configuration at λ=100)
python -m intervention_robust_refusal.sentiment.train_gpt2 \
    --match mean --mean_penalty_type l2 --lambda_mean 100 \
    --multi_layer 1 \
    --last_token_ratio 0.3333 --random_pool_ratio 0.3333 \
    --out_dir ckpt/gpt2_mean_l2_lam100

# Mean L1
python -m intervention_robust_refusal.sentiment.train_gpt2 \
    --match mean --mean_penalty_type l1 --lambda_mean 100 \
    --multi_layer 1 \
    --last_token_ratio 0.3333 --random_pool_ratio 0.3333 \
    --out_dir ckpt/gpt2_mean_l1_lam100

# Cov L2 (Frobenius² / H²)
python -m intervention_robust_refusal.sentiment.train_gpt2 \
    --match cov --cov_penalty_type l2 --lambda_cov 100 \
    --multi_layer 1 \
    --last_token_ratio 0.3333 --random_pool_ratio 0.3333 \
    --out_dir ckpt/gpt2_cov_l2_lam100
```

The full λ-sweep `{10, 50, 100, 500}` is the same command with `--lambda_*`
varied. Evaluate (perplexity + mean gap + linear/MLP probe):

```bash
python -m intervention_robust_refusal.sentiment.eval_sentiment --model ckpt/gpt2_mean_l2_lam100

# Compare against post-hoc LEACE / INLP on frozen baseline embeddings
# (reference numbers: linear probe 0.79 → LEACE 0.57, INLP 0.62)
python -m intervention_robust_refusal.sentiment.eval_sentiment --model gpt2 --baseline gpt2
```

## Refusal main results

### 1. Build the dataset

Filters WildGuardMix to non-adversarial rows with a valid `prompt_harm_label`,
generates fresh responses with the source model, scores each with
WildGuard-7B, balances harmful/harmless, and writes a stratified 80/20 split.

```bash
# Default — keep all rows even where the source model's behaviour disagrees
# with prompt_harm_label
python -m intervention_robust_refusal.refusal.data_pipeline \
    --source_model meta-llama/Llama-3.2-1B-Instruct \
    --out_dir data/refusal

# Filtered setting: drop rows where the source model's behaviour disagrees
# with the gold prompt label (used for the main training runs).
python -m intervention_robust_refusal.refusal.data_pipeline \
    --source_model meta-llama/Llama-3.2-1B-Instruct \
    --out_dir data/refusal --filter_train
```

Produces `train.parquet` and `holdout.parquet`. Each row carries `prompt`,
`prompt_harm_label` (gold), `wildguard_reference_label` (WildGuardMix's own
response-refusal label), `response`, and `predicted_refusal` from
WildGuard-7B.

### 2. Train the four refusal conditions

```bash
# L1 — mean matching, sentence-pool (mean over user-message tokens), no KD
python -m intervention_robust_refusal.refusal.train_llama \
    --data_dir data/refusal \
    --match mean --mean_penalty_type l1 --lambda_mean 100 \
    --multi_layer 1 \
    --last_token_ratio 0.3333 --random_pool_ratio 0.3333 \
    --out_dir ckpt/llama_l1

# Cov L2 — covariance matching, sentence-pool, no KD
python -m intervention_robust_refusal.refusal.train_llama \
    --data_dir data/refusal \
    --match cov --cov_penalty_type l2 --lambda_cov 100 \
    --multi_layer 1 \
    --last_token_ratio 0.3333 --random_pool_ratio 0.3333 \
    --out_dir ckpt/llama_cov

# L1 + KD — chat-template pool, KD on
python -m intervention_robust_refusal.refusal.train_llama \
    --data_dir data/refusal \
    --match mean --mean_penalty_type l1 --lambda_mean 100 \
    --multi_layer 1 \
    --chat_template_pool_ratio 1.0 \
    --kd_lambda 1.0 --kd_T 1.0 \
    --out_dir ckpt/llama_l1_kd

# Cov L2 + KD — headline configuration
python -m intervention_robust_refusal.refusal.train_llama \
    --data_dir data/refusal \
    --match cov --cov_penalty_type l2 --lambda_cov 100 \
    --multi_layer 1 \
    --chat_template_pool_ratio 1.0 \
    --kd_lambda 1.0 --kd_T 1.0 \
    --out_dir ckpt/llama_cov_kd
```

Defaults: `lr=2e-5`, `weight_decay=0.01`, `micro_batch=2 × grad_accum=64`
(effective batch 128), 20 epochs, bf16, multi-layer matching across every
transformer block.

### 3. K=0 behavioural evaluation

Greedy generation on the holdout + WildGuard refusal-rate, plus linear and
MLP probes on the chat-template last-token hidden states at the auto-
selected probe layer (`int(0.65 · num_hidden_layers)`).

```bash
for M in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-1B \
         ckpt/llama_l1 ckpt/llama_cov ckpt/llama_l1_kd ckpt/llama_cov_kd; do
  python -m intervention_robust_refusal.refusal.eval_refusal \
      --model "$M" --data_dir data/refusal \
      --out_json out/k0_$(basename "$M").json
done
```

### 4. Iterated single-direction ablation

The attacker's loop, extending Arditi's K=1 protocol to K≥1 by re-extracting
mean-difference candidates *after* each ablated direction is installed.

```bash
for M in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-1B \
         ckpt/llama_l1_kd ckpt/llama_cov_kd; do
  python -m intervention_robust_refusal.refusal.iterated_ablation \
      --model "$M" --data_dir data/refusal --K_max 16 \
      --out_json out/ablate_$(basename "$M").json
done
```

Defaults: `prune_layer_percentage=0.2` (drop final 20% of layers from
candidates), `scan_positions=-5,-4,-3,-2,-1`, `kl_threshold=0.1`,
`kl_alpha=1.0` (fallback weight), `induce_refusal_threshold=0.0`,
`steering_coeff=1.0`, `norm_threshold=0.01`, refusal-token set = `{"I"}`
(Arditi's Llama-3 choice).

---

## Design decisions

Brief notes on the non-obvious choices behind the code.

### Why a training-time matching objective, not post-hoc LEACE?

LEACE *erases* a concept from frozen representations: applied to refusal it
would destroy the model's ability to distinguish harmful from harmless
prompts entirely. We want the opposite — preserve refusal behaviour while
spreading the signal so it's hard to localise. A post-hoc projection is also
patchable by anyone with weights, defeating the threat model. The
training-time loss bakes the redistribution into the parameters themselves.
LEACE appears only as an *upper bound on linear erasure* in the sentiment
baselines.

### Why the three-site Arditi hook (input + attn + mlp), not just the block input?

If we only project the residual stream at the block input, attention and MLP
can write the direction *back* into the stream within the same layer and the
ablation is undone. Hooking the two submodule outputs as well guarantees the
direction is removed at every write site. This is the original Arditi
formulation and the basis our defense has to actually defeat
(`shared/hooks.py:ablation_context`).

### Why pool with `mixed_batch_readout`?

The matching loss only equalises statistics at the positions we pool over.
A model can satisfy a loss applied at, say, the last token by *just* moving
the last-token vector and leaving the rest of the sequence untouched — and
the attacker happily extracts the refusal direction from one of the other
positions instead. Randomising the pool per sample (last-token / random
token subset / chat-template positions / uniform mean) forces the model to
reshape representations broadly, not at a single locus.

In the refusal experiments, **alignment of the pool with the position the
attacker uses matters** — sentence-pool matching collapses K=0 behaviour
while leaving the chat-template position linearly classifiable, whereas
chat-template pool matching closes that exact loophole
(`shared/readouts.py:mixed_batch_readout`).

### Why the LM loss runs over the entire prompt with `add_generation_prompt=True`?

In the refusal pipeline the input is **prompt only** (chat template applied,
no response) — there's no assistant turn to predict. Running the LM loss over
all input positions (HuggingFace's default with `labels=input_ids`) keeps the
prompt distribution stable while the matching loss reshapes the hidden
states, and avoids tying the regularisation strength to a single token's
loss (`refusal/train_llama.py:collate`).

### Why KD only at the last input position?

The refusal decision in instruction-tuned LLMs is dominated by the
distribution at the first generated token. KL'ing the full sequence to the
teacher would over-constrain the student and conflict with the matching
loss; KL'ing only at the last input position pins down exactly the
behaviour we want to preserve (refuse-vs-comply at the moment of
generation) and leaves the rest of the hidden states free to reorganise
(`refusal/train_llama.py`).

### Why `T=1.0` for KD?

Higher temperatures (Hinton et al. 2015) help when distilling a large
teacher into a small student. Here the teacher and student have *identical*
architectures and the student is a fine-tune of the teacher, so the
distributions are already on a comparable scale and softening is
unnecessary (`shared/losses.py:kd_kl_loss`).

### Why `int(0.65 · num_hidden_layers)` as the default probe layer?

Refusal information in instruction-tuned Llama emerges late but plateaus
before the final layers. 0.65× depth is in that plateau and matches where
the iterated-ablation search ends up picking directions
(`refusal/eval_refusal.py`).

### Why the iterated ablation uses three scores: refusal, KL, steering?

A naive K-greedy attack — "pick whichever direction most suppresses refusal
on harmful prompts" — would happily nuke the harmless distribution too. The
scoring (extending Arditi) filters candidates with three signals:

  - **refusal log-odds** on harmful prompts (lower = better suppression)
  - **KL(baseline ‖ ablated)** on harmless prompts (must stay below
    `kl_threshold` — the ablation can't break general behaviour)
  - **induce-refusal log-odds** when the *raw* mean-diff is added at the
    source layer with `coeff=1.0` (must be ≥ 0 — sanity check that the
    direction actually carries refusal both ways)

We pick the survivor with the lowest harmful refusal, with a soft fallback
`argmin(refusal + kl_alpha · kl)` if no candidate satisfies all three
(`refusal/iterated_ablation.py:run_iterated`).

### Why drop the final 20% of layers from ablation candidates?

The very last layers carry information that's already half-decoded into
token probabilities; their mean-difference vectors are dominated by output-
head structure rather than the abstract refusal feature, and ablating there
tends to score high on KL without actually suppressing behaviour. 0.2 is
Arditi's published default and we keep it for comparability.

### Why QR-orthonormalize the direction set before each ablation?

After the first direction is installed, the residual mean shifts, and the
raw mean-diffs at subsequent iterations are no longer orthogonal to what's
already ablated. The projector `a - (a · D^T) · D` is only idempotent when
`D`'s rows are mutually orthonormal — without QR, the second direction would
partially undo the first. We re-orthonormalise the whole stack each
iteration (`shared/hooks.py:prepare_directions`).

### Why H² normalisation on the covariance penalty?

Without it, the magnitude of the covariance term scales as O(H²) in the
hidden size H, while the mean penalty scales as O(1). Normalising by H²
makes `lambda_mean` and `lambda_cov` directly comparable and keeps the
sweep `{10, 50, 100, 500}` meaningful across both
(`shared/losses.py:compute_cov_penalty`).

## Citation

If you build on this code, please cite the paper:

```bibtex
@misc{mody2025refusal,
  title  = {Increasing the Intervention-Robustness of Refusal in Open-Weight Language Models},
  author = {Mody, Deepanshu and Vogelstein, Acey and Merchan, Jonathan},
  year   = {2025},
  note   = {Under review, ICML 2025}
}
```

See [`CITATION.cff`](./CITATION.cff) for the machine-readable form (renders as
a "Cite this repository" button on GitHub).

## License

Released under the [Apache License 2.0](./LICENSE).

## References

- Arditi et al. 2024. *Refusal in language models is mediated by a single direction.* NeurIPS.
- Belrose, Schneider-Joseph et al. 2023. *LEACE: Perfect linear concept erasure in closed form.* NeurIPS.
- Ravfogel et al. 2020. *Null it out: Guarding protected attributes by iterative nullspace projection.* ACL.
- Hinton, Vinyals, Dean. 2015. *Distilling the knowledge in a neural network.* arXiv:1503.02531.
- Han et al. 2024. *WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of large language models.* NeurIPS D&B.
- Allen Institute for AI. 2024. *WildGuardMix.* HuggingFace dataset `allenai/wildguardmix`.
