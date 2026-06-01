"""Increasing the intervention-robustness of refusal in open-weight LLMs.

A training-time objective that combines class-conditional mean/covariance
matching with KL distillation against a frozen Instruct teacher. It
redistributes the refusal signal across many residual-stream directions so
that low-rank linear ablation (Arditi et al., 2024) no longer disables refusal
in one shot — raising the rank of the required linear attack from 1 to at least
16 while preserving the original Instruct model's refusal behaviour.
"""

__version__ = "0.1.0"
