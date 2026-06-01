"""Package-level sanity checks: every module byte-compiles, and the
dependency-light modules import cleanly.

Modules that depend on the optional ``wildguard`` extra (the WildGuard wrapper
and the refusal data/eval/ablation entry points) are byte-compiled but not
imported here, so the suite runs without the heavy vLLM stack installed.
"""
import importlib
import py_compile
from pathlib import Path

import intervention_robust_refusal as pkg

PKG_ROOT = Path(pkg.__file__).parent

# Importable without the optional `wildguard` extra installed.
LIGHTWEIGHT_MODULES = [
    "intervention_robust_refusal",
    "intervention_robust_refusal.shared.hooks",
    "intervention_robust_refusal.shared.losses",
    "intervention_robust_refusal.shared.readouts",
    "intervention_robust_refusal.shared.probes",
    "intervention_robust_refusal.shared.erasure",
    "intervention_robust_refusal.sentiment.train_gpt2",
    "intervention_robust_refusal.sentiment.eval_sentiment",
    "intervention_robust_refusal.refusal.train_llama",
]


def test_all_source_files_byte_compile():
    files = sorted(PKG_ROOT.rglob("*.py"))
    assert files, "no source files discovered"
    for f in files:
        py_compile.compile(str(f), doraise=True)


def test_lightweight_modules_import():
    for name in LIGHTWEIGHT_MODULES:
        importlib.import_module(name)


def test_version_is_exposed():
    assert isinstance(pkg.__version__, str)
    assert pkg.__version__
