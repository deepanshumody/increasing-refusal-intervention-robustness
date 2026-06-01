"""WildGuard-7B refusal scoring via the official `wildguard` package (vLLM-backed)."""
from wildguard import load_wildguard


def load_classifier():
    return load_wildguard()


def score_refusals(classifier, prompts, responses):
    """Returns list of bool (True iff response is a refusal).

    Wraps the wildguard classify() interface, which expects a list of dicts
    {prompt, response} and returns dicts with `response_refusal`,
    `prompt_harmfulness`, `response_harmfulness` and `is_parsing_error`.
    """
    items = [{"prompt": p, "response": r} for p, r in zip(prompts, responses, strict=False)]
    results = classifier.classify(items)
    out = []
    for r in results:
        if r.get("is_parsing_error"):
            out.append(False)
            continue
        ref = r.get("response_refusal")
        out.append(str(ref).lower() == "refusal" or ref is True)
    return out


def score_full(classifier, prompts, responses):
    """Returns list of dicts with all WildGuard fields per item (no parsing)."""
    items = [{"prompt": p, "response": r} for p, r in zip(prompts, responses, strict=False)]
    return classifier.classify(items)
