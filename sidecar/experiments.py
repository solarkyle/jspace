from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Iterable


_TOKEN_MARKERS = "\u0120\u2581"


def normalize_token_family(text: str) -> str:
    """Return a deterministic family key for tokenizer surface variants.

    This intentionally stays conservative. It merges case, leading-space, and
    Unicode compatibility variants, but does not use embeddings or a learned
    semantic clustering rule that could erase genuine overload.
    """
    value = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    value = value.lstrip(_TOKEN_MARKERS).strip().casefold()
    value = re.sub(r"\s+", " ", value)
    return value or "<blank>"


def aggregate_top_token_families(
    token_ids: Iterable[int],
    probabilities: Iterable[float],
    tokenizer: Any,
    *,
    remainder_mass: float = 0.0,
) -> dict[str, float]:
    families: dict[str, float] = {}
    for token_id, probability in zip(token_ids, probabilities):
        label = tokenizer.decode([int(token_id)])
        key = normalize_token_family(label)
        families[key] = families.get(key, 0.0) + float(probability)
    if remainder_mass > 0:
        families["<unobserved-tail>"] = float(remainder_mass)
    return families


def entropy_from_masses(masses: Iterable[float]) -> float:
    return -sum(float(p) * math.log(max(float(p), 1e-12)) for p in masses if p > 0)


_NAMES = [
    "Ada",
    "Bram",
    "Cleo",
    "Dax",
    "Enzo",
    "Faye",
    "Gita",
    "Hugo",
    "Iris",
    "Juno",
    "Kira",
    "Luca",
    "Mara",
    "Niko",
    "Orla",
    "Pia",
]
_VALUES = [
    "amber",
    "birch",
    "coral",
    "dune",
    "ember",
    "frost",
    "grove",
    "hazel",
    "ivory",
    "jade",
    "kelp",
    "lilac",
    "moss",
    "navy",
    "opal",
    "pearl",
]


def binding_prompt(binding_count: int, *, target_index: int | None = None) -> tuple[str, str, str]:
    """Build one deterministic associative-recall overload prompt."""
    count = max(1, min(int(binding_count), len(_NAMES)))
    if target_index is None:
        target_index = count - 1
    target_index = max(0, min(int(target_index), count - 1))
    pairs = [f"{_NAMES[i]} = {_VALUES[i]}" for i in range(count)]
    target_name = _NAMES[target_index]
    expected = _VALUES[target_index]
    prompt = (
        "Memorize these temporary bindings. They are arbitrary and only valid "
        "for this question.\n"
        + "; ".join(pairs)
        + f".\nReturn only the value bound to {target_name}."
    )
    return prompt, target_name, expected


def answer_matches_expected(text: str, expected: str) -> bool:
    """Score terse associative-recall output while tolerating exact repetition."""
    match = re.search(r"[\w-]+", text.casefold())
    if not match:
        return False
    word = match.group(0)
    target = expected.casefold()
    if word == target:
        return True
    if not target:
        return False
    return word == target * (len(word) // len(target)) and len(word) % len(target) == 0

