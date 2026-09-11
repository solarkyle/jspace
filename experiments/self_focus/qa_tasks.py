"""Synthetic grounded QA items for the six-condition error-awareness study.

Every item is a short fictional passage plus a five-option question with exactly one
mechanically verified answer. Fictional entities throughout, so outside knowledge
cannot substitute for reading the passage.

Four item types:
    supported        the passage states the answer
    two_hop          the answer needs two stated facts combined
    missing          the passage omits the needed fact, so INSUFFICIENT is correct
    false_premise    the question presupposes something the passage contradicts

Splits are by FAMILY, fixed before any model output. A family is one fact template
with its counterfactual variants, and a family never spans the development and
evaluation splits. Development and evaluation draw from separate family seeds.

Grading is strict designated-field parsing. Substring matching is not used anywhere:
the v3 pilot used it for task correctness and the resulting figures were upper
bounds rather than measurements.
"""

from __future__ import annotations

from experiments.self_focus import core

OPTION_LETTERS = ["A", "B", "C", "D", "E"]

# Fictional entities. No real-world knowledge can help.
PLACES = ["Tarnwick", "Belmoor", "Caldry", "Withen", "Ospreyhold",
          "Marrowgate", "Lindhollow", "Quarrenmouth", "Stilbridge", "Ashenvale"]
PEOPLE = ["Orsa Venn", "Dalie Crowe", "Imre Tolst", "Nera Fitch", "Perrin Oake",
          "Sabel Mork", "Tovin Rask", "Ylla Brandt", "Corin Delph", "Maren Ives"]
GOODS = ["saltglass", "copperwort", "blackmeal", "tinbark", "greywool",
         "emberfruit", "stonemoss", "finchseed", "ruddleclay", "palewax"]
ROLES = ["harbourmaster", "archivist", "ferry clerk", "seed warden", "toll keeper"]


def _pick(seq, *key):
    """Deterministic selection. Stable hash, never Python's salted hash()."""
    digest = core.stable_id(*key)
    return seq[int(digest[:8], 16) % len(seq)]


def _shuffle(items, *key):
    order = sorted(range(len(items)), key=lambda i: core.stable_id(*key, i))
    return [items[i] for i in order]


def _family(seed: str, index: int) -> dict:
    """One fact family: the shared entities every variant in it reuses."""
    key = (seed, "family", index)
    place = _pick(PLACES, *key, "place")
    person = _pick(PEOPLE, *key, "person")
    good = _pick(GOODS, *key, "good")
    role = _pick(ROLES, *key, "role")
    year = 1800 + int(core.stable_id(*key, "year")[:4], 16) % 90
    quantity = 10 + int(core.stable_id(*key, "qty")[:4], 16) % 80
    second = 10 + int(core.stable_id(*key, "qty2")[:4], 16) % 80
    return {"family_id": f"{seed}-{index}", "place": place, "person": person,
            "good": good, "role": role, "year": year,
            "quantity": quantity, "second": second}


def _options(correct: str, distractors: list, *key) -> tuple:
    """Five options, correct answer in a position that varies by item."""
    pool = [correct] + [d for d in distractors if d != correct][:4]
    while len(pool) < 5:
        pool.append(f"none of these ({len(pool)})")
    shuffled = _shuffle(pool[:5], *key, "opts")
    options = list(zip(OPTION_LETTERS, shuffled))
    letter = next(k for k, v in options if v == correct)
    return options, letter


def _items_for_family(fam: dict, seed: str) -> list:
    """One item of each type per family, plus one counterfactual variant."""
    p, who, good = fam["place"], fam["person"], fam["good"]
    role, year = fam["role"], fam["year"]
    qty, second = fam["quantity"], fam["second"]
    items = []

    # supported
    passage = (f"The {role} of {p} was {who}. In {year} the town recorded {qty} crates "
               f"of {good} passing through its gate.")
    opts, letter = _options(who, PEOPLE, fam["family_id"], "supported")
    items.append({"item_type": "supported", "passage": passage,
                  "question": f"Who was the {role} of {p}?",
                  "options": opts, "correct_letter": letter,
                  "correct_value": who, "answerable": True})

    # counterfactual variant of the same family: a different person, same shape
    other = _pick([x for x in PEOPLE if x != who], fam["family_id"], "alt")
    passage_cf = (f"The {role} of {p} was {other}. In {year} the town recorded {qty} "
                  f"crates of {good} passing through its gate.")
    opts, letter = _options(other, PEOPLE, fam["family_id"], "supported_cf")
    items.append({"item_type": "supported", "passage": passage_cf,
                  "question": f"Who was the {role} of {p}?",
                  "options": opts, "correct_letter": letter,
                  "correct_value": other, "answerable": True,
                  "counterfactual_of": "supported"})

    # two hop: sum of two stated quantities
    total = qty + second
    passage = (f"In {year}, {p} received {qty} crates of {good} by river and {second} "
               f"crates by road. No other shipments arrived that year.")
    distract = [str(qty), str(second), str(total + 5), str(abs(qty - second))]
    opts, letter = _options(str(total), distract, fam["family_id"], "two_hop")
    items.append({"item_type": "two_hop", "passage": passage,
                  "question": f"How many crates of {good} did {p} receive in {year} in total?",
                  "options": opts, "correct_letter": letter,
                  "correct_value": str(total), "answerable": True})

    # missing evidence: the passage never gives the asked quantity
    passage = (f"The {role} of {p} was {who}. The town's {good} trade was recorded in "
               f"a ledger that has since been lost.")
    opts, letter = _options("INSUFFICIENT", [str(qty), str(second), "none", "zero"],
                            fam["family_id"], "missing")
    items.append({"item_type": "missing", "passage": passage,
                  "question": f"How many crates of {good} passed through {p} in {year}?",
                  "options": opts, "correct_letter": letter,
                  "correct_value": "INSUFFICIENT", "answerable": False})

    # false premise: the question presupposes a contradicted fact
    passage = (f"{p} has never had a harbour; all goods arrive overland. {who} served "
               f"as its {role} in {year}.")
    opts, letter = _options("FALSE_PREMISE", [who, p, str(qty), "INSUFFICIENT"],
                            fam["family_id"], "false_premise")
    items.append({"item_type": "false_premise", "passage": passage,
                  "question": f"In which year was the harbour of {p} rebuilt?",
                  "options": opts, "correct_letter": letter,
                  "correct_value": "FALSE_PREMISE", "answerable": False})

    for item in items:
        item["family_id"] = fam["family_id"]
        item["split_seed"] = seed
    return items


def build(n_dev_families: int = 4, n_eval_families: int = 16,
          dev_seed: str = "devA", eval_seed: str = "evalB") -> dict:
    """Development and evaluation items, split by family before any model output."""
    dev, ev = [], []
    for i in range(n_dev_families):
        dev += _items_for_family(_family(dev_seed, i), dev_seed)
    for i in range(n_eval_families):
        ev += _items_for_family(_family(eval_seed, i), eval_seed)
    for idx, item in enumerate(dev):
        item["item_id"] = f"dev-{idx:03d}"
        item["split"] = "dev"
    for idx, item in enumerate(ev):
        item["item_id"] = f"eval-{idx:03d}"
        item["split"] = "eval"
    return {"dev": dev, "eval": ev}


# --------------------------------------------------------------------------- #
# prompting and strict grading
# --------------------------------------------------------------------------- #

QA_SCHEMA = "CHOICE=<A|B|C|D|E>; CONFIDENCE=<integer 0..100>"


def qa_task_block(item: dict) -> str:
    listed = "\n".join(f"{k}) {v}" for k, v in item["options"])
    return (
        f"Passage:\n{item['passage']}\n"
        f"\n"
        f"Question: {item['question']}\n"
        f"{listed}\n"
        f"\n"
        f"Answer using only the passage. If the passage does not contain the needed "
        f"information, choose the INSUFFICIENT option. If the question assumes "
        f"something the passage contradicts, choose the FALSE_PREMISE option.\n"
        f"Respond with exactly one line and nothing else:\n{QA_SCHEMA}"
    )


def parse_qa(text: str) -> dict:
    """Strict designated-field parsing. No substring matching anywhere."""
    out = {"valid": False, "choice": None, "confidence": None, "reason": None}
    line = None
    for candidate in (text or "").splitlines():
        if "CHOICE=" in candidate:
            line = candidate.strip()
            break
    if line is None:
        out["reason"] = "no line containing CHOICE="
        return out
    fields = {}
    for chunk in line.split(";"):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        fields[key.strip().upper()] = value.strip()
    raw = fields.get("CHOICE", "").strip().strip(".").upper()
    if raw not in OPTION_LETTERS:
        out["reason"] = f"CHOICE={raw!r} is not one of {OPTION_LETTERS}"
        return out
    out["choice"] = raw
    conf = fields.get("CONFIDENCE", "").strip().strip(".").rstrip("%")
    try:
        value = int(conf)
    except (TypeError, ValueError):
        out["reason"] = f"CONFIDENCE={conf!r} is not an integer"
        return out
    if not 0 <= value <= 100:
        out["reason"] = f"CONFIDENCE={value} outside 0..100"
        return out
    out["confidence"] = value
    out["valid"] = True
    return out
