import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TOOL_NAMES = {"web_search", "calculator", "get_weather", "read_file", "send_email", "get_calendar"}


def load_json(name):
    with (ROOT / name).open("r", encoding="utf-8") as f:
        return json.load(f)


class Reporter:
    def __init__(self):
        self.failed = False

    def check(self, label, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        suffix = f" - {detail}" if detail else ""
        print(f"{status} {label}{suffix}")
        if not ok:
            self.failed = True


def validate_graded(items, reporter):
    reporter.check("graded_clues: 240 items", len(items) == 240, f"found {len(items)}")

    by_entity = defaultdict(list)
    for item in items:
        by_entity[item.get("entity")].append(item)

    reporter.check("graded_clues: 80 unique entities", len(by_entity) == 80, f"found {len(by_entity)}")

    entity_depths_ok = True
    shrink_ok = True
    aliases_in_valid_ok = True
    depth3_ok = True
    entity_category_ok = True
    for entity, rows in by_entity.items():
        depths = sorted(row.get("clue_depth") for row in rows)
        if depths != [1, 2, 3]:
            entity_depths_ok = False
        categories = {row.get("category") for row in rows}
        if len(categories) != 1:
            entity_category_ok = False
        by_depth = {row.get("clue_depth"): row for row in rows}
        if set(by_depth) == {1, 2, 3}:
            sizes = [len(by_depth[d].get("valid_set", [])) for d in [1, 2, 3]]
            if not (sizes[0] > sizes[1] > sizes[2]):
                shrink_ok = False
            aliases = by_depth[3].get("aliases", [])
            for depth in [1, 2, 3]:
                valid = by_depth[depth].get("valid_set", [])
                if not all(alias in valid for alias in aliases):
                    aliases_in_valid_ok = False
            if by_depth[3].get("valid_set", []) != aliases:
                depth3_ok = False

    reporter.check("graded_clues: each entity has exactly depths 1, 2, and 3", entity_depths_ok)
    reporter.check("graded_clues: entity category is stable across depths", entity_category_ok)
    reporter.check("graded_clues: every target alias appears in valid_set at every depth", aliases_in_valid_ok)
    reporter.check("graded_clues: valid_set sizes strictly shrink with depth", shrink_ok)
    reporter.check("graded_clues: depth-3 valid_set equals target aliases", depth3_ok)

    category_counts = Counter()
    for entity, rows in by_entity.items():
        if rows:
            category_counts[rows[0].get("category")] += 1
    categories_ok = bool(category_counts) and all(count >= 8 for count in category_counts.values())
    reporter.check("graded_clues: every category has at least 8 entities", categories_ok, dict(sorted(category_counts.items())))

    duplicate_entity_rows = len(items) != 3 * len(by_entity)
    reporter.check("graded_clues: no duplicate entities beyond the three required depths", not duplicate_entity_rows)

    no_emdash = "\u2014" not in json.dumps(items, ensure_ascii=False)
    reporter.check("graded_clues: no em dash characters", no_emdash)


def validate_tool_calls(items, reporter):
    reporter.check("tool_calls: 100 items", len(items) == 100, f"found {len(items)}")

    flavors = Counter(item.get("flavor") for item in items)
    reporter.check("tool_calls: flavor counts 60/25/15", flavors == {"solvable": 60, "missing_tool": 25, "no_tool": 15}, dict(flavors))

    solvable_tools_ok = all(
        item.get("expected_tool") in TOOL_NAMES
        for item in items
        if item.get("flavor") == "solvable"
    )
    reporter.check("tool_calls: every solvable expected_tool is in the toolbox", solvable_tools_ok)

    null_expected_ok = all(
        item.get("expected_tool") is None
        for item in items
        if item.get("flavor") in {"missing_tool", "no_tool"}
    )
    reporter.check("tool_calls: non-solvable expected_tool is null", null_expected_ok)

    systems = [item.get("system") for item in items]
    system_identical = len(set(systems)) == 1 if systems else False
    reporter.check("tool_calls: system prompt byte-identical across all items", system_identical)

    questions = [item.get("q") for item in items]
    no_duplicate_q = len(questions) == len(set(questions))
    reporter.check("tool_calls: no duplicate q", no_duplicate_q)

    no_emdash = "\u2014" not in json.dumps(items, ensure_ascii=False)
    reporter.check("tool_calls: no em dash characters", no_emdash)


def print_samples(name, items):
    print(f"\nSamples from {name}:")
    random.seed(20260707 + len(name))
    for sample in random.sample(items, 3):
        print(json.dumps(sample, ensure_ascii=True, sort_keys=True))


def main():
    reporter = Reporter()
    graded = load_json("graded_clues.json")
    tools = load_json("tool_calls.json")

    validate_graded(graded, reporter)
    validate_tool_calls(tools, reporter)
    print_samples("graded_clues.json", graded)
    print_samples("tool_calls.json", tools)

    if reporter.failed:
        print("\nOVERALL FAIL")
        return 1
    print("\nOVERALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
