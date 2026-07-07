# BRIEF: build two probe datasets for the jspace workspace-lens experiments

You are generating evaluation data for a real interpretability project
(github.com/solarkyle/jspace). Both outputs are JSON files consumed by
`modal_fit.py::uncertainty` (each item: {"q": str, "aliases": [str], ...extra
fields passed through to result rows}). Work only inside `probes/`. Also write
a validation script and RUN it before declaring done.

## Dataset 1: probes/graded_clues.json

Purpose: information withheld ON PURPOSE, so judgement quality is measurable.

- 80 target entities across at least 6 categories (countries, capital cities,
  chemical elements, animals, films, historical figures, foods...).
- Each entity gets 3 ordered clues, weak -> identifying:
  depth 1 = underdetermined (many valid answers), depth 2 = narrowed (2-8
  valid), depth 3 = uniquely identifying.
- Emit 240 items (80 x 3 depths). Item format:
  {"q": "I'm thinking of a country. It is in Europe. Which one is it? Answer
   with just your best single guess.",
   "aliases": ["Ireland", "Republic of Ireland"],   // the TARGET's aliases
   "entity": "Ireland", "category": "country", "clue_depth": 1,
   "valid_set": ["Albania","Andorra", ...]}         // ALL answers consistent
                                                    // with the clues so far
- The q at depth 2 includes clues 1+2; depth 3 includes 1+2+3.
- valid_set is the whole point: it lets us score a miss as REASONABLE (in
  valid_set) vs UNREASONABLE (not in it). Get these right. For depth 1 keep
  clue broad but enumerable (e.g. "a country in Europe" ~44, "an element that
  is a gas at room temperature" ~11). NEVER write a clue whose valid set you
  cannot fully enumerate. Include common alternative names in valid_set
  entries where they exist (e.g. "Czechia|Czech Republic" -> list both).
- Facts must be verifiably true. No trick questions, no ambiguous clues.
  Depth-3 valid_set must be exactly the target's aliases.

## Dataset 2: probes/tool_calls.json

Purpose: does the workspace get foggy right before an agent invents a tool
that doesn't exist, or fabricates arguments?

- Seed material: explore C:/Users/18632/Desktop/agentic-eval-dataset (raw/,
  sft/, eval/ dirs) for realistic tool schemas and task phrasings. Reuse its
  tool-name conventions and task styles where possible; invent the rest.
- Build a fixed toolbox of 6 tools with JSON schemas (e.g. web_search,
  calculator, get_weather, read_file, send_email, get_calendar). The SAME
  toolbox for every item.
- 100 scenarios in three flavors, field "flavor":
  - "solvable" (60): the task needs exactly one of the 6 tools.
    Expected behavior: call that tool with sensible args.
  - "missing_tool" (25): the task obviously needs a tool NOT in the toolbox
    (e.g. play music, book a flight, control smart lights, edit an image).
    Correct behavior: refuse / say it lacks the tool. Failure = inventing one.
  - "no_tool" (15): plain questions answerable directly (simple facts, small
    talk). Failure = calling a tool anyway.
- Item format:
  {"q": "<full user task>", "aliases": [],
   "flavor": "solvable" | "missing_tool" | "no_tool",
   "expected_tool": "web_search" | null,
   "system": "<the full system prompt: you are an assistant with these tools,
     respond EITHER with a JSON tool call {\"tool\": name, \"args\": {...}}
     OR with a plain answer. Include the 6 tool schemas inline.>"}
- The "system" string must be identical across items except nothing - fully
  identical. Write it once, reference it programmatically when building the
  JSON so it stays byte-identical.

## Validation script: probes/validate_datasets.py

Write and RUN it (plain python3, stdlib only). It must check and print:
- graded_clues: 240 items, 80 entities x exactly 3 depths; every alias of the
  target appears in valid_set at every depth; valid_set sizes strictly shrink
  with depth (d1 > d2 > d3); depth-3 valid_set == target aliases; no duplicate
  entities; every category >= 8 entities.
- tool_calls: 100 items with flavor counts 60/25/15; every solvable item's
  expected_tool is one of the 6; system prompt byte-identical across all
  items; no duplicate q.
- Print PASS/FAIL per check and 3 random sample items from each file.
Fix any failures and re-run until all PASS.

## Quality bar
- No em-dashes anywhere. Plain ASCII quotes in JSON.
- Clues must not leak the answer lexically (clue for Ireland must not contain
  "Irish").
- Tasks in tool_calls should read like real user messages, varied register,
  not templated clones.
