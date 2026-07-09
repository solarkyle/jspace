from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

ANSI_RESET = "\x1b[0m"
ANSI_RED = "\x1b[31m"
ANSI_GREEN = "\x1b[32m"

ANSI_RAMP = [
    17,
    18,
    19,
    20,
    21,
    27,
    33,
    39,
    45,
    51,
    87,
    123,
    159,
    195,
    231,
]


def print_snapshot(snapshot: Any, threshold: float = 0.6) -> None:
    """Print an ANSI heatmap for a Snapshot."""
    print(render_ansi(snapshot, threshold=threshold))


def render_ansi(snapshot: Any, threshold: float = 0.6) -> str:
    grid = snapshot.grid
    layers = list(grid.get("layers") or [])
    columns = list(grid.get("columns") or [])
    values = list(grid.get("values") or [])
    answer_col = int(grid.get("answer_col", -1))
    order = list(reversed(range(len(layers))))

    lines = []
    if columns:
        lines.append("      " + " ".join(f"{idx:02d}" for idx in range(len(columns))))
    for row_idx in order:
        layer = layers[row_idx]
        row = values[row_idx] if row_idx < len(values) else []
        cells = [_ansi_cell(float(row[col]) if col < len(row) else 0.0) for col in range(len(columns))]
        lines.append(f"L{int(layer):03d}  " + " ".join(cells))

    if columns and answer_col >= 0:
        lines.append("      " + " ".join(" ^" if idx == answer_col else "  " for idx in range(len(columns))))
    elif columns:
        lines.append("      answer column not in top candidates")

    if columns:
        lines.append("")
        lines.append("legend:")
        for idx, token in enumerate(columns):
            lines.append(f"  {idx:02d}: {_legend_token(token)}")

    noisy = float(snapshot.noise) >= threshold
    color = ANSI_RED if noisy else ANSI_GREEN
    tag = "NOISY" if noisy else "clean"
    lines.append("")
    lines.append(f"noise={float(snapshot.noise):.3f} {color}{tag}{ANSI_RESET}")
    return "\n".join(lines)


def write_html(snapshot: Any, path: str | Path, threshold: float = 0.6) -> Path:
    """Write a standalone HTML heatmap for a Snapshot."""
    out = Path(path)
    out.write_text(render_html(snapshot, threshold=threshold), encoding="utf-8")
    return out


def render_html(snapshot: Any, threshold: float = 0.6) -> str:
    grid = snapshot.grid
    layers = list(grid.get("layers") or [])
    columns = list(grid.get("columns") or [])
    values = list(grid.get("values") or [])
    answer_col = int(grid.get("answer_col", -1))
    order = list(reversed(range(len(layers))))
    noisy = float(snapshot.noise) >= threshold
    tag = "NOISY" if noisy else "clean"
    tag_class = "noisy" if noisy else "clean"

    rows = []
    header = "".join(f"<th>{idx:02d}</th>" for idx in range(len(columns)))
    rows.append(f"<tr><th>layer</th>{header}</tr>")
    for row_idx in order:
        layer = layers[row_idx]
        row = values[row_idx] if row_idx < len(values) else []
        cells = []
        for col in range(len(columns)):
            prob = float(row[col]) if col < len(row) else 0.0
            mark = " answer" if col == answer_col else ""
            cells.append(
                f'<td class="cell{mark}" title="{prob:.6f}" '
                f'style="background:{_html_color(prob)}"></td>'
            )
        rows.append(f"<tr><th>L{int(layer):03d}</th>{''.join(cells)}</tr>")

    legend = "".join(
        f"<li><code>{idx:02d}</code> {html.escape(_legend_token(token))}</li>"
        for idx, token in enumerate(columns)
    )
    answer_note = (
        f"<p>answer column: <code>{answer_col:02d}</code></p>"
        if answer_col >= 0
        else "<p>answer column not in top candidates</p>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jspace snapshot</title>
<style>
body {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  margin: 32px;
  color: #111827;
  background: #f8fafc;
}}
h1 {{ font-size: 20px; margin: 0 0 12px; }}
.meta {{ margin: 0 0 20px; color: #374151; }}
.tag {{ font-weight: 700; }}
.tag.clean {{ color: #15803d; }}
.tag.noisy {{ color: #b91c1c; }}
table {{
  border-collapse: collapse;
  background: white;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.14);
}}
th {{
  padding: 5px 7px;
  font-size: 12px;
  color: #334155;
  text-align: right;
}}
td.cell {{
  width: 26px;
  height: 20px;
  border: 1px solid rgba(15, 23, 42, 0.12);
}}
td.answer {{
  outline: 2px solid #ef4444;
  outline-offset: -2px;
}}
.legend {{
  columns: 2;
  max-width: 760px;
  padding-left: 0;
  list-style: none;
}}
.legend li {{ margin: 4px 0; }}
code {{ color: #0f172a; }}
</style>
</head>
<body>
<h1>jspace snapshot</h1>
<p class="meta">model: {html.escape(str(snapshot.model_id))} | quant: {html.escape(str(snapshot.quant))} | noise: {float(snapshot.noise):.3f} <span class="tag {tag_class}">{tag}</span></p>
<table aria-label="workspace heatmap">
{''.join(rows)}
</table>
{answer_note}
<h2>Legend</h2>
<ul class="legend">{legend}</ul>
</body>
</html>
"""


def _ansi_cell(prob: float) -> str:
    idx = _ramp_index(prob, len(ANSI_RAMP))
    fg = 16 if idx >= len(ANSI_RAMP) - 4 else 231
    return f"\x1b[48;5;{ANSI_RAMP[idx]}m\x1b[38;5;{fg}m  {ANSI_RESET}"


def _html_color(prob: float) -> str:
    idx = _ramp_index(prob, 7)
    colors = [
        "#071952",
        "#0b3b8f",
        "#075985",
        "#0891b2",
        "#22d3ee",
        "#bae6fd",
        "#ffffff",
    ]
    return colors[idx]


def _ramp_index(prob: float, count: int) -> int:
    # sqrt perceptual scaling: band-layer probabilities are small, a linear
    # ramp renders real heatmaps almost uniformly dark (matches chat.html)
    value = math.sqrt(min(1.0, max(0.0, prob)))
    return min(count - 1, int(value * (count - 1)))


def _legend_token(token: str) -> str:
    if token == "":
        return "<space>"
    return repr(token)
