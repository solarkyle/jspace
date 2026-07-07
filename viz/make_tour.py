from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


WIDTH = 960
HEIGHT = 540
DPI = 100
BASE_FPS = 10
SIZE_LIMIT_MB = 14.0

BG = "#0d1017"
PANEL = "#151a24"
TEXT = "#e6e9f0"
DIM = "#8a93a6"
GREEN = "#57c47a"
RED = "#e0655f"
GOLD = "#d4a24e"
BLUE = "#5eb0ff"

ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = Path(__file__).resolve().parent
FRAME_DIR = VIZ_DIR / "tour_frames"
OUT_GIF = VIZ_DIR / "jspace_tour.gif"

SCENE_FRAMES = {
    1: 30,
    2: 96,
    3: 96,
    4: 80,
    5: 96,
    6: 40,
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": TEXT,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
    }
)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def smoothstep(v: float) -> float:
    v = clamp(v)
    return v * v * (3.0 - 2.0 * v)


def hex_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def blend(a: str | np.ndarray, b: str | np.ndarray, t: float) -> tuple[float, float, float]:
    aa = hex_rgb(a) if isinstance(a, str) else np.asarray(a, dtype=float)
    bb = hex_rgb(b) if isinstance(b, str) else np.asarray(b, dtype=float)
    out = aa * (1.0 - t) + bb * t
    return float(out[0]), float(out[1]), float(out[2])


def lighten(color: str, amount: float) -> tuple[float, float, float]:
    return blend(color, "#ffffff", amount)


def wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=True))


def make_canvas():
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    return fig, ax


def fig_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
    plt.close(fig)
    return arr


def add_panel(ax, x: float, y: float, w: float, h: float, alpha: float = 1.0, radius: float = 8.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=PANEL,
        edgecolor=blend(PANEL, BLUE, 0.18),
        linewidth=1.0,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def add_text(
    ax,
    x: float,
    y: float,
    s: str,
    size: float = 12,
    color: str | tuple[float, float, float] = TEXT,
    weight: str = "normal",
    ha: str = "left",
    va: str = "center",
    alpha: float = 1.0,
    linespacing: float = 1.12,
    **kwargs,
):
    return ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        alpha=alpha,
        linespacing=linespacing,
        clip_on=True,
        **kwargs,
    )


def draw_chip(ax, x: float, y: float, label: str, alpha: float, scale: float = 1.0):
    w = max(42, 9.0 * len(label) + 18) * scale
    h = 24 * scale
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=8",
            facecolor=blend(PANEL, BLUE, 0.10),
            edgecolor=blend(DIM, BLUE, 0.35),
            linewidth=0.8,
            alpha=alpha,
        )
    )
    add_text(ax, x, y, label, size=8.5 * scale, color=TEXT, ha="center", alpha=min(1.0, alpha + 0.18))


def load_uncertainty() -> list[dict]:
    path = ROOT / "data" / "uncertainty_trivia_gemma-4-e4b-it.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qa() -> list[dict]:
    return json.loads((ROOT / "data" / "qa_dump.json").read_text(encoding="utf-8"))


def load_emotion() -> list[dict]:
    return json.loads((ROOT / "data" / "emotion_matrix_5models.json").read_text(encoding="utf-8"))


def prepare_data():
    uncertainty = load_uncertainty()
    logprobs = np.array([float(r["bl_first_token_logprob"]) for r in uncertainty], dtype=float)
    median_logprob = float(np.median(logprobs))
    confident = [r for r in uncertainty if float(r["bl_first_token_logprob"]) > median_logprob]
    confident.sort(key=lambda r: (float(r["mean_entropy"]), not bool(r["correct"])))

    qa = load_qa()
    wrong = next(r for r in qa if "Downtown" in r["q"] and not bool(r["correct"]))
    correct_matches = [r for r in qa if "Lion's Gate" in r["q"] and bool(r["correct"])]
    correct = correct_matches[0] if correct_matches else next(r for r in qa if bool(r["correct"]))

    emotion = load_emotion()
    order = [
        "google/gemma-4-E4B-it",
        "google/gemma-4-12B-it",
        "huihui-ai/Huihui-gemma-4-12B-it-abliterated",
        "google/gemma-4-26B-A4B-it",
        "Qwen/Qwen3.6-27B",
    ]
    by_model = {r["model"]: r for r in emotion}
    emotion = [by_model[m] for m in order if m in by_model]

    return {
        "uncertainty": uncertainty,
        "confident": confident,
        "median_logprob": median_logprob,
        "qa_correct": correct,
        "qa_wrong": wrong,
        "qa_all": qa,
        "emotion": emotion,
    }


def scene_title(_data, p: float) -> np.ndarray:
    fig, ax = make_canvas()
    rng = np.random.default_rng(17)
    labels = ["token", "rank", "entropy", "belief", "answer", "layers", "workspace", "fog", "gate", "calm"]
    starts = []
    ends = []
    for i, label in enumerate(labels):
        starts.append((rng.uniform(90, 870), rng.uniform(95, 465), rng.uniform(-18, 18), rng.uniform(-10, 12)))
        ends.append((128 + i * 78, 356))

    snap = smoothstep((p - 0.24) / 0.68)
    drift = smoothstep(p / 0.55)
    for i, label in enumerate(labels):
        sx, sy, dx, dy = starts[i]
        ex, ey = ends[i]
        fog_x = sx + dx * math.sin(1.7 * p + i) * (1.0 - drift * 0.3)
        fog_y = sy + dy * math.cos(1.4 * p + i * 0.7) * (1.0 - drift * 0.3)
        x = fog_x * (1.0 - snap) + ex * snap
        y = fog_y * (1.0 - snap) + ey * snap
        alpha = 0.11 + 0.62 * snap
        draw_chip(ax, x, y, label, alpha=alpha, scale=0.92 + 0.08 * snap)

    for i in range(len(ends) - 1):
        x1, y1 = ends[i]
        x2, y2 = ends[i + 1]
        ax.plot([x1, x2], [y1, y2], color=BLUE, lw=1.2, alpha=0.35 * snap)

    title_a = smoothstep((p - 0.10) / 0.42)
    add_text(ax, WIDTH / 2, 276, "Reading a model's mind", size=40, weight="bold", ha="center", alpha=title_a)
    add_text(
        ax,
        WIDTH / 2,
        234,
        "Anthropic's global-workspace lens on 5 open models, replicated in 24h",
        size=16.5,
        color=DIM,
        ha="center",
        alpha=title_a,
    )
    line_w = 330 * smoothstep((p - 0.48) / 0.38)
    ax.plot([WIDTH / 2 - line_w / 2, WIDTH / 2 + line_w / 2], [204, 204], color=GOLD, lw=2.2, alpha=0.9)
    return fig_to_array(fig)


def axis_map(x0, y0, w, h, xmin, xmax, ymin, ymax):
    def mx(x):
        return x0 + (x - xmin) / (xmax - xmin) * w

    def my(y):
        return y0 + (y - ymin) / (ymax - ymin) * h

    return mx, my


def scene_fog(data, p: float) -> np.ndarray:
    fig, ax = make_canvas()
    confident = data["confident"]
    depths = np.linspace(0.25, 0.75, 21)
    arrs = np.array([r["layer_entropies"] for r in confident], dtype=float)
    correct_mask = np.array([bool(r["correct"]) for r in confident], dtype=bool)
    means_correct = arrs[correct_mask].mean(axis=0)
    means_wrong = arrs[~correct_mask].mean(axis=0)

    y_min = float(np.floor(arrs.min() * 2.0) / 2.0)
    y_max = float(np.ceil(arrs.max() * 2.0) / 2.0)
    x0, y0, w, h = 86, 86, 790, 350
    mx, my = axis_map(x0, y0, w, h, 0.25, 0.75, y_min, y_max)

    add_panel(ax, 54, 46, 852, 438)
    add_text(ax, 70, 506, "THE FOG", size=13, color=GOLD, weight="bold")
    title_a = smoothstep((p - 0.68) / 0.22)
    add_text(
        ax,
        490,
        506,
        "wrong answers are visibly foggier\nBEFORE the model speaks",
        size=16,
        weight="bold",
        ha="center",
        alpha=max(0.35, title_a),
    )

    for t in np.linspace(y_min, y_max, 5):
        yy = my(t)
        ax.plot([x0, x0 + w], [yy, yy], color="#273041", lw=0.8, alpha=0.65)
        add_text(ax, x0 - 12, yy, f"{t:.1f}", size=8.5, color=DIM, ha="right")
    for t in [0.25, 0.50, 0.75]:
        xx = mx(t)
        ax.plot([xx, xx], [y0, y0 + h], color="#202838", lw=0.8, alpha=0.5)
        add_text(ax, xx, y0 - 24, f"{t:.2f}", size=9, color=DIM, ha="center")
    ax.plot([x0, x0 + w], [y0, y0], color=DIM, lw=1.0, alpha=0.7)
    ax.plot([x0, x0], [y0, y0 + h], color=DIM, lw=1.0, alpha=0.7)
    add_text(ax, x0 + w / 2, 38, "band depth", size=10, color=DIM, ha="center")
    add_text(ax, 27, y0 + h / 2, "entropy", size=10, color=DIM, ha="center", rotation=90)

    line_progress = smoothstep(p / 0.62)
    n = int(1 + line_progress * (len(confident) - 1))
    k = max(2, int(2 + line_progress * (len(depths) - 2)))
    xs = [mx(x) for x in depths[:k]]
    for row in confident[:n]:
        ys = [my(v) for v in row["layer_entropies"][:k]]
        ax.plot(xs, ys, color=GREEN if row["correct"] else RED, lw=0.72, alpha=0.12)

    mean_progress = smoothstep((p - 0.43) / 0.34)
    mk = max(2, int(2 + mean_progress * (len(depths) - 2)))
    mxs = [mx(x) for x in depths[:mk]]
    ax.plot(mxs, [my(v) for v in means_correct[:mk]], color=GREEN, lw=3.2, alpha=0.95 * mean_progress)
    ax.plot(mxs, [my(v) for v in means_wrong[:mk]], color=RED, lw=3.2, alpha=0.95 * mean_progress)

    ann = smoothstep((p - 0.70) / 0.22)
    if ann > 0:
        cx, cy = mx(0.71), my(float(means_correct[-2]))
        wx, wy = mx(0.69), my(float(means_wrong[-2]))
        add_text(ax, 640, 178, "confident + correct:\ncalm inside", size=12.5, color=GREEN, weight="bold", alpha=ann)
        add_text(ax, 630, 356, "confident + WRONG:\nfog", size=12.5, color=RED, weight="bold", alpha=ann)
        ax.add_patch(FancyArrowPatch((628, 180), (cx, cy), arrowstyle="->", color=GREEN, lw=1.4, alpha=ann))
        ax.add_patch(FancyArrowPatch((626, 352), (wx, wy), arrowstyle="->", color=RED, lw=1.4, alpha=ann))

    return fig_to_array(fig)


def sanitize_token(tok: str) -> str:
    tok = str(tok).strip().replace("\n", " ")
    if not tok:
        return "\u00b7"
    if any(ord(ch) >= 0x0500 for ch in tok):
        return "\u00b7"
    tok = " ".join(tok.split())
    if len(tok) > 13:
        tok = tok[:11] + ".."
    return tok


def entropy_fill(entropy: float, e_min: float, e_max: float, alpha_scale: float = 1.0):
    t = (entropy - e_min) / max(1e-9, e_max - e_min)
    heat = blend(GREEN, RED, smoothstep(t))
    return blend(PANEL, heat, 0.58 * alpha_scale)


def draw_verdict(ax, x: float, y: float, text: str, color: str, alpha: float = 1.0):
    w = 78 if text == "CORRECT" else 64
    ax.add_patch(
        FancyBboxPatch(
            (x, y - 12),
            w,
            24,
            boxstyle="round,pad=0,rounding_size=7",
            facecolor=blend(PANEL, color, 0.36),
            edgecolor=color,
            linewidth=1.0,
            alpha=alpha,
        )
    )
    add_text(ax, x + w / 2, y, text, size=9.5, color=TEXT, weight="bold", ha="center", alpha=alpha)


def draw_qa_column(ax, example: dict, x: float, y: float, w: float, h: float, e_min: float, e_max: float, p: float, caption: str):
    verdict = "CORRECT" if bool(example["correct"]) else "WRONG"
    verdict_color = GREEN if bool(example["correct"]) else RED
    add_panel(ax, x, y, w, h)
    add_text(ax, x + 16, y + h - 26, wrap(example["q"], 43), size=10.5, color=TEXT, weight="bold", va="top")
    add_text(ax, x + 16, y + h - 78, "model answer:", size=9, color=DIM)
    add_text(ax, x + 118, y + h - 78, str(example["model_answer"])[:26], size=11, color=TEXT, weight="bold")
    draw_verdict(ax, x + w - 104, y + h - 78, verdict, verdict_color)

    row_top = y + h - 120
    row_bottom = y + 72
    row_h = (row_top - row_bottom) / 21.0
    layers = sorted(example["layers"], key=lambda z: int(z["layer"]), reverse=True)
    scan = smoothstep(p)
    scan_y = row_bottom + scan * (row_top - row_bottom)
    token_x = x + 70
    token_w = w - 88
    cell_w = token_w / 4.0

    add_text(ax, x + 17, row_top + 14, "layer", size=8.5, color=DIM)
    add_text(ax, token_x, row_top + 14, "top tokens inside the workspace", size=8.5, color=DIM)

    for i, layer in enumerate(layers):
        yy = row_top - (i + 1) * row_h
        center_y = yy + row_h / 2.0
        revealed = center_y <= scan_y + 0.1
        bg = entropy_fill(float(layer["entropy"]), e_min, e_max, 1.0 if revealed else 0.38)
        ax.add_patch(Rectangle((x + 14, yy + 0.7), w - 28, row_h - 1.4, facecolor=bg, edgecolor="none", alpha=0.95))
        add_text(ax, x + 26, center_y, str(layer["layer"]), size=7.6, color=DIM if not revealed else TEXT, ha="center")
        if revealed:
            toks = [sanitize_token(t) for t in layer["top"][:4]]
            for j, tok in enumerate(toks):
                ax.add_patch(
                    Rectangle(
                        (token_x + j * cell_w + 2, yy + 2.0),
                        cell_w - 4,
                        row_h - 4.0,
                        facecolor=blend(BG, PANEL, 0.55),
                        edgecolor=blend(PANEL, BLUE, 0.14),
                        linewidth=0.4,
                        alpha=0.72,
                    )
                )
                add_text(
                    ax,
                    token_x + j * cell_w + cell_w / 2,
                    center_y,
                    tok,
                    size=6.6,
                    color=TEXT,
                    ha="center",
                    alpha=0.98,
                )

    ax.plot([x + 12, x + w - 12], [scan_y, scan_y], color=GOLD, lw=2.0, alpha=0.96)
    cap_a = smoothstep((p - 0.78) / 0.18)
    add_text(ax, x + w / 2, y + 36, wrap(caption, 38), size=11.5, color=GOLD, weight="bold", ha="center", alpha=cap_a)


def scene_qa(data, p: float) -> np.ndarray:
    fig, ax = make_canvas()
    add_text(ax, 48, 510, "CONFIDENTLY WRONG, ONE REAL QUESTION", size=13, color=GOLD, weight="bold")
    add_text(ax, 912, 510, "scan: shallow to deep layers", size=10, color=DIM, ha="right")

    entropies = []
    for row in data["qa_all"]:
        entropies.extend(float(layer["entropy"]) for layer in row["layers"])
    e_min, e_max = min(entropies), max(entropies)

    draw_qa_column(
        ax,
        data["qa_correct"],
        36,
        58,
        426,
        420,
        e_min,
        e_max,
        p,
        "holds ONE semantic category",
    )
    draw_qa_column(
        ax,
        data["qa_wrong"],
        498,
        58,
        426,
        420,
        e_min,
        e_max,
        p,
        "rummages through a name soup, then fluently says the wrong thing",
    )
    return fig_to_array(fig)


def model_short_label(model: str) -> str:
    if "E4B" in model:
        return "E4B 4B"
    if "abliterated" in model:
        return "12B ablit"
    if "26B" in model:
        return "26B MoE"
    if "Qwen" in model:
        return "Qwen 27B"
    return "12B"


def emotion_color(value: float):
    norm = clamp((value + 8.0) / 16.0)
    rgb = np.array(plt.get_cmap("RdBu_r")(norm)[:3])
    return blend(PANEL, rgb, 0.94)


def draw_emotion_map(ax, item: dict, x: float, y: float, size: float, p: float, idx: int):
    local = smoothstep((p - idx * 0.115) / 0.23)
    if local <= 0:
        return
    conds = ["fury", "terror", "grief", "euphoria", "amusement"]
    lex = conds
    scale = 0.84 + 0.16 * local
    s = size * scale
    x0 = x + (size - s) / 2.0
    y0 = y + (size - s) / 2.0
    cell = s / 5.0

    add_text(ax, x + size / 2, y + size + 24, model_short_label(item["model"]), size=10.2, color=TEXT, weight="bold", ha="center", alpha=local)
    for r, cond in enumerate(conds):
        for c, lx in enumerate(lex):
            value = float(item["delta_matrix"][cond][lx])
            cy = y0 + (4 - r) * cell
            cx = x0 + c * cell
            ax.add_patch(
                Rectangle(
                    (cx, cy),
                    cell - 1.2,
                    cell - 1.2,
                    facecolor=emotion_color(value),
                    edgecolor=blend(BG, PANEL, 0.7),
                    linewidth=0.35,
                    alpha=local,
                )
            )

    letters = ["F", "T", "G", "E", "A"]
    for c, letter in enumerate(letters):
        add_text(ax, x0 + c * cell + cell / 2, y0 - 11, letter, size=6.8, color=DIM, ha="center", alpha=local)
    if idx == 0:
        for r, letter in enumerate(letters):
            add_text(ax, x0 - 9, y0 + (4 - r) * cell + cell / 2, letter, size=6.8, color=DIM, ha="right", alpha=local)

    diag = smoothstep((p - 0.54 - idx * 0.035) / 0.26)
    diag_cells = int(math.ceil(diag * 5))
    for d in range(diag_cells):
        ax.add_patch(
            Rectangle(
                (x0 + d * cell - 1.0, y0 + (4 - d) * cell - 1.0),
                cell + 0.8,
                cell + 0.8,
                facecolor="none",
                edgecolor=GOLD,
                linewidth=2.0,
                alpha=min(1.0, local * diag),
            )
        )


def scene_emotion(data, p: float) -> np.ndarray:
    fig, ax = make_canvas()
    add_text(ax, 48, 506, "EMOTION DIAGONAL", size=13, color=GOLD, weight="bold")
    add_panel(ax, 34, 72, 892, 390)
    size = 112
    gap = 58
    start_x = 64
    y = 238
    for i, item in enumerate(data["emotion"]):
        draw_emotion_map(ax, item, start_x + i * (size + gap), y, size, p, i)

    add_text(ax, 89, 210, "condition", size=8.5, color=DIM, rotation=90, ha="center")
    add_text(ax, 480, 188, "lexicon", size=8.5, color=DIM, ha="center")
    cap_a = smoothstep((p - 0.68) / 0.24)
    add_text(
        ax,
        WIDTH / 2,
        124,
        wrap(
            "tell it to SECRETLY feel an emotion: the right emotion lights up inside, and the diagonal sharpens with capability",
            60,
        ),
        size=13.5,
        color=TEXT,
        weight="bold",
        ha="center",
        alpha=cap_a,
    )
    return fig_to_array(fig)


def zscore(vals: np.ndarray) -> np.ndarray:
    std = float(vals.std())
    if std < 1e-9:
        return vals * 0.0
    return (vals - vals.mean()) / std


def scene_router(data, p: float) -> np.ndarray:
    fig, ax = make_canvas()
    rows = data["uncertainty"]
    xs = np.array([float(r["bl_first_token_logprob"]) for r in rows], dtype=float)
    ys = np.array([float(r["mean_entropy"]) for r in rows], dtype=float)
    correct = np.array([bool(r["correct"]) for r in rows], dtype=bool)
    distrust = zscore(ys) - zscore(xs)
    order = np.argsort(-distrust)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.10
    x0, y0, w, h = 86, 86, 620, 362
    mx, my = axis_map(x0, y0, w, h, x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)

    add_panel(ax, 48, 48, 860, 436)
    add_text(ax, 70, 506, "THE ROUTER", size=13, color=GOLD, weight="bold")

    for t in np.linspace(y_min, y_max, 5):
        yy = my(t)
        ax.plot([x0, x0 + w], [yy, yy], color="#273041", lw=0.8, alpha=0.6)
        add_text(ax, x0 - 12, yy, f"{t:.1f}", size=8.5, color=DIM, ha="right")
    for t in np.linspace(x_min, x_max, 5):
        xx = mx(t)
        ax.plot([xx, xx], [y0, y0 + h], color="#202838", lw=0.8, alpha=0.5)
        add_text(ax, xx, y0 - 22, f"{t:.2f}", size=8.2, color=DIM, ha="center")
    ax.plot([x0, x0 + w], [y0, y0], color=DIM, lw=1.0, alpha=0.7)
    ax.plot([x0, x0], [y0, y0 + h], color=DIM, lw=1.0, alpha=0.7)
    add_text(ax, x0 + w / 2, 38, "baseline first-token logprob", size=10, color=DIM, ha="center")
    add_text(ax, 28, y0 + h / 2, "mean entropy", size=10, color=DIM, ha="center", rotation=90)

    sx = [mx(v) for v in xs]
    sy = [my(v) for v in ys]
    colors = [GREEN if c else RED for c in correct]
    ax.scatter(sx, sy, s=17, c=colors, alpha=0.56, linewidths=0)

    budget = 0.5 * smoothstep(p)
    k = int(round(len(rows) * budget))
    selected = order[:k]
    if k > 0:
        ax.scatter(
            [sx[i] for i in selected],
            [sy[i] for i in selected],
            s=52,
            facecolors="none",
            edgecolors=GOLD,
            linewidths=1.45,
            alpha=0.92,
        )

    total_correct = int(correct.sum())
    selected_correct = int(correct[selected].sum()) if k else 0
    projected = (total_correct - selected_correct + 0.90 * k) / len(rows) * 100.0
    budget_pct = budget * 100.0

    add_panel(ax, 730, 134, 146, 236, alpha=0.96)
    add_text(ax, 803, 330, "live budget", size=11, color=DIM, ha="center")
    add_text(ax, 803, 288, f"escalated:\n{budget_pct:0.0f}%", size=24, color=GOLD, weight="bold", ha="center")
    add_text(ax, 803, 220, "accuracy:", size=11, color=DIM, ha="center")
    add_text(ax, 803, 184, f"42.8% ->\n{projected:0.1f}%", size=20, color=TEXT, weight="bold", ha="center")

    cap_a = smoothstep((p - 0.72) / 0.22)
    add_text(
        ax,
        480,
        468,
        "route on thoughts, not outputs:\none forward pass, no labels, no extra model",
        size=14,
        color=TEXT,
        weight="bold",
        ha="center",
        alpha=max(0.45, cap_a),
    )
    return fig_to_array(fig)


def scene_end(_data, p: float) -> np.ndarray:
    fig, ax = make_canvas()
    fade = smoothstep(p / 0.35)
    add_text(ax, WIDTH / 2, 480, "What held up", size=34, color=TEXT, weight="bold", ha="center", alpha=fade)
    add_text(
        ax,
        WIDTH / 2,
        432,
        wrap(
            "replicates on 4/5 models (pre-registered gate passed 3/4); fails on Qwen 27B, whose logprobs are already calibrated. Misses reported.",
            64,
        ),
        size=13,
        color=DIM,
        ha="center",
        alpha=fade,
    )

    table_a = smoothstep((p - 0.18) / 0.42)
    add_panel(ax, 150, 158, 660, 214, alpha=table_a)
    rows = [
        ("replication", "4/5 models"),
        ("pre-registered gate", "3/4 passed"),
        ("known miss", "Qwen 27B calibrated logprobs"),
        ("reporting", "misses reported"),
    ]
    for i, (k, v) in enumerate(rows):
        yy = 334 - i * 43
        ax.plot([178, 782], [yy - 23, yy - 23], color="#263044", lw=0.8, alpha=0.6 * table_a)
        add_text(ax, 194, yy, k, size=12, color=DIM, alpha=table_a)
        add_text(ax, 478, yy, v, size=14, color=TEXT if i != 2 else GOLD, weight="bold", alpha=table_a)

    link_a = smoothstep((p - 0.56) / 0.34)
    links = [
        "github.com/solarkyle/jspace",
        "demo: solarkyle.github.io/jspace/demo",
        "lenses: hf.co/solarkyle/jspace-lenses",
    ]
    for i, link in enumerate(links):
        add_text(ax, WIDTH / 2, 102 - i * 28, link, size=14.5, color=BLUE, ha="center", alpha=link_a)
    return fig_to_array(fig)


SCENE_RENDERERS = {
    1: scene_title,
    2: scene_fog,
    3: scene_qa,
    4: scene_emotion,
    5: scene_router,
    6: scene_end,
}


def save_preview(scene: int, arr: np.ndarray):
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(FRAME_DIR / f"scene{scene}_preview.png")


def render_all(data) -> list[np.ndarray]:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAME_DIR.glob("scene*_preview.png"):
        old.unlink()

    frames: list[np.ndarray] = []
    for scene in range(1, 7):
        total = SCENE_FRAMES[scene]
        preview_at = max(0, int(total * 0.82) - 1)
        renderer = SCENE_RENDERERS[scene]
        for i in range(total):
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == total:
                print(f"scene {scene}: frame {i + 1}/{total}", flush=True)
            p = i / max(1, total - 1)
            arr = renderer(data, p)
            frames.append(arr)
            if i == preview_at:
                save_preview(scene, arr)
    return frames


def pil_constants():
    if hasattr(Image, "Palette"):
        adaptive = Image.Palette.ADAPTIVE
    else:
        adaptive = Image.ADAPTIVE
    if hasattr(Image, "Dither"):
        dither_none = Image.Dither.NONE
    else:
        dither_none = Image.NONE
    return adaptive, dither_none


def decimate_frames(frames: list[np.ndarray], src_fps: int, dst_fps: int) -> list[np.ndarray]:
    if dst_fps >= src_fps:
        return frames
    target = max(1, int(round(len(frames) * dst_fps / src_fps)))
    indices = np.linspace(0, len(frames) - 1, target).round().astype(int)
    return [frames[int(i)] for i in indices]


def write_gif(frames: list[np.ndarray], colors: int, fps: int, path: Path):
    adaptive, dither_none = pil_constants()
    pil_frames = [
        Image.fromarray(frame, mode="RGB").convert("P", palette=adaptive, colors=colors, dither=dither_none)
        for frame in frames
    ]
    duration = int(round(1000 / fps))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )


def export_under_budget(frames: list[np.ndarray]) -> tuple[float, int, int, int]:
    attempts = [
        (128, BASE_FPS),
        (96, BASE_FPS),
        (64, BASE_FPS),
        (64, 9),
        (48, 9),
        (32, 9),
    ]
    last = None
    for colors, fps in attempts:
        out_frames = decimate_frames(frames, BASE_FPS, fps)
        if last is None:
            print(f"export: {colors} colors at {fps} fps", flush=True)
        elif colors != last[0]:
            print(f"export: reducing palette to {colors} colors", flush=True)
        elif fps != last[1]:
            print(f"export: dropping fps to {fps}", flush=True)
        write_gif(out_frames, colors, fps, OUT_GIF)
        size_mb = OUT_GIF.stat().st_size / (1024 * 1024)
        print(f"export: {size_mb:.2f} MB with {len(out_frames)} frames", flush=True)
        if size_mb <= SIZE_LIMIT_MB:
            return size_mb, len(out_frames), colors, fps
        last = (colors, fps)
    return OUT_GIF.stat().st_size / (1024 * 1024), len(decimate_frames(frames, BASE_FPS, attempts[-1][1])), attempts[-1][0], attempts[-1][1]


def main():
    data = prepare_data()
    frames = render_all(data)
    size_mb, frame_count, colors, fps = export_under_budget(frames)
    total_seconds = frame_count / fps
    print(f"final: {OUT_GIF.as_posix()}", flush=True)
    print(f"final: {size_mb:.2f} MB, {frame_count} frames, {colors} colors, {fps} fps, {total_seconds:.1f}s", flush=True)
    if size_mb > SIZE_LIMIT_MB:
        raise SystemExit(f"GIF is still over {SIZE_LIMIT_MB:.0f} MB after reductions: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
