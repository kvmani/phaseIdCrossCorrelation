from __future__ import annotations

from pathlib import Path


SVG_WIDTH = 1680
SVG_HEIGHT = 980
TITLE_Y = 62
SUBTITLE_Y = 98
BLOCK_W = 300
BLOCK_H = 170
ROW1_Y = 170
ROW2_Y = 455
ROW1_X = [70, 440, 810, 1180]
ROW2_X = [310, 680, 1050]

COLORS = {
    "input": "#dbeafe",
    "conv_light": "#c7f9cc",
    "conv_mid": "#86efac",
    "conv_dark": "#4ade80",
    "reshape": "#fde68a",
    "head": "#fca5a5",
    "output": "#e9d5ff",
    "text": "#111827",
    "muted": "#475569",
    "stroke": "#1f2937",
    "guide": "#334155",
    "canvas": "#f8fafc",
    "panel": "#ffffff",
}


def _rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    stroke: str = COLORS["stroke"],
    stroke_width: float = 2.2,
    rx: float = 18,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 24,
    weight: str = "400",
    anchor: str = "middle",
    fill: str = COLORS["text"],
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{value}</text>'
    )


def _multiline_text(
    x: float,
    y: float,
    lines: list[str],
    *,
    size: int = 22,
    line_gap: int = 28,
    anchor: str = "middle",
    weight: str = "400",
    fill: str = COLORS["text"],
) -> str:
    out = [
        f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
    ]
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else line_gap
        out.append(f'<tspan x="{x}" dy="{dy}">{line}</tspan>')
    out.append("</text>")
    return "".join(out)


def _path(points: list[tuple[float, float]], *, stroke: str = COLORS["guide"], stroke_width: float = 4.0, marker_end: str | None = None) -> str:
    d = " M " + " L ".join(f"{x},{y}" for x, y in points)
    marker = f' marker-end="url(#{marker_end})"' if marker_end else ""
    return (
        f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{marker}/>'
    )


def _badge(x: float, y: float, text: str, *, fill: str = "#ffffff", stroke: str = "#94a3b8") -> str:
    width = max(120, 14 * len(text))
    return "".join(
        [
            _rect(x, y, width, 34, fill=fill, stroke=stroke, stroke_width=1.4, rx=10),
            _text(x + width / 2, y + 24, text, size=19, weight="700", fill=COLORS["guide"]),
        ]
    )


def _block(x: float, y: float, *, title: str, lines: list[str], output_tag: str, fill: str) -> str:
    cx = x + BLOCK_W / 2
    parts = [
        _rect(x, y, BLOCK_W, BLOCK_H, fill=fill),
        _text(cx, y + 34, title, size=30, weight="700"),
        _multiline_text(cx, y + 76, lines, size=23, line_gap=28),
        _text(cx, y + BLOCK_H + 34, output_tag, size=20, weight="700", fill=COLORS["guide"]),
    ]
    return "".join(parts)


def build_svg() -> str:
    row1 = [
        {
            "x": ROW1_X[0],
            "y": ROW1_Y,
            "title": "Input Pattern",
            "lines": ["1 x 128 x 128 grayscale image", "Circular mask enabled", "Normalization: mean 0, std 1"],
            "output": "Tensor: 1 x 128 x 128",
            "fill": COLORS["input"],
        },
        {
            "x": ROW1_X[1],
            "y": ROW1_Y,
            "title": "Feature Block 1",
            "lines": ["Conv 3x3, 1 -> 32", "BatchNorm + ReLU", "MaxPool 2x2"],
            "output": "32 x 64 x 64",
            "fill": COLORS["conv_light"],
        },
        {
            "x": ROW1_X[2],
            "y": ROW1_Y,
            "title": "Feature Block 2",
            "lines": ["Conv 3x3, 32 -> 64", "BatchNorm + ReLU", "MaxPool 2x2"],
            "output": "64 x 32 x 32",
            "fill": COLORS["conv_mid"],
        },
        {
            "x": ROW1_X[3],
            "y": ROW1_Y,
            "title": "Feature Block 3",
            "lines": ["Conv 3x3, 64 -> 128", "BatchNorm + ReLU", "AdaptiveAvgPool 1x1"],
            "output": "128 x 1 x 1",
            "fill": COLORS["conv_dark"],
        },
    ]
    row2 = [
        {
            "x": ROW2_X[0],
            "y": ROW2_Y,
            "title": "Flatten",
            "lines": ["Vectorize pooled tensor", "No learned parameters"],
            "output": "128 features",
            "fill": COLORS["reshape"],
        },
        {
            "x": ROW2_X[1],
            "y": ROW2_Y,
            "title": "Linear Head",
            "lines": ["Fully connected layer", "128 -> 3 logits"],
            "output": "3 logits",
            "fill": COLORS["head"],
        },
        {
            "x": ROW2_X[2],
            "y": ROW2_Y,
            "title": "Prediction",
            "lines": ["Argmax for class label", "Softmax only for reporting"],
            "output": "Al / Cu / Ni",
            "fill": COLORS["output"],
        },
    ]

    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        """
  <defs>
    <marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="5" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L11,5 L0,10 z" fill="#334155"/>
    </marker>
  </defs>
""",
        f'<rect x="0" y="0" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="{COLORS["canvas"]}"/>',
        _text(SVG_WIDTH / 2, TITLE_Y, "CNN-32 Architecture Used in the Current EBSD Phase-Classification Study", size=34, weight="700"),
        _text(
            SVG_WIDTH / 2,
            SUBTITLE_Y,
            "Exact implementation: SimpleCnnClassifier(width=32), input size 128 x 128, three output classes",
            size=22,
            fill=COLORS["muted"],
        ),
    ]

    for block in row1 + row2:
        fragments.append(
            _block(
                block["x"],
                block["y"],
                title=block["title"],
                lines=block["lines"],
                output_tag=block["output"],
                fill=block["fill"],
            )
        )

    row1_mid_y = ROW1_Y + BLOCK_H / 2
    row2_mid_y = ROW2_Y + BLOCK_H / 2
    for left, right in zip(row1[:-1], row1[1:]):
        fragments.append(
            _path(
                [
                    (left["x"] + BLOCK_W + 8, row1_mid_y),
                    (right["x"] - 14, row1_mid_y),
                ],
                marker_end="arrow",
            )
        )

    fragments.append(
        _path(
            [
                (row1[-1]["x"] + BLOCK_W / 2, ROW1_Y + BLOCK_H + 10),
                (row1[-1]["x"] + BLOCK_W / 2, ROW2_Y - 48),
                (row2[0]["x"] + BLOCK_W / 2, ROW2_Y - 48),
                (row2[0]["x"] + BLOCK_W / 2, ROW2_Y - 12),
            ],
            marker_end="arrow",
        )
    )

    for left, right in zip(row2[:-1], row2[1:]):
        fragments.append(
            _path(
                [
                    (left["x"] + BLOCK_W + 8, row2_mid_y),
                    (right["x"] - 14, row2_mid_y),
                ],
                marker_end="arrow",
            )
        )

    fragments.extend(
        [
            _text(590, 386, "Pooling halves spatial resolution", size=20, weight="700", fill=COLORS["guide"]),
            _text(1330, 386, "Global average pooling removes spatial layout", size=20, weight="700", fill=COLORS["guide"]),
            _text(860, 725, "Only the final linear layer performs class scoring", size=20, weight="700", fill=COLORS["guide"]),
        ]
    )

    legend_x = 70
    legend_y = 744
    legend_w = 620
    legend_h = 206
    legend_rows = [
        (COLORS["input"], "Input and preprocessing"),
        (COLORS["conv_mid"], "Learned convolutional feature extraction"),
        (COLORS["reshape"], "Tensor reshape"),
        (COLORS["head"], "Classifier head"),
        (COLORS["output"], "Prediction and probability reporting"),
    ]
    fragments.append(_rect(legend_x, legend_y, legend_w, legend_h, fill=COLORS["panel"], stroke_width=2.0, rx=20))
    fragments.append(_text(legend_x + 42, legend_y + 38, "Legend", size=28, weight="700", anchor="start"))
    yy = legend_y + 74
    for fill, label in legend_rows:
        fragments.append(_rect(legend_x + 28, yy - 18, 38, 24, fill=fill, stroke=COLORS["stroke"], stroke_width=1.4, rx=6))
        fragments.append(_text(legend_x + 84, yy, label, size=22, anchor="start"))
        yy += 31

    panel_x = 760
    panel_y = 748
    panel_w = 850
    panel_h = 192
    notes = [
        "Source file: src/phase_id_xcorr/ml/models.py",
        "Convolution blocks: Conv2d -> BatchNorm2d -> ReLU",
        "Pooling schedule: MaxPool after blocks 1 and 2, AdaptiveAvgPool after block 3",
        "Classifier output dimension: 3 logits for Al, Cu, and Ni",
    ]
    fragments.append(_rect(panel_x, panel_y, panel_w, panel_h, fill=COLORS["panel"], stroke_width=2.0, rx=20))
    fragments.append(_text(panel_x + 32, panel_y + 38, "Implementation Notes", size=28, weight="700", anchor="start"))
    note_y = panel_y + 76
    for line in notes:
        fragments.append(_text(panel_x + 38, note_y, f"- {line}", size=22, anchor="start"))
        note_y += 31

    fragments.append("</svg>")
    return "".join(fragments)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "assets" / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cnn32_architecture.svg"
    out_path.write_text(build_svg(), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
