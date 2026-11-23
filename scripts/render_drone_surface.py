"""Render the recorded drone controller's thrust surface for the product page."""

import json
import re
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "site" / "assets"


def project(x: float, y: float, z: float) -> tuple[float, float]:
    return 355 + 52 * x + 47 * y, 309 - 19 * x + 20 * y - 150 * z


def coords(x: float, y: float, z: float) -> str:
    px, py = project(x, y, z)
    return f"{px:.2f},{py:.2f}"


def line(points: list[str], color: str, width: float) -> str:
    return f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def main() -> None:
    run = json.loads((ROOT / "studio/demo-data/runs/drone_altitude_pid.json").read_text())
    sl = run["default_slice"]
    xs, ys, grid = sl["x_vals"], sl["y_vals"], sl["prog_grid"]
    ids = sl["regime_grid"]
    indices = sorted({*range(0, len(xs), 2), len(xs) - 1})
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 708 475" role="img">',
        "<title>Drone altitude controller, sampled thrust at battery_v = 11.55</title>",
        '<g fill="none" stroke-linejoin="round">',
    ]
    for v in range(-3, 4):
        parts.append(line([coords(v, -3, 0), coords(v, 3, 0)], "#c5c7bc", 0.8))
        parts.append(line([coords(-3, v, 0), coords(3, v, 0)], "#c5c7bc", 0.8))
    for x, y in [(-3, -3), (-3, 3), (3, 3), (3, -3)]:
        parts.append(
            line([coords(x, y, 0), coords(x, y, grid[ys.index(y)][xs.index(x)])], "#b0b3a7", 1)
        )
    cells = list(pairwise(indices))
    for i, i2 in reversed(cells):
        for j, j2 in cells:
            corners = [(j, i), (j, i2), (j2, i2), (j2, i)]
            points = " ".join(coords(xs[col], ys[row], grid[row][col]) for row, col in corners)
            z = sum(grid[row][col] for row, col in corners) / 4
            t = min(1, max(0, z / 1.15))
            fill = tuple(round(a + (b - a) * t) for a, b in zip((231, 233, 220), (223, 98, 61)))
            parts.append(
                f'<polygon points="{points}" fill="rgb{fill}" stroke="#68695f" stroke-width=".63"/>'
            )
    parts.extend(["</g>", '<g font-family="monospace" font-size="15" fill="#505348">'])
    for x, y, dx, dy, label in [
        (-3, -3, -14, 18, "−3"),
        (-3, 3, -12, 20, "−3, +3"),
        (3, 3, 12, 8, "+3"),
        (0, 3, 0, 32, "alt_error"),
        (-3, 0, -45, 32, "vert_vel"),
    ]:
        px, py = project(x, y, 0)
        parts.append(f'<text x="{px + dx:.1f}" y="{py + dy:.1f}">{label}</text>')
    parts.append("</g></svg>")
    (ASSETS / "drone-surface.svg").write_text("\n".join(parts) + "\n")
    row = min(range(len(ys)), key=lambda j: abs(ys[j] - 0.064))
    records = []
    for i, x in enumerate(xs):
        px, py = project(x, ys[row], grid[row][i])
        records.append(
            {
                "x": round(x, 4),
                "region": ids[row][i],
                "thrust": round(grid[row][i], 6),
                "network": round(sl["nn_grid"][row][i], 6),
                "px": round(px, 2),
                "py": round(py, 2),
            }
        )
    source = run["result"]["code"]["c"]
    branches = dict(re.findall(r"/\* Regime (\d+):[^*]*\*/\s*(out_values\[0\] = [^;]+;)", source))
    if any(str(point["region"]) not in branches for point in records):
        raise ValueError("Recorded trace includes a region without C99 source")
    data = {"velocity": round(ys[row], 4), "points": records, "branches": branches}
    (ASSETS / "drone-trace.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
