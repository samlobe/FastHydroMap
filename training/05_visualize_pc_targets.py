#!/usr/bin/env python3
"""Create a lightweight SVG report for PC target/trust inspection."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "all_residue_results.csv"
OUT_DIR = ROOT / "figures"
TARGETS = ("PC1", "PC2", "PC3")


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else math.nan


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def load_rows(path: Path) -> list[dict[str, float | bool | str]]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            fdewet = as_float(raw, "Fdewet_pred")
            avg_waters = as_float(raw, "avg_n_waters")
            fdewet_mask = avg_waters > 7.0 and 3.8 <= fdewet <= 8.7
            rows.append(
                {
                    "pdb_id": raw["pdb_id"],
                    "aa": raw["aa"],
                    "trusted": as_bool(raw.get("trusted", "")),
                    "fdewet_mask": fdewet_mask,
                    "Fdewet_pred": fdewet,
                    "avg_n_waters": avg_waters,
                    "sasa": as_float(raw, "sasa"),
                    "PC1": as_float(raw, "PC1"),
                    "PC2": as_float(raw, "PC2"),
                    "PC3": as_float(raw, "PC3"),
                }
            )
    return rows


def finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def quantile(values: list[float], q: float) -> float:
    xs = sorted(finite(values))
    if not xs:
        return math.nan
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    xvals, yvals = zip(*pairs)
    mx, my = mean(xvals), mean(yvals)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xvals))
    den_y = math.sqrt(sum((y - my) ** 2 for y in yvals))
    return num / (den_x * den_y) if den_x and den_y else math.nan


def describe(values: list[float]) -> dict[str, float]:
    xs = finite(values)
    return {
        "n": float(len(xs)),
        "mean": mean(xs),
        "median": median(xs),
        "min": min(xs),
        "p05": quantile(xs, 0.05),
        "p95": quantile(xs, 0.95),
        "max": max(xs),
    }


def scale(value: float, low: float, high: float, start: float, stop: float) -> float:
    if not math.isfinite(value) or high <= low:
        return (start + stop) / 2.0
    return start + (value - low) * (stop - start) / (high - low)


def tick_values(low: float, high: float, count: int = 5) -> list[float]:
    if high <= low:
        return [low]
    return [low + (high - low) * i / (count - 1) for i in range(count)]


def panel_title(label: str, x: int, y: int) -> str:
    return f'<text x="{x}" y="{y}" class="title">{label}</text>'


def scatter_svg(
    rows: list[dict[str, float | bool | str]],
    x_key: str,
    y_key: str,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    max_points: int = 5000,
) -> str:
    xs = [float(r[x_key]) for r in rows]
    ys = [float(r[y_key]) for r in rows]
    x_low, x_high = quantile(xs, 0.01), quantile(xs, 0.99)
    y_low, y_high = quantile(ys, 0.01), quantile(ys, 0.99)
    pad_l, pad_r, pad_t, pad_b = 48, 14, 24, 38
    px0, px1 = x + pad_l, x + width - pad_r
    py0, py1 = y + pad_t, y + height - pad_b
    stride = max(1, len(rows) // max_points)
    sampled = rows[::stride]

    parts = [panel_title(title, x, y + 12), f'<rect x="{px0}" y="{py0}" width="{px1-px0}" height="{py1-py0}" class="plot"/>']
    for tv in tick_values(x_low, x_high):
        tx = scale(tv, x_low, x_high, px0, px1)
        parts.append(f'<line x1="{tx:.1f}" y1="{py0}" x2="{tx:.1f}" y2="{py1}" class="grid"/>')
        parts.append(f'<text x="{tx:.1f}" y="{py1+18}" class="tick" text-anchor="middle">{tv:.1f}</text>')
    for tv in tick_values(y_low, y_high):
        ty = scale(tv, y_low, y_high, py1, py0)
        parts.append(f'<line x1="{px0}" y1="{ty:.1f}" x2="{px1}" y2="{ty:.1f}" class="grid"/>')
        parts.append(f'<text x="{px0-8}" y="{ty+4:.1f}" class="tick" text-anchor="end">{tv:.1f}</text>')

    for trusted in (False, True):
        color = "#9ca3af" if not trusted else "#2563eb"
        opacity = "0.28" if not trusted else "0.5"
        for row in sampled:
            if bool(row["trusted"]) != trusted:
                continue
            xv, yv = float(row[x_key]), float(row[y_key])
            if not (math.isfinite(xv) and math.isfinite(yv)):
                continue
            cx = scale(max(min(xv, x_high), x_low), x_low, x_high, px0, px1)
            cy = scale(max(min(yv, y_high), y_low), y_low, y_high, py1, py0)
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.6" fill="{color}" opacity="{opacity}"/>')
    parts.append(f'<text x="{(px0+px1)/2:.1f}" y="{y+height-4}" class="label" text-anchor="middle">{x_key}</text>')
    parts.append(f'<text x="{x+12}" y="{(py0+py1)/2:.1f}" class="label rotate" text-anchor="middle">{y_key}</text>')
    return "\n".join(parts)


def hist_svg(
    rows: list[dict[str, float | bool | str]],
    key: str,
    x: int,
    y: int,
    width: int,
    height: int,
    bins: int = 34,
) -> str:
    values = [float(r[key]) for r in rows]
    low, high = quantile(values, 0.01), quantile(values, 0.99)
    trusted = [0] * bins
    untrusted = [0] * bins
    for row in rows:
        value = float(row[key])
        if not math.isfinite(value):
            continue
        ix = min(bins - 1, max(0, int((value - low) / (high - low) * bins))) if high > low else 0
        if bool(row["trusted"]):
            trusted[ix] += 1
        else:
            untrusted[ix] += 1

    max_count = max([a + b for a, b in zip(trusted, untrusted)] + [1])
    pad_l, pad_r, pad_t, pad_b = 48, 14, 24, 38
    px0, px1 = x + pad_l, x + width - pad_r
    py0, py1 = y + pad_t, y + height - pad_b
    bar_w = (px1 - px0) / bins
    parts = [panel_title(f"{key} distribution", x, y + 12), f'<rect x="{px0}" y="{py0}" width="{px1-px0}" height="{py1-py0}" class="plot"/>']
    for i, (u_count, t_count) in enumerate(zip(untrusted, trusted)):
        bx = px0 + i * bar_w
        uh = (py1 - py0) * u_count / max_count
        th = (py1 - py0) * t_count / max_count
        parts.append(f'<rect x="{bx:.1f}" y="{py1-uh:.1f}" width="{bar_w-1:.1f}" height="{uh:.1f}" fill="#9ca3af" opacity="0.55"/>')
        parts.append(f'<rect x="{bx:.1f}" y="{py1-uh-th:.1f}" width="{bar_w-1:.1f}" height="{th:.1f}" fill="#2563eb" opacity="0.72"/>')
    for tv in tick_values(low, high):
        tx = scale(tv, low, high, px0, px1)
        parts.append(f'<text x="{tx:.1f}" y="{py1+18}" class="tick" text-anchor="middle">{tv:.1f}</text>')
    parts.append(f'<text x="{(px0+px1)/2:.1f}" y="{y+height-4}" class="label" text-anchor="middle">{key}</text>')
    return "\n".join(parts)


def write_summary(rows: list[dict[str, float | bool | str]], path: Path) -> None:
    columns = ["target", "group", "n", "mean", "median", "min", "p05", "p95", "max", "corr_fdewet", "corr_avg_n_waters", "corr_sasa"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for target in TARGETS:
            for label, subset in (
                ("all", rows),
                ("trusted", [r for r in rows if bool(r["trusted"])]),
                ("untrusted", [r for r in rows if not bool(r["trusted"])]),
            ):
                vals = [float(r[target]) for r in subset]
                stats = describe(vals)
                writer.writerow(
                    {
                        "target": target,
                        "group": label,
                        **{k: f"{v:.6g}" for k, v in stats.items()},
                        "corr_fdewet": f"{pearson(vals, [float(r['Fdewet_pred']) for r in subset]):.6g}",
                        "corr_avg_n_waters": f"{pearson(vals, [float(r['avg_n_waters']) for r in subset]):.6g}",
                        "corr_sasa": f"{pearson(vals, [float(r['sasa']) for r in subset]):.6g}",
                    }
                )


def write_svg(rows: list[dict[str, float | bool | str]], path: Path) -> None:
    trusted_n = sum(1 for r in rows if bool(r["trusted"]))
    fdewet_mask_n = sum(1 for r in rows if bool(r["fdewet_mask"]))
    pdb_n = len({str(r["pdb_id"]) for r in rows})
    width, height = 1200, 1320
    panels = [
        hist_svg(rows, "PC1", 40, 150, 350, 260),
        hist_svg(rows, "PC2", 425, 150, 350, 260),
        hist_svg(rows, "PC3", 810, 150, 350, 260),
        scatter_svg(rows, "Fdewet_pred", "PC1", 40, 455, 350, 260, "PC1 vs Fdewet"),
        scatter_svg(rows, "avg_n_waters", "PC1", 425, 455, 350, 260, "PC1 vs waters"),
        scatter_svg(rows, "sasa", "PC1", 810, 455, 350, 260, "PC1 vs SASA"),
        scatter_svg(rows, "PC1", "PC2", 40, 760, 350, 260, "PC1 vs PC2"),
        scatter_svg(rows, "PC1", "PC3", 425, 760, 350, 260, "PC1 vs PC3"),
        scatter_svg(rows, "PC2", "PC3", 810, 760, 350, 260, "PC2 vs PC3"),
    ]
    text_lines = [
        "FastHydroMap PC target inspection",
        f"{len(rows):,} residues across {pdb_n:,} PDBs",
        f"trusted column: {trusted_n:,} residues ({trusted_n / len(rows):.1%})",
        f"Fdewet-derived trust rule: {fdewet_mask_n:,} residues ({fdewet_mask_n / len(rows):.1%})",
        "Blue = trusted, gray = untrusted; axes are clipped to the 1st and 99th percentiles.",
    ]
    summary = "\n".join(
        f'<text x="40" y="{38 + i * 22}" class="subtitle">{line}</text>'
        if i else f'<text x="40" y="42" class="heading">{line}</text>'
        for i, line in enumerate(text_lines)
    )
    legend = '<circle cx="40" cy="1120" r="6" fill="#2563eb" opacity="0.72"/><text x="54" y="1125" class="subtitle">trusted</text><circle cx="140" cy="1120" r="6" fill="#9ca3af" opacity="0.55"/><text x="154" y="1125" class="subtitle">untrusted</text>'
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .heading {{ font: 700 26px sans-serif; fill: #111827; }}
  .subtitle {{ font: 14px sans-serif; fill: #374151; }}
  .title {{ font: 700 15px sans-serif; fill: #111827; }}
  .label {{ font: 12px sans-serif; fill: #374151; }}
  .tick {{ font: 10px sans-serif; fill: #4b5563; }}
  .plot {{ fill: #f9fafb; stroke: #d1d5db; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
  .rotate {{ transform-box: fill-box; transform-origin: center; transform: rotate(-90deg); }}
</style>
<rect width="100%" height="100%" fill="white"/>
{summary}
{legend}
{chr(10).join(panels)}
</svg>
"""
    path.write_text(svg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DATA_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.csv)
    svg_path = args.out_dir / "pc_target_report.svg"
    summary_path = args.out_dir / "pc_target_summary.csv"
    write_svg(rows, svg_path)
    write_summary(rows, summary_path)
    print(f"[done] wrote {svg_path}")
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
