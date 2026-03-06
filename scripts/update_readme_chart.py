"""Reads performance CSV, regenerates the chart and updates README stats."""

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "performance_weekly_performance.csv"
CHART_PATH = REPO_ROOT / "assets" / "performance_chart.png"
README_PATH = REPO_ROOT / "README.md"


def load_csv(path: Path) -> list[dict]:
    """Load CSV rows as list of dicts."""
    assert path.exists(), f"CSV not found: {path}"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) > 0, "CSV is empty"
    return rows


def compute_stats(rows: list[dict]) -> dict:
    """Derive summary stats from CSV rows."""
    gws = [int(r["game_week"]) for r in rows]
    my_pts = [int(r["my_points"]) for r in rows]
    avg_pts = [int(r["average_points"]) for r in rows]
    deltas = [int(r["delta_vs_avg"]) for r in rows]
    total = int(rows[-1]["total_points"])

    beat_or_matched = sum(1 for d in deltas if d >= 0)
    cumulative_delta = total - sum(avg_pts)

    best_idx = int(np.argmax(my_pts))
    worst_idx = int(np.argmin(my_pts))

    stats = {
        "num_gws": len(gws),
        "total_points": total,
        "cumulative_delta": cumulative_delta,
        "beat_count": beat_or_matched,
        "beat_pct": round(100 * beat_or_matched / len(gws)),
        "best_gw": gws[best_idx],
        "best_pts": my_pts[best_idx],
        "best_avg": avg_pts[best_idx],
        "worst_gw": gws[worst_idx],
        "worst_pts": my_pts[worst_idx],
        "worst_avg": avg_pts[worst_idx],
    }
    assert stats["num_gws"] == len(rows), "GW count mismatch"
    assert stats["total_points"] > 0, "Total points must be positive"
    return stats


def generate_chart(rows: list[dict], dest: Path) -> None:
    """Create the performance line chart matching existing style."""
    gws = [int(r["game_week"]) for r in rows]
    my_pts = [int(r["my_points"]) for r in rows]
    avg_pts = [int(r["average_points"]) for r in rows]

    my_arr = np.array(my_pts, dtype=float)
    avg_arr = np.array(avg_pts, dtype=float)

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(
        gws, my_pts, color="#2ecc71", marker="o",
        linewidth=2, markersize=6, label="My Points", zorder=3,
    )
    ax.plot(
        gws, avg_pts, color="#e67e22", marker="s",
        linewidth=2, markersize=6, label="Average Points", zorder=3,
    )

    ax.fill_between(
        gws, my_arr, avg_arr,
        where=(my_arr >= avg_arr),
        interpolate=True, alpha=0.2, color="#2ecc71",
    )
    ax.fill_between(
        gws, my_arr, avg_arr,
        where=(my_arr < avg_arr),
        interpolate=True, alpha=0.2, color="#e74c3c",
    )

    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Points")
    ax.set_title("My Points vs Average  (25/26 Season)", fontweight="bold")
    ax.set_xticks(gws)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert dest.exists(), f"Chart not saved to {dest}"


def update_readme(stats: dict, path: Path) -> None:
    """Replace the performance section in the README."""
    assert path.exists(), f"README not found: {path}"
    text = path.read_text(encoding="utf-8")

    sign = "+" if stats["cumulative_delta"] >= 0 else ""
    new_section = (
        f"## Model Performance (25/26 Season - "
        f"{stats['num_gws']} GWs)\n"
        f"\n"
        f"![My Points vs Average](assets/performance_chart.png)\n"
        f"\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Total Points | {stats['total_points']} |\n"
        f"| Cumulative Points Above Average | "
        f"{sign}{stats['cumulative_delta']} |\n"
        f"| Gameweeks Beat or Matched Average | "
        f"{stats['beat_count']} / {stats['num_gws']} "
        f"({stats['beat_pct']}%) |\n"
        f"| Best Gameweek | GW{stats['best_gw']}: "
        f"{stats['best_pts']} pts (avg {stats['best_avg']}) |\n"
        f"| Worst Gameweek | GW{stats['worst_gw']}: "
        f"{stats['worst_pts']} pts (avg {stats['worst_avg']}) |"
    )

    pattern = re.compile(
        r"## Model Performance.*?\| Worst Gameweek \|[^\n]+",
        re.DOTALL,
    )
    assert pattern.search(text), "Could not find performance section"
    updated = pattern.sub(new_section, text)
    assert updated != text, "README was not changed"

    path.write_text(updated, encoding="utf-8")


def main() -> int:
    """Entry point."""
    rows = load_csv(CSV_PATH)
    stats = compute_stats(rows)
    generate_chart(rows, CHART_PATH)
    update_readme(stats, README_PATH)
    print(
        f"Updated chart and README for "
        f"{stats['num_gws']} gameweeks "
        f"({stats['total_points']} total pts, "
        f"{'+' if stats['cumulative_delta'] >= 0 else ''}"
        f"{stats['cumulative_delta']} vs avg)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
