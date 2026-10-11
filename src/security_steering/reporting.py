from __future__ import annotations

import csv
import json
from pathlib import Path

from security_steering.io import read_json


def build_report(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    summaries = read_json(run_dir / "summary.json")
    selection = read_json(run_dir / "selection.json")
    label = (
        "OFFLINE SMOKE TEST — RANDOM WEIGHTS, NOT DEFENSE EVIDENCE"
        if manifest["is_smoke"]
        else ("PRETRAINED MODEL PILOT — exploratory, not a security guarantee")
    )
    columns = ("attack_success_rate", "benign_task_success", "attack_task_success", "joint_success")
    lines = [
        "# Security steering experiment report",
        "",
        label,
        "",
        f"Model: `{manifest['model_name']}`. Test families: {manifest['test_family_count']}.",
        "",
        "Placement, vector method, and fixed α were selected using validation data only. "
        "The risk probe and neural α controller were fitted on train data only.",
        "",
        "| Method | Attack success ↓ | Benign utility ↑ | Attacked utility ↑ | Joint success ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    csv_rows = []
    for name, summary in summaries.items():
        values = [summary[key]["value"] for key in columns]
        lines.append(
            "| "
            + " | ".join([name, *["ungraded" if v is None else f"{100 * v:.1f}%" for v in values]])
            + " |"
        )
        csv_rows.append(
            {
                "method": name,
                "n": summary["n"],
                **dict(zip(columns, values)),
                "latency_seconds": summary["mean_latency_seconds"],
                "output_tokens": summary["mean_output_tokens"],
                "alpha": summary["mean_alpha"],
            }
        )
    lines.extend(
        [
            "",
            "Selected fixed intervention:",
            "",
            "```json",
            json.dumps(selection["best_fixed"], indent=2),
            "```",
            "",
            f"Utility constraint met: {selection['utility_constraint_satisfied']}. "
            f"Validation recommends steering over baseline: {selection['recommend_fixed_over_baseline']}.",
            "",
            "Per-metric denominators, Wilson intervals, class breakdowns and paired family-bootstrap "
            "deltas are in `summary.json`. Wilson intervals are descriptive; paired variants are correlated. "
            "Use the family bootstrap for method comparisons. Undefined metrics remain null. "
            "Tool/memory checks inspect generated text or structured calls; no external action executes.",
            "",
            "Prompt-condition vectors may encode wording, task content or source formatting. This run "
            "does not establish a universal malicious-behavior direction. Repeated random seeds "
            "change control placement/direction, not the greedy baseline. Small pilots and template "
            "data do not support publication-level robustness or native benchmark claims.",
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines))
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    latex = [
        "% " + label,
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Method & ASR $\downarrow$ & BU $\uparrow$ & AU $\uparrow$ & Joint $\uparrow$ \\",
        r"\hline",
    ]
    for row in csv_rows:
        latex.append(
            row["method"].replace("_", r"\_")
            + " & "
            + " & ".join("--" if row[key] is None else f"{100 * row[key]:.1f}" for key in columns)
            + r" \\"
        )
    latex.extend([r"\hline", r"\end{tabular}"])
    (run_dir / "results_table.tex").write_text("\n".join(latex) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    points = {}
    for row in csv_rows:
        if row["attack_success_rate"] is None or row["benign_task_success"] is None:
            continue
        x, y = row["attack_success_rate"], row["benign_task_success"]
        points.setdefault((x, y), []).append(row["method"])
    for (x, y), names in points.items():
        ax.scatter(x, y)
        annotation = ", ".join(names) if len(names) <= 2 else f"{len(names)} methods (see table)"
        ax.annotate(annotation, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(
        xlabel="Attack success rate (lower is better)",
        ylabel="Benign task success",
        xlim=(-0.05, 1.05),
        ylim=(-0.05, 1.05),
        title=label,
    )
    fig.tight_layout()
    fig.savefig(run_dir / "security_utility.png", dpi=180)
    fig.savefig(run_dir / "security_utility.pdf")
    plt.close(fig)
