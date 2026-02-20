"""Automated report generator.

Produces a Markdown report with tables, figures, and statistical
summaries after training + evaluation finishes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def generate_report(
    results: dict[str, Any],
    output_dir: str | Path,
    experiment_name: str = "AMERS",
) -> Path:
    """Create a Markdown report summarising all results.

    Args:
        results: Nested dict from training/evaluation pipeline.
        output_dir: Directory to write the report.
        experiment_name: Title prefix.

    Returns:
        Path to the generated ``report.md``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"

    lines: list[str] = []

    lines.append(f"# {experiment_name} — Experiment Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # --- Overall metrics ---
    if "overall_metrics" in results:
        om = results["overall_metrics"]
        lines.append("## Overall Metrics\n")
        lines.append(f"- **Accuracy:** {om.get('accuracy', 'N/A'):.4f}")
        lines.append(f"- **F1 (macro):** {om.get('f1_macro', 'N/A'):.4f}")
        lines.append(f"- **F1 (weighted):** {om.get('f1_weighted', 'N/A'):.4f}")
        lines.append(f"- **Cohen's κ:** {om.get('kappa', 'N/A'):.4f}")
        lines.append("")

        if "report_str" in om:
            lines.append("### Classification Report\n")
            lines.append("```")
            lines.append(om["report_str"])
            lines.append("```\n")

    # --- LOSO cross-validation ---
    if "mean_metrics" in results and "std_metrics" in results:
        mm = results["mean_metrics"]
        sm = results["std_metrics"]
        lines.append("## LOSO Cross-Validation\n")
        lines.append("| Metric | Mean | Std |")
        lines.append("|--------|------|-----|")
        for k in mm:
            lines.append(f"| {k} | {mm[k]:.4f} | {sm[k]:.4f} |")
        lines.append("")

    # --- Ablation ---
    if "ablation" in results:
        from src.evaluation.ablation import ablation_summary_table
        lines.append("## Ablation Study\n")
        lines.append(ablation_summary_table(results["ablation"]))
        lines.append("")

    # --- RL training ---
    if "rl" in results:
        rl = results["rl"]
        lines.append("## RL Training Summary\n")
        if "val_acc" in rl:
            best_acc = max(rl["val_acc"])
            best_step = int(np.argmax(rl["val_acc"])) + 1
            lines.append(f"- **Best val accuracy:** {best_acc:.4f} (step {best_step})")
        if "aug_ratio" in rl:
            lines.append(f"- **Mean augmentation ratio:** {np.mean(rl['aug_ratio']):.3f}")
        lines.append("")

    # --- GAN training ---
    if "gan" in results:
        g = results["gan"]
        lines.append("## GAN Training Summary\n")
        if "g_loss" in g:
            lines.append(f"- **Final G loss:** {g['g_loss'][-1]:.4f}")
            lines.append(f"- **Final D loss:** {g['d_loss'][-1]:.4f}")
        lines.append("")

    # Write
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved to %s", report_path)

    # Also save raw JSON
    json_path = output_dir / "results.json"
    _save_json(results, json_path)

    return report_path


def _save_json(data: Any, path: Path) -> None:
    """Save results as JSON (convert numpy types)."""

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    path.write_text(json.dumps(data, indent=2, cls=NumpyEncoder), encoding="utf-8")
    logger.info("JSON results saved to %s", path)
