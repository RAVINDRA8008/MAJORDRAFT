"""
Generate clean, final-outcome-only figures for the AMERS major project report.

Output folder:
    amers/docs/final_report_figures
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "docs" / "final_report_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#f7f8fa",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "font.size": 11,
        "figure.dpi": 180,
    }
)

C = {
    "blue": "#2563eb",
    "teal": "#0d9488",
    "orange": "#f59e0b",
    "green": "#16a34a",
    "slate": "#64748b",
    "red": "#dc2626",
}


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"saved: {path}")


# Source values from paper.tex (final results table and discussion)
methods = ["No Augmentation", "Static GAN", "DRL-GAN", "DRL-GAN + Speech"]

sd_acc = np.array([82.2, 85.3, 87.1, 88.4])
sd_f1 = np.array([81.5, 84.7, 86.4, 87.8])
sd_uar = np.array([80.8, 83.9, 85.7, 87.2])

si_acc = np.array([64.3, 67.8, 70.5, 72.1])
si_f1 = np.array([62.8, 66.4, 69.1, 71.0])
si_uar = np.array([61.5, 65.2, 68.3, 70.2])


# 01: Final headline scores
fig, ax = plt.subplots(figsize=(8.5, 4.8))
labels = ["Subject-Dependent", "Subject-Independent"]
vals = [88.4, 72.1]
bars = ax.bar(labels, vals, color=[C["blue"], C["teal"]], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.9, f"{v:.1f}%", ha="center", fontweight="bold")
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Final Outcome Accuracy")
save(fig, "01_final_outcome_accuracy")


# 02: Final model metrics by protocol
fig, ax = plt.subplots(figsize=(9.4, 5.2))
metric_names = ["Accuracy", "F1", "UAR"]
x = np.arange(len(metric_names))
width = 0.34
sd_vals = [88.4, 87.8, 87.2]
si_vals = [72.1, 71.0, 70.2]
b1 = ax.bar(x - width / 2, sd_vals, width, label="Subject-Dependent", color=C["blue"])
b2 = ax.bar(x + width / 2, si_vals, width, label="Subject-Independent", color=C["teal"])
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 100)
ax.set_title("Final Model Metrics Across Protocols")
ax.legend()
save(fig, "02_final_metrics_sd_vs_si")


# 03: Subject-dependent comparison across methods
fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(len(methods))
width = 0.24
ax.bar(x - width, sd_acc, width, label="Accuracy", color=C["blue"])
ax.bar(x, sd_f1, width, label="F1", color=C["orange"])
ax.bar(x + width, sd_uar, width, label="UAR", color=C["green"])
for i, v in enumerate(sd_acc):
    ax.text(i - width, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=10)
ax.set_ylim(55, 95)
ax.set_ylabel("Score (%)")
ax.set_title("Subject-Dependent Performance by Method")
ax.legend()
save(fig, "03_sd_performance_by_method")


# 04: Subject-independent comparison across methods
fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(len(methods))
width = 0.24
ax.bar(x - width, si_acc, width, label="Accuracy", color=C["blue"])
ax.bar(x, si_f1, width, label="F1", color=C["orange"])
ax.bar(x + width, si_uar, width, label="UAR", color=C["green"])
for i, v in enumerate(si_acc):
    ax.text(i - width, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=10)
ax.set_ylim(55, 80)
ax.set_ylabel("Score (%)")
ax.set_title("Subject-Independent Performance by Method")
ax.legend()
save(fig, "04_si_performance_by_method")


# 05: Generalization gap (SD-SI) per method
fig, ax = plt.subplots(figsize=(8.5, 4.9))
gaps = sd_acc - si_acc
bars = ax.bar(methods, gaps, color=[C["slate"], C["orange"], C["teal"], C["green"]])
for b, g in zip(bars, gaps):
    ax.text(b.get_x() + b.get_width() / 2, g + 0.25, f"{g:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("Gap (%)")
ax.set_ylim(14, 19)
ax.set_title("Generalization Gap (Subject-Dependent minus Subject-Independent Accuracy)")
save(fig, "05_generalization_gap")


# 06: Improvement over no augmentation
fig, ax = plt.subplots(figsize=(8.8, 5.1))
delta_sd = sd_acc - sd_acc[0]
delta_si = si_acc - si_acc[0]
labels = ["No Aug", "Static GAN", "DRL-GAN", "DRL-GAN + Speech"]
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width / 2, delta_sd, width, color=C["blue"], label="Subject-Dependent")
ax.bar(x + width / 2, delta_si, width, color=C["teal"], label="Subject-Independent")
for i in range(len(labels)):
    ax.text(x[i] - width / 2, delta_sd[i] + 0.15, f"{delta_sd[i]:.1f}", ha="center", fontsize=8)
    ax.text(x[i] + width / 2, delta_si[i] + 0.15, f"{delta_si[i]:.1f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Improvement (pp)")
ax.set_ylim(0, 9)
ax.set_title("Improvement Relative to No Augmentation")
ax.legend()
save(fig, "06_improvement_over_no_aug")


# 07: Multimodal fusion gain (DRL-GAN+Speech vs DRL-GAN)
fig, ax = plt.subplots(figsize=(7.8, 4.8))
metrics = ["Accuracy", "F1", "UAR"]
gain_sd = np.array([88.4, 87.8, 87.2]) - np.array([87.1, 86.4, 85.7])
gain_si = np.array([72.1, 71.0, 70.2]) - np.array([70.5, 69.1, 68.3])
x = np.arange(len(metrics))
width = 0.35
b1 = ax.bar(x - width / 2, gain_sd, width, color=C["blue"], label="Subject-Dependent")
b2 = ax.bar(x + width / 2, gain_si, width, color=C["teal"], label="Subject-Independent")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f"+{b.get_height():.1f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Gain (pp)")
ax.set_ylim(0, 2.1)
ax.set_title("Multimodal Fusion Gain over EEG-only DRL-GAN")
ax.legend()
save(fig, "07_multimodal_fusion_gain")


# 08: Final ablation (from text)
fig, ax = plt.subplots(figsize=(8.6, 4.9))
abl_labels = ["Static GAN\n(50%)", "Adaptive DRL-GAN\n(EEG only)", "Adaptive DRL-GAN\n+ Speech"]
abl_vals = [67.8, 70.5, 72.1]
bars = ax.bar(abl_labels, abl_vals, color=[C["orange"], C["blue"], C["green"]], width=0.55)
for b, v in zip(bars, abl_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(64, 75)
ax.set_title("Ablation Outcomes (Subject-Independent)")
save(fig, "08_ablation_subject_independent")


# 09: PPO augmentation schedule (from narrative in paper)
fig, ax = plt.subplots(figsize=(9.4, 4.8))
epochs = np.array([1, 30, 70, 100])
ratios = np.array([0.90, 0.85, 0.55, 0.30])
ax.plot(epochs, ratios, marker="o", linewidth=2.5, color=C["teal"])
ax.fill_between(epochs, ratios, color=C["teal"], alpha=0.15)
for e, r in zip(epochs, ratios):
    ax.text(e, r + 0.03, f"{r:.2f}", ha="center", fontsize=9)
ax.set_xlim(0, 102)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Epoch")
ax.set_ylabel("Synthetic-to-Real Ratio")
ax.set_title("Adaptive PPO Augmentation Trend During Training")
save(fig, "09_ppo_augmentation_schedule")


# 10: Model footprint and training overhead
fig, ax = plt.subplots(figsize=(8.8, 5.1))
parts = ["EEG Encoder", "Speech Encoder", "Fusion + Heads"]
params = [2.5, 2.9, 1.9]
bars = ax.bar(parts, params, color=[C["blue"], C["orange"], C["teal"]], width=0.58)
for b, v in zip(bars, params):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.1f}M", ha="center", fontweight="bold")
ax.text(1.0, 3.45, "Total parameters: 7.3M", ha="center", fontsize=11, color="#334155")
ax.text(1.0, 3.25, "Training overhead vs static GAN: 1.6x", ha="center", fontsize=10, color="#334155")
ax.set_ylim(0, 3.8)
ax.set_ylabel("Parameters (Millions)")
ax.set_title("Model Size Breakdown and Training Cost")
save(fig, "10_model_size_and_cost")


# 11: Key hyperparameters used in final model
fig, ax = plt.subplots(figsize=(10.5, 6.0))
ax.axis("off")
ax.set_title("Final Training Hyperparameters", pad=14)
rows = [
    ("GAN optimizer", "Adam, lr=2e-4, batch=128, epochs=200"),
    ("GAN stability", "Wasserstein loss, gradient penalty=10, 5 critic updates"),
    ("Classifier", "MLP 256-128-64, dropout=0.5, Adam lr=1e-3"),
    ("Regularization", "weight decay=1e-4, early stopping=20 epochs"),
    ("PPO policy net", "2 dense layers x 64, lr=3e-4"),
    ("PPO settings", "gamma=0.99, lambda_GAE=0.95, epsilon=0.2"),
    ("PPO update", "policy update every 10 epochs, 4 minibatch epochs"),
]
for i, (k, v) in enumerate(rows):
    y = 0.92 - i * 0.12
    ax.text(0.04, y, k, fontsize=11, fontweight="bold", color="#0f172a", transform=ax.transAxes)
    ax.text(0.36, y, v, fontsize=11, color="#334155", transform=ax.transAxes)
save(fig, "11_final_hyperparameters")


# 12: Practical impact map from conclusion
fig, ax = plt.subplots(figsize=(10.5, 5.8))
ax.axis("off")
ax.set_title("Practical Impact of Final System", pad=14)
apps = [
    "Mental health monitoring",
    "Adaptive human-computer interaction",
    "Affect-aware recommendation",
    "Real-time cognitive state tracking",
    "Safety-critical operator support",
]
for i, t in enumerate(apps):
    y = 0.86 - i * 0.15
    ax.add_patch(plt.Rectangle((0.05, y - 0.045), 0.03, 0.06, transform=ax.transAxes, color=C["teal"], alpha=0.85))
    ax.text(0.10, y, t, fontsize=12, color="#0f172a", va="center", transform=ax.transAxes)
ax.text(
    0.05,
    0.08,
    "These application pathways are derived from the final conclusion section in the paper.",
    fontsize=10,
    color="#475569",
    transform=ax.transAxes,
)
save(fig, "12_practical_impact_map")

print("done")


# 13: System architecture block diagram (report-friendly)
fig, ax = plt.subplots(figsize=(12, 6.8))
ax.axis("off")
ax.set_title("Final System Architecture Overview", pad=16)

def box(x, y, w, h, text, color):
    rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=color, edgecolor="#1e293b", linewidth=1.4)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color="white", fontweight="bold", transform=ax.transAxes)


box(0.03, 0.70, 0.23, 0.16, "DEAP EEG\nPreprocessing + DE Features", C["blue"])
box(0.03, 0.40, 0.23, 0.16, "Conditional GAN\nFeature Synthesis", C["teal"])
box(0.03, 0.10, 0.23, 0.16, "PPO Controller\nAdaptive Ratio Selection", C["slate"])

box(0.39, 0.55, 0.23, 0.20, "EEG Branch\nClassifier Backbone", C["orange"])
box(0.39, 0.22, 0.23, 0.20, "Speech Branch\nCNN-LSTM Embeddings", C["green"])

box(0.75, 0.40, 0.22, 0.22, "Late Fusion\nFinal 4-Class Output", C["red"])

ax.annotate("", xy=(0.39, 0.66), xytext=(0.26, 0.78), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"))
ax.annotate("", xy=(0.39, 0.66), xytext=(0.26, 0.48), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"))
ax.annotate("", xy=(0.39, 0.32), xytext=(0.26, 0.18), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"))
ax.annotate("", xy=(0.75, 0.51), xytext=(0.62, 0.64), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"))
ax.annotate("", xy=(0.75, 0.51), xytext=(0.62, 0.32), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"))

ax.text(0.02, 0.95, "EEG Stream", fontsize=10, color="#334155", transform=ax.transAxes)
ax.text(0.40, 0.95, "Multimodal Representation", fontsize=10, color="#334155", transform=ax.transAxes)
ax.text(0.77, 0.95, "Decision", fontsize=10, color="#334155", transform=ax.transAxes)

save(fig, "13_system_architecture_overview")


# 14: End-to-end training workflow timeline
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.axis("off")
ax.set_title("End-to-End Training Workflow", pad=14)

steps = [
    "EEG + Speech\nData Intake",
    "Feature\nExtraction",
    "GAN\nAugmentation",
    "PPO Ratio\nUpdate",
    "Model\nTraining",
    "Validation\nMetrics",
    "Late Fusion\nInference",
]

x_positions = np.linspace(0.06, 0.94, len(steps))
for i, (x, s) in enumerate(zip(x_positions, steps)):
    ax.add_patch(plt.Circle((x, 0.52), 0.055, transform=ax.transAxes, facecolor=[C["blue"], C["teal"], C["orange"], C["slate"], C["green"], C["blue"], C["red"]][i], edgecolor="#1e293b", linewidth=1.2))
    ax.text(x, 0.52, s, ha="center", va="center", fontsize=9, color="white", fontweight="bold", transform=ax.transAxes)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x_positions[i + 1] - 0.06, 0.52), xytext=(x + 0.06, 0.52), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.6, color="#475569"))

ax.text(0.06, 0.20, "Adaptive loop: Validation metrics feed PPO to regulate synthetic-to-real ratio.", fontsize=10, color="#334155", transform=ax.transAxes)
save(fig, "14_training_workflow_timeline")


# 15: Compact results heatmap panel
fig, ax = plt.subplots(figsize=(9.2, 5.6))
matrix = np.array([
    [82.2, 81.5, 80.8, 64.3, 62.8, 61.5],
    [85.3, 84.7, 83.9, 67.8, 66.4, 65.2],
    [87.1, 86.4, 85.7, 70.5, 69.1, 68.3],
    [88.4, 87.8, 87.2, 72.1, 71.0, 70.2],
])

im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
ax.set_yticks(np.arange(4))
ax.set_yticklabels(methods)
ax.set_xticks(np.arange(6))
ax.set_xticklabels(["SD Acc", "SD F1", "SD UAR", "SI Acc", "SI F1", "SI UAR"], rotation=25, ha="right")
ax.set_title("Final Results Matrix (All Metrics)")

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8, color="#0f172a")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Score (%)")

save(fig, "15_results_heatmap_panel")


print("done")
