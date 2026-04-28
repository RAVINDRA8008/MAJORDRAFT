"""
Generate a diverse, final-draft figure pack (15 images) for AMERS report.
Focus: non-repetitive visuals (architecture, flow, protocol, outcomes, impact).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "docs" / "final_report_figures_v2"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#f8fafc",
        "axes.grid": True,
        "grid.alpha": 0.18,
        "font.size": 11,
        "figure.dpi": 190,
    }
)

C = {
    "blue": "#2563eb",
    "teal": "#0f766e",
    "orange": "#f59e0b",
    "green": "#16a34a",
    "red": "#dc2626",
    "slate": "#475569",
    "violet": "#7c3aed",
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=230)
    plt.close(fig)


# Final paper values
methods = ["No Aug", "Static GAN", "DRL-GAN", "DRL-GAN+Speech"]
sd = [82.2, 85.3, 87.1, 88.4]
si = [64.3, 67.8, 70.5, 72.1]
sd_f1 = [81.5, 84.7, 86.4, 87.8]
si_f1 = [62.8, 66.4, 69.1, 71.0]
sd_uar = [80.8, 83.9, 85.7, 87.2]
si_uar = [61.5, 65.2, 68.3, 70.2]


# 01 hero scorecard
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.axis("off")
ax.set_title("AMERS Final Outcome Scorecard", pad=10)
for i, (title, val, color) in enumerate([
    ("Subject-Dependent Accuracy", "88.4%", C["blue"]),
    ("Subject-Independent Accuracy", "72.1%", C["teal"]),
    ("Best Improvement over Baseline", "+6.2 pp", C["green"]),
    ("Training Overhead", "1.6x", C["orange"]),
]):
    x = 0.03 + i * 0.24
    rect = plt.Rectangle((x, 0.22), 0.21, 0.56, transform=ax.transAxes, facecolor=color, edgecolor="#0f172a", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + 0.105, 0.58, val, ha="center", va="center", fontsize=20, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(x + 0.105, 0.35, title, ha="center", va="center", fontsize=10, color="white", transform=ax.transAxes)
save(fig, "01_final_scorecard")


# 02 system architecture diagram
fig, ax = plt.subplots(figsize=(12, 6.7))
ax.axis("off")
ax.set_title("System Architecture (Final)", pad=14)

def block(x, y, w, h, txt, color):
    ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=color, edgecolor="#1e293b", linewidth=1.4))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10, color="white", fontweight="bold", transform=ax.transAxes)


block(0.03, 0.66, 0.23, 0.18, "DEAP EEG\nPreprocessing", C["blue"])
block(0.03, 0.40, 0.23, 0.18, "cGAN\nFeature Generation", C["teal"])
block(0.03, 0.14, 0.23, 0.18, "PPO Agent\nRatio Control", C["slate"])
block(0.40, 0.50, 0.24, 0.21, "EEG Classifier", C["orange"])
block(0.40, 0.19, 0.24, 0.21, "Speech CNN-LSTM\n(IEMOCAP)", C["green"])
block(0.75, 0.35, 0.21, 0.25, "Late Fusion\n4-Class Output", C["red"])

for a, b in [((0.26, 0.75), (0.40, 0.62)), ((0.26, 0.49), (0.40, 0.62)), ((0.26, 0.23), (0.40, 0.30)), ((0.64, 0.62), (0.75, 0.48)), ((0.64, 0.30), (0.75, 0.48))]:
    ax.annotate("", xy=b, xytext=a, xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"))
save(fig, "02_system_architecture")


# 03 training pipeline timeline
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.axis("off")
ax.set_title("Training Pipeline Timeline", pad=12)
steps = ["Data", "Features", "GAN", "PPO", "Train", "Validate", "Fuse"]
xs = np.linspace(0.07, 0.93, len(steps))
cols = [C["blue"], C["blue"], C["teal"], C["slate"], C["green"], C["orange"], C["red"]]
for i, (x, s) in enumerate(zip(xs, steps)):
    ax.add_patch(plt.Circle((x, 0.52), 0.055, transform=ax.transAxes, facecolor=cols[i], edgecolor="#0f172a", linewidth=1.2))
    ax.text(x, 0.52, s, ha="center", va="center", color="white", fontweight="bold", transform=ax.transAxes)
    if i < len(xs) - 1:
        ax.annotate("", xy=(xs[i+1]-0.06, 0.52), xytext=(x+0.06, 0.52), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"))
ax.text(0.07, 0.22, "Validation metrics continuously drive PPO for adaptive augmentation.", transform=ax.transAxes, color="#334155")
save(fig, "03_training_timeline")


# 04 protocol design (SD vs SI)
fig, ax = plt.subplots(figsize=(10.6, 5.6))
ax.axis("off")
ax.set_title("Evaluation Protocol Design", pad=12)
ax.add_patch(plt.Rectangle((0.05, 0.56), 0.40, 0.30, transform=ax.transAxes, facecolor=C["blue"], alpha=0.9))
ax.add_patch(plt.Rectangle((0.55, 0.56), 0.40, 0.30, transform=ax.transAxes, facecolor=C["teal"], alpha=0.9))
ax.text(0.25, 0.71, "Subject-Dependent\n5-Fold per Subject", color="white", ha="center", va="center", fontweight="bold", transform=ax.transAxes)
ax.text(0.75, 0.71, "Subject-Independent\nLOSO", color="white", ha="center", va="center", fontweight="bold", transform=ax.transAxes)
ax.add_patch(plt.Rectangle((0.07, 0.18), 0.36, 0.23, transform=ax.transAxes, facecolor="#e2e8f0", edgecolor="#64748b"))
ax.add_patch(plt.Rectangle((0.57, 0.18), 0.36, 0.23, transform=ax.transAxes, facecolor="#e2e8f0", edgecolor="#64748b"))
ax.text(0.25, 0.295, "Higher score:\n88.4%", ha="center", va="center", fontweight="bold", color="#0f172a", transform=ax.transAxes)
ax.text(0.75, 0.295, "Harder setting:\n72.1%", ha="center", va="center", fontweight="bold", color="#0f172a", transform=ax.transAxes)
save(fig, "04_protocol_design")


# 05 compact metric matrix heatmap
fig, ax = plt.subplots(figsize=(9.4, 5.8))
m = np.array([
    [82.2, 81.5, 80.8, 64.3, 62.8, 61.5],
    [85.3, 84.7, 83.9, 67.8, 66.4, 65.2],
    [87.1, 86.4, 85.7, 70.5, 69.1, 68.3],
    [88.4, 87.8, 87.2, 72.1, 71.0, 70.2],
])
im = ax.imshow(m, cmap="YlGnBu", aspect="auto")
ax.set_yticks(np.arange(len(methods)))
ax.set_yticklabels(methods)
ax.set_xticks(np.arange(6))
ax.set_xticklabels(["SD Acc", "SD F1", "SD UAR", "SI Acc", "SI F1", "SI UAR"], rotation=25, ha="right")
for i in range(4):
    for j in range(6):
        ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=8)
ax.set_title("Final Results Matrix")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
save(fig, "05_results_matrix")


# 06 generalization slope graph
fig, ax = plt.subplots(figsize=(9.2, 5.4))
for i, name in enumerate(methods):
    ax.plot([0, 1], [sd[i], si[i]], marker="o", linewidth=2.2, label=name)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Subject-Dependent", "Subject-Independent"])
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(60, 92)
ax.set_title("Generalization Shift Across Methods")
ax.legend(loc="lower left")
save(fig, "06_generalization_slope")


# 07 ablation waterfall
fig, ax = plt.subplots(figsize=(9.2, 5.0))
labels = ["Static GAN", "+ Adaptive PPO", "+ Speech Fusion", "Final"]
vals = [67.8, 70.5, 72.1, 72.1]
inc = [67.8, 2.7, 1.6, 0]
base = np.cumsum([0] + inc[:-1])
for i in range(len(labels)):
    color = C["orange"] if i == 0 else (C["blue"] if i == 1 else (C["teal"] if i == 2 else C["green"]))
    ax.bar(i, inc[i] if i < 3 else vals[-1], bottom=base[i] if i < 3 else 0, color=color, width=0.58)
    txt = vals[i] if i in [0, 1, 2] else vals[-1]
    ax.text(i, (base[i] + (inc[i] if i < 3 else vals[-1])) + 0.4, f"{txt:.1f}%", ha="center", fontsize=9)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 78)
ax.set_title("Ablation Contribution (Subject-Independent)")
save(fig, "07_ablation_waterfall")


# 08 modality contribution donut
fig, ax = plt.subplots(figsize=(7.0, 5.7))
parts = ["EEG Encoder 2.5M", "Speech Encoder 2.9M", "Fusion/Heads 1.9M"]
sizes = [2.5, 2.9, 1.9]
wedges, _ = ax.pie(sizes, colors=[C["blue"], C["orange"], C["teal"]], startangle=110, wedgeprops=dict(width=0.40, edgecolor="white"))
ax.text(0, 0, "7.3M\nTotal", ha="center", va="center", fontweight="bold")
ax.legend(wedges, parts, loc="center left", bbox_to_anchor=(1.0, 0.5))
ax.set_title("Model Parameter Composition")
save(fig, "08_parameter_donut")


# 09 PPO state-action-reward infographic
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.axis("off")
ax.set_title("PPO Control Logic", pad=12)
boxes = [
    (0.04, "State\nLosses, variance, diversity, progress", C["blue"]),
    (0.36, "Action\nChoose augmentation ratio\n{0,0.25,0.5,0.75,1.0}", C["teal"]),
    (0.68, "Reward\nImprove val loss + stability + diversity", C["green"]),
]
for x, t, col in boxes:
    ax.add_patch(plt.Rectangle((x, 0.32), 0.26, 0.36, transform=ax.transAxes, facecolor=col, edgecolor="#0f172a", linewidth=1.3))
    ax.text(x + 0.13, 0.50, t, ha="center", va="center", color="white", fontweight="bold", transform=ax.transAxes)
ax.annotate("", xy=(0.36, 0.50), xytext=(0.30, 0.50), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"))
ax.annotate("", xy=(0.68, 0.50), xytext=(0.62, 0.50), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155"))
save(fig, "09_ppo_logic")


# 10 augmentation schedule curve
fig, ax = plt.subplots(figsize=(9.6, 4.8))
epochs = [1, 30, 70, 100]
ratio = [0.90, 0.85, 0.55, 0.30]
ax.plot(epochs, ratio, marker="o", linewidth=2.6, color=C["teal"])
ax.fill_between(epochs, ratio, color=C["teal"], alpha=0.15)
for e, r in zip(epochs, ratio):
    ax.text(e, r + 0.03, f"{r:.2f}", ha="center", fontsize=9)
ax.set_xlabel("Epoch")
ax.set_ylabel("Synthetic/Real Ratio")
ax.set_ylim(0, 1.05)
ax.set_title("Adaptive Augmentation Schedule")
save(fig, "10_augmentation_schedule")


# 11 class mapping chart DEAP to IEMOCAP labels
fig, ax = plt.subplots(figsize=(10.8, 5.4))
ax.axis("off")
ax.set_title("Cross-Dataset Emotion Label Mapping", pad=10)
left = ["HVHA", "HVLA", "LVHA", "LVLA"]
right = ["Happiness/Excitement", "Contentment", "Anger/Frustration", "Sadness"]
for i in range(4):
    y = 0.82 - i * 0.18
    ax.add_patch(plt.Rectangle((0.07, y - 0.06), 0.22, 0.10, transform=ax.transAxes, facecolor=C["blue"], edgecolor="#0f172a"))
    ax.text(0.18, y - 0.01, left[i], ha="center", va="center", color="white", fontweight="bold", transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.71, y - 0.06), 0.22, 0.10, transform=ax.transAxes, facecolor=C["teal"], edgecolor="#0f172a"))
    ax.text(0.82, y - 0.01, right[i], ha="center", va="center", color="white", fontweight="bold", transform=ax.transAxes)
    ax.annotate("", xy=(0.71, y - 0.01), xytext=(0.29, y - 0.01), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.6, color="#334155"))
save(fig, "11_label_mapping")


# 12 final hyperparameter card (replacement for old weak pic)
fig, ax = plt.subplots(figsize=(11.2, 6.0))
ax.axis("off")
ax.set_title("Final Hyperparameter Card", pad=12)
rows = [
    ("GAN", "Adam lr=2e-4, batch=128, epochs=200"),
    ("GAN Stability", "Wasserstein + GP=10, 5 critic updates"),
    ("Classifier", "MLP 256-128-64, dropout=0.5, wd=1e-4"),
    ("Classifier Train", "Adam lr=1e-3, early stopping=20"),
    ("PPO Network", "2x64 dense layers, lr=3e-4"),
    ("PPO Settings", "gamma=0.99, lambda_GAE=0.95, epsilon=0.2"),
    ("PPO Updates", "every 10 epochs, 4 minibatch epochs"),
]
for i, (k, v) in enumerate(rows):
    y = 0.88 - i * 0.11
    ax.add_patch(plt.Rectangle((0.05, y - 0.045), 0.22, 0.075, transform=ax.transAxes, facecolor=C["slate"], alpha=0.95))
    ax.text(0.16, y - 0.008, k, ha="center", va="center", color="white", fontweight="bold", transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.29, y - 0.045), 0.66, 0.075, transform=ax.transAxes, facecolor="#e2e8f0", edgecolor="#94a3b8"))
    ax.text(0.31, y - 0.008, v, ha="left", va="center", color="#0f172a", transform=ax.transAxes)
save(fig, "12_hyperparameter_card")


# 13 practical application map
fig, ax = plt.subplots(figsize=(10.8, 5.8))
ax.axis("off")
ax.set_title("Practical Application Map", pad=12)
center = plt.Circle((0.5, 0.52), 0.12, transform=ax.transAxes, facecolor=C["blue"], edgecolor="#0f172a", linewidth=1.2)
ax.add_patch(center)
ax.text(0.5, 0.52, "Final\nAMERS", ha="center", va="center", color="white", fontweight="bold", transform=ax.transAxes)
apps = [
    (0.20, 0.80, "Mental Health\nMonitoring"),
    (0.80, 0.80, "Adaptive\nHCI"),
    (0.20, 0.24, "Affect-aware\nRecommendation"),
    (0.80, 0.24, "Real-time\nCognitive Tracking"),
]
for x, y, t in apps:
    ax.add_patch(plt.Rectangle((x - 0.11, y - 0.07), 0.22, 0.12, transform=ax.transAxes, facecolor=C["teal"], edgecolor="#0f172a"))
    ax.text(x, y - 0.01, t, ha="center", va="center", color="white", fontweight="bold", transform=ax.transAxes)
    ax.annotate("", xy=(x, y - 0.08 if y > 0.5 else y + 0.06), xytext=(0.5, 0.52), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.4, color="#334155"))
save(fig, "13_application_map")


# 14 limitations and future scope panel
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.axis("off")
ax.set_title("Limitations and Future Scope", pad=12)
ax.add_patch(plt.Rectangle((0.05, 0.15), 0.42, 0.72, transform=ax.transAxes, facecolor="#fee2e2", edgecolor="#ef4444", linewidth=1.4))
ax.add_patch(plt.Rectangle((0.53, 0.15), 0.42, 0.72, transform=ax.transAxes, facecolor="#dcfce7", edgecolor="#22c55e", linewidth=1.4))
ax.text(0.26, 0.80, "Current Limitations", ha="center", va="center", fontweight="bold", color="#991b1b", transform=ax.transAxes)
ax.text(0.74, 0.80, "Future Scope", ha="center", va="center", fontweight="bold", color="#166534", transform=ax.transAxes)
left_items = ["EEG-speech not instance aligned", "No diffusion baseline comparison", "Extra training cost from PPO loop"]
right_items = ["Add diffusion augmentation baseline", "Domain adaptation for broader generalization", "Optimize PPO update efficiency"]
for i, t in enumerate(left_items):
    ax.text(0.08, 0.66 - i * 0.18, f"• {t}", color="#7f1d1d", transform=ax.transAxes)
for i, t in enumerate(right_items):
    ax.text(0.56, 0.66 - i * 0.18, f"• {t}", color="#14532d", transform=ax.transAxes)
save(fig, "14_limitations_future_scope")


# 15 final summary infographic
fig, ax = plt.subplots(figsize=(11.4, 6.0))
ax.axis("off")
ax.set_title("Final Draft Summary", pad=12)
summary = [
    ("Problem", "Data scarcity + inter-subject variability in EEG emotion recognition"),
    ("Core Method", "cGAN augmentation + PPO adaptive control + EEG-speech late fusion"),
    ("Best Result", "88.4% (subject-dependent), 72.1% (subject-independent)"),
    ("Key Gains", "+6.2 pp over no augmentation, reduced SD-SI gap"),
    ("Impact", "Generalizable framework for affective computing applications"),
]
for i, (k, v) in enumerate(summary):
    y = 0.86 - i * 0.16
    ax.add_patch(plt.Circle((0.07, y), 0.025, transform=ax.transAxes, facecolor=C["violet"], edgecolor="#0f172a"))
    ax.text(0.07, y, str(i + 1), ha="center", va="center", color="white", fontweight="bold", transform=ax.transAxes)
    ax.text(0.12, y + 0.02, k, color="#1e1b4b", fontweight="bold", transform=ax.transAxes)
    ax.text(0.12, y - 0.03, v, color="#334155", transform=ax.transAxes)
save(fig, "15_final_summary_infographic")


print("Generated 15 diverse report figures in", OUT)
