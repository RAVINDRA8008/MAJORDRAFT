"""Visualization utilities: t-SNE, loss curves, augmentation-ratio plots."""

from __future__ import annotations

import logging
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["Happy", "Sad", "Angry", "Neutral"]


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    title: str = "Loss Curve",
    save_path: str | None = None,
) -> None:
    """Plot training (and optionally validation) loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss")
    if val_losses is not None:
        ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved: %s", save_path)
    plt.show()
    plt.close(fig)


def plot_accuracy_curves(
    train_acc: list[float],
    val_acc: list[float],
    title: str = "Accuracy Curve",
    save_path: str | None = None,
) -> None:
    """Plot training and validation accuracy over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_acc, label="Train Acc")
    ax.plot(val_acc, label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str] = EMOTION_LABELS,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    """Render and optionally save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    ax.set_title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved: %s", save_path)
    plt.show()
    plt.close(fig)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str] = EMOTION_LABELS,
    title: str = "t-SNE Embedding Visualization",
    perplexity: float = 30.0,
    save_path: str | None = None,
) -> None:
    """2D t-SNE scatter plot of embeddings coloured by class label."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))
    for idx, name in enumerate(class_names):
        mask = labels == idx
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=name, alpha=0.6, s=10)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_augmentation_ratios(
    ratios: list[float],
    title: str = "RL Augmentation Ratio over Epochs",
    save_path: str | None = None,
) -> None:
    """Bar/line chart of the GAN augmentation ratio selected by the PPO agent."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ratios, marker="o", markersize=3, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Augmentation Ratio")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_gan_distributions(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    feature_idx: int = 0,
    title: str = "Real vs Synthetic Feature Distribution",
    save_path: str | None = None,
) -> None:
    """Overlay histograms of a single feature dimension for real vs synthetic."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(real_features[:, feature_idx], bins=60, alpha=0.5, label="Real", density=True)
    ax.hist(fake_features[:, feature_idx], bins=60, alpha=0.5, label="Synthetic", density=True)
    ax.set_xlabel(f"Feature [{feature_idx}]")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
