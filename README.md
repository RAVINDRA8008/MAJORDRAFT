# AMERS — Adaptive Multimodal Emotion Recognition System

An advanced emotion recognition system combining **EEG** and **Speech** modalities using **Conditional GANs** for data augmentation and **Reinforcement Learning (PPO)** for adaptive augmentation control.

## Architecture Overview

```
EEG (.dat) → Preprocessing → DE Features → cGAN Augmentation → EEG Encoder ──┐
                                                ↑ PPO RL Agent                │→ Late Fusion → Softmax → Emotion
Speech (.wav) → Preprocessing → MFCCs → CNN-LSTM Speech Encoder ─────────────┘
```

## Key Components

| Module | Description |
|--------|------------|
| **EEG Preprocessing** | Bandpass filtering, 1s epoch segmentation, Differential Entropy extraction |
| **Conditional GAN** | Generates synthetic EEG features conditioned on emotion labels |
| **PPO RL Agent** | Dynamically selects GAN augmentation ratio per training epoch |
| **CNN-LSTM Speech Encoder** | Extracts emotion embeddings from MFCC sequences |
| **Late Fusion Classifier** | Combines EEG + Speech embeddings for 4-class emotion classification |

## Emotions Classified

`Happy` | `Sad` | `Angry` | `Neutral`

## Datasets

- **DEAP** — 32 subjects, 40-channel EEG, 40 trials each
- **IEMOCAP** — 5 sessions, ~5,500 utterances with emotion labels

## Development Platform

- **Google Colab Pro** — GPU training (A100/V100/T4)
- **VS Code** — Remote IDE connected to Colab
- **Google Drive** — Persistent data and checkpoint storage

## Quick Start (Colab)

```python
# 1. Mount Drive & install deps
from google.colab import drive
drive.mount('/content/drive')
!pip install -q mne librosa soundfile gymnasium omegaconf

# 2. Add code to path
import sys, os
sys.path.insert(0, "/content/drive/MyDrive/AMERS/code")
os.chdir("/content/drive/MyDrive/AMERS/code")

# 3. Run full pipeline
!python scripts/train_full_pipeline.py --config /content/drive/MyDrive/AMERS/config/default.yaml
```

## Project Structure

```
src/
├── data/           # Data loading and preprocessing
├── models/         # Neural network architectures
├── rl/             # PPO agent and RL environment
├── training/       # Training loops for each phase
├── evaluation/     # Metrics and evaluation
└── utils/          # Config, paths, checkpointing, seeding
```

## Documentation

- [Product Requirements Document (PRD)](../PRD_Adaptive_Multimodal_Emotion_Recognition_System.md)
- [Development Guide](../DEVELOPMENT_GUIDE.md)

## License

Academic use only. DEAP and IEMOCAP datasets require separate license agreements.
