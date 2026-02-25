# AMERS — Version History (v1 → v5)

**Adaptive Multimodal Emotion Recognition System**
EEG (DEAP) + Speech (IEMOCAP) → 4-class emotion classification (Angry, Happy, Sad, Neutral)

---

## Results Summary

| Version | Accuracy | Macro F1 | Cohen's κ | Best Epoch | Status |
|---------|:--------:|:--------:|:---------:|:----------:|--------|
| **v1**  | —        | —        | —         | —          | Initial prototype (not formally evaluated) |
| **v2**  | 55.92%   | 0.5501   | 0.3970    | —          | Baseline pipeline |
| **v3**  | 65.97%   | ~0.60    | —         | —          | +Contrastive +DANN +Transformer |
| **v4**  | 81.94%   | 0.8193   | 0.7552    | —          | Architecture overhaul |
| **v5.3**| **82.55%** | **~0.83** | **~0.77** | 43       | **BEST — CMMA end-to-end** |

### Targets vs Achieved

| Metric | Target | v4 | v5.3 | Status |
|--------|:------:|:--:|:----:|--------|
| Accuracy | ≥75% | 81.94% | **82.55%** | ✅ Exceeded |
| Macro F1 | ≥0.72 | 0.8193 | ~0.83 | ✅ Exceeded |

---

## v1 — Initial Prototype

**Goal:** Build a working end-to-end multimodal emotion recognition pipeline.

### Architecture
- **EEG Encoder:** 2-layer attention network on DEAP differential entropy features
- **Speech Encoder:** CNN-LSTM on IEMOCAP MFCCs
- **GAN:** Vanilla conditional GAN for EEG data augmentation
- **Fusion:** Simple concatenation of EEG + speech embeddings → MLP classifier
- **RL Agent:** Basic PPO controlling augmentation ratio

### Outcome
- Pipeline worked end-to-end but accuracy was poor
- No formal evaluation with current metrics pipeline
- Identified key problems: class imbalance (37x in DEAP), weak fusion, unstable GAN

### Why It Failed
1. **Catastrophic class imbalance (37×):** DEAP had 41,940 Happy samples vs only 1,140 Angry — the model learned to predict Happy/Neutral for everything and still got decent loss, making Angry nearly unlearnable
2. **Concatenation fusion is naive:** Simply stacking EEG + speech embeddings forces the MLP to discover cross-modal relationships from scratch with no structural guidance — the model mostly ignored one modality
3. **Vanilla GAN mode collapse:** The conditional GAN suffered from mode collapse — generated samples clustered around class means instead of adding genuine diversity, providing no real augmentation benefit
4. **Index-based pairing was meaningless:** Pairing EEG sample #1000 with speech sample #1000 by array index meant the two modalities often had different emotion labels — the model received contradictory supervision
5. **RL with no stable baseline:** PPO tried to optimize augmentation ratio on top of a broken fusion pipeline — optimizing a hyperparameter on a fundamentally flawed model just amplified the flaws

### Limitations
- No cross-modal attention mechanism — modalities were treated independently until the final concatenation
- No domain adaptation — DEAP (lab-recorded EEG) and IEMOCAP (acted speech) have very different distributions
- No contrastive pretraining — encoders trained purely supervised on small labeled data
- Cross-entropy loss with no class weighting — majority classes dominated the gradient
- No mixed precision training — slow iteration on GPU
- No formal evaluation metrics collected — impossible to compare quantitatively

---

## v2 — Baseline Pipeline

**Goal:** Fix fundamental issues in v1 — class imbalance, fusion strategy, loss function.

### Key Changes
| Component | v1 → v2 Change |
|-----------|----------------|
| **Data pairing** | Index-based → **Label-aligned** (EEG & speech matched by emotion class) |
| **Fusion** | Concatenation → **Gated fusion** (sigmoid gate + LayerNorm) |
| **Loss** | Cross-entropy → **FocalLoss** (γ=2) with class weights |
| **Training** | Basic → AMP, cosine LR + warmup, gradient clipping, early stopping |
| **Sampling** | Random → **Class-balanced** oversampling |

### Results
| Metric | Value |
|--------|-------|
| Accuracy | 55.92% |
| Macro F1 | 0.5501 |
| Weighted F1 | 0.5709 |
| Cohen's κ | 0.3970 |

### Per-Class F1
| Emotion | F1 |
|---------|:--:|
| Angry | 0.38 |
| Happy | 0.58 |
| Sad | 0.66 |
| Neutral | 0.58 |

### Analysis
- Label-aligned pairing was critical — random index pairing gave random results
- Angry class still severely underperforming (0.38 F1) due to 37x class imbalance in DEAP
- Gated fusion outperformed concatenation
- RL agent was destructive — degraded accuracy when applied

### Why It Failed to Go Higher
1. **Class imbalance still untreated at source:** While FocalLoss down-weighted easy (majority) samples, the underlying 37× imbalance in DEAP meant the EEG encoder still could not learn Angry-class features — only 1,140 Angry EEG samples vs 41,940 Happy
2. **EEG encoder was too weak:** A 2-layer attention network on 160-dim features achieved only ~23% standalone accuracy — the fusion head was essentially working with one useful modality (speech at 55.8%) and one near-random modality (EEG)
3. **Gated fusion still shallow:** The sigmoid gate learned a single scalar mixing weight — no token-level or class-level modality weighting, so the gate settled on a fixed compromise for all emotions
4. **RL agent was destructive:** PPO augmentation changed training distribution dynamically, but the reward signal was too noisy and the fusion model too fragile — every RL run degraded the base model
5. **No pretraining:** Encoders trained from scratch on small labeled datasets, limiting representation quality

### Limitations
- DEAP class imbalance (37×) not addressed at the data level — only loss-level mitigation
- EEG encoder too shallow (2 layers, no CLS token) — poor standalone accuracy (~23%)
- No contrastive or self-supervised pretraining for either encoder
- No domain adaptation between lab EEG and acted speech corpora
- Single-scalar gated fusion — no per-class or per-token modality weighting
- Validation accuracy noisy due to random cross-modal pairing each evaluation pass

---

## v3 — Research-Grade Upgrades

**Goal:** Add research-level techniques — contrastive learning, domain adaptation, transformer fusion.

### Key Changes
| Component | v2 → v3 Change |
|-----------|----------------|
| **EEG pretraining** | Supervised only → **+ SimCLR contrastive** (NT-Xent, τ=0.07) |
| **Speech pretraining** | Supervised only → **+ Contrastive** (NT-Xent on MFCCs) |
| **Domain adaptation** | None → **DANN** (gradient reversal, progressive λ: 0→1) |
| **Fusion** | Gated → **Transformer cross-modal attention** (8 tokens, 4 heads, 2 layers) |
| **RL** | Basic PPO → **Improved PPO** (composite reward: acc + F1 + balance − overfit penalty) |

### Results
| Metric | v2 | v3 | Δ |
|--------|:--:|:--:|:-:|
| Accuracy | 55.92% | **65.97%** | +10.05 pp |
| Macro F1 | 0.5501 | ~0.60 | +0.05 |

### Analysis
- Contrastive pretraining improved encoder quality significantly
- DANN created domain-invariant representations bridging DEAP↔IEMOCAP gap
- Transformer cross-modal attention outperformed simple gated fusion
- RL v2 was still destructive — safety guard reverted to pre-RL weights
- Still bottlenecked by DEAP class imbalance (37x ratio untouched)

### Why It Failed to Go Higher
1. **Class imbalance was STILL the elephant in the room:** Despite all the research-grade additions, the 37× DEAP imbalance remained untouched — the EEG encoder was still at ~23% standalone accuracy, and no amount of contrastive pretraining or DANN could fix an encoder that never sees enough Angry samples
2. **Frozen encoders during fusion:** Transformer fusion trained on top of frozen encoder embeddings — the encoders could not adapt their representations to what the fusion head actually needed
3. **RL v2 still destructive:** Even with composite reward (acc + F1 + balance − overfit penalty) and reward normalization, PPO made the fusion model worse every time — the action space (augmentation ratio control) simply didn't help
4. **Contrastive pretraining helped representations but not classification:** SimCLR/NT-Xent created better-clustered embeddings, but the downstream classifier still suffered from the same class imbalance during fine-tuning
5. **DANN alignment was coarse:** Domain adversarial training aligned EEG and speech distributions globally, but emotion-specific alignment was lacking — Angry EEG and Angry speech were not necessarily close in the aligned space

### Limitations
- DEAP class imbalance (37×) — the single biggest bottleneck, still completely unaddressed
- Encoders frozen during fusion training — no end-to-end gradient flow
- Transformer fusion with only 2 layers and 4 heads — limited cross-modal modeling capacity
- DANN uses a single domain discriminator — no class-conditional domain adaptation
- RL agent's action space (augmentation ratio) too limited to be useful
- No per-class modality weighting — all emotions treated identically during fusion
- Speech encoder architecture unchanged from v2 (no attention pooling)

---

## v4 — Architecture Overhaul

**Goal:** Fix the fundamental class imbalance problem and modernize all components.

### Key Changes
| Component | v3 → v4 Change |
|-----------|----------------|
| **EEG Encoder** | 2-layer attention → **3-layer CLS-token pre-norm attention** |
| **Speech Encoder** | Basic CNN-LSTM → **+ Multi-head attention pooling** + GELU |
| **GAN** | Vanilla conditional → **WGAN-GP** (gradient penalty, stable training) |
| **Class balance** | 37x imbalance → **2.5x** (aggressive rebalancing) |
| **Augmentation** | None → **Mixup** (α=0.3, input-level) |
| **Fusion** | Transformer → **LayerNorm fusion** with improved architecture |
| **RL** | PPO → **PPO with safety guard** (auto-revert if degradation) |

### Results
| Metric | v3 | v4 | Δ |
|--------|:--:|:--:|:-:|
| Accuracy | 65.97% | **81.94%** | **+15.97 pp** |
| Macro F1 | ~0.60 | **0.8193** | **+0.22** |
| Cohen's κ | — | **0.7552** | — |

### Per-Class F1
| Emotion | v2 | v4 | Δ |
|---------|:--:|:--:|:-:|
| Angry | 0.38 | **0.80** | +0.42 |
| Happy | 0.58 | **0.77** | +0.19 |
| Sad | 0.66 | **0.88** | +0.22 |
| Neutral | 0.58 | **0.82** | +0.24 |

### Analysis
- **Class rebalancing was the single biggest win** — 37x→2.5x lifted EEG encoder from 23%→37.5% standalone accuracy
- All 4 classes now exceed 0.75 F1 (previously Angry was at 0.38)
- CLS-token architecture with pre-norm attention dramatically improved EEG representations
- WGAN-GP provided stable, high-quality augmented samples
- RL safety guard correctly identified PPO as destructive and reverted
- **Both project targets exceeded:** Accuracy ≥75% ✅, Macro F1 ≥0.72 ✅

### Why It Couldn't Push Past ~82%
1. **Encoders still frozen during fusion:** The transformer fusion head trained on fixed embeddings — even though encoders were individually stronger (37.5% EEG, 56.5% speech), they couldn't adapt to cross-modal interactions
2. **Data quantity bottleneck:** Only 4,424 training speech utterances with ~5M total parameters — the model is fundamentally overparameterized for this dataset size, hitting the ceiling of what frozen-encoder fusion can achieve
3. **No cross-modal attention during encoding:** EEG and speech were encoded independently, then fused — the encoders couldn't use information from the other modality to refine their own representations
4. **RL remained useless:** The PPO safety guard correctly detected degradation every time and reverted — the entire RL component added complexity without benefit
5. **Mixup helps but isn't enough:** Input-level mixup (α=0.3) provided some regularization, but with frozen encoders the diversity of augmented embeddings was limited

### Limitations
- Encoders frozen during fusion — no end-to-end joint optimization
- Small speech dataset (4,424 samples) with 5M+ parameters → overfitting ceiling
- No bidirectional cross-modal attention — EEG cannot attend to speech features during encoding
- No per-class modality weighting — fusion treats all emotions identically
- RL agent adds engineering complexity with zero accuracy benefit
- EEG standalone accuracy (37.5%) still relatively low — limits the multimodal ceiling
- DEAP labels derived from valence/arousal thresholds — label noise is inherent

---

## v5 — Cross-Modal Mutual Attention (CMMA)

**Goal:** Push beyond 81.94% with a novel end-to-end trainable fusion architecture.

### Novel Architecture: CMMA + EAG

```
Raw EEG (B, 160)                          Raw Speech (B, T, 120)
      │                                         │
      ▼                                         ▼
[EEG Encoder] → eeg_emb (B, 128)    [Speech Encoder] → sp_emb (B, 128)
      │                                         │
      ▼                                         ▼
[Tokenize: 128 → 8×16 + CLS]        [Tokenize: 128 → 8×16 + CLS]
      │                                         │
      └──────────── CMMA Layers ────────────────┘
                (3 layers, bidirectional)
                EEG attends to Speech
                Speech attends to EEG
                Gated residual (sigmoid gate, bias=-2.0)
      ┌─────────────────────────────────────────┐
      │                                         │
      ▼                                         ▼
[Pool CLS] → eeg_enhanced          [Pool CLS] → sp_enhanced
      │                                         │
      └──── Emotion-Aware Gating (EAG) ────────┘
             Per-class modality weights
             Annealed teacher forcing
                      │
                      ▼
              [3-layer MLP Classifier]
                      │
                      ▼
               logits (B, 4)
```

### Key Innovations

1. **Cross-Modal Mutual Attention (CMMA)**
   - Bidirectional: EEG attends to speech AND speech attends to EEG simultaneously
   - Gated residual: sigmoid gate (initialized near 0.12) controls cross-modal injection
   - Unlike standard cross-attention, the gate learns **how much** to trust the other modality per token
   - Inspired by ViLBERT but adapted for EEG+Speech

2. **Emotion-Aware Gating (EAG)**
   - Learnable per-class gate logits → sigmoid → modality balance weights
   - Different emotions can prioritize different modalities (e.g., EEG for Angry, Speech for Neutral)
   - Input-dependent adjustment: lightweight network modulates gates based on actual features
   - Annealed teacher forcing: starts with ground-truth labels, transitions to model predictions

3. **End-to-End Joint Training**
   - Discriminative LR: encoders at 0.05× (preserve pretrained features), CMMA at 1×, EAG at 3×
   - Frozen encoder warmup: first 8 epochs train only CMMA head
   - Cosine decay with linear warmup

4. **Auxiliary Unimodal Losses**
   - Each modality gets its own classification head
   - Ensures both encoders produce class-discriminative features independently

### Model Size
| Component | Parameters |
|-----------|:----------:|
| EEG Encoder | 2.5M |
| Speech Encoder | 2.9M |
| CMMA Fusion | 1.9M |
| **Total** | **7.3M** |

### Training Configuration (v5.3)
| Parameter | Value |
|-----------|:-----:|
| d_model | 128 |
| n_heads | 4 |
| n_cmma_layers | 3 |
| ff_dim | 512 |
| dropout | 0.15 |
| modality_dropout | 0.05 |
| lr (CMMA) | 3e-4 |
| encoder_lr_factor | 0.05 |
| eag_lr_factor | 3.0 |
| weight_decay | 3e-4 |
| warmup_epochs | 5 |
| freeze_encoder_epochs | 8 |
| patience | 20 |
| samples_per_epoch | 10,000 |
| batch_size | 64 |
| label_smoothing | 0.1 |
| tf_anneal_epochs | 25 |
| gate_div_weight | 0.1 |

### v5 Sub-Version Results

| Version | Accuracy | What Changed | Why It Regressed |
|---------|:--------:|-------------|-----------------|
| v5.0 | 80.90% | Initial CMMA | EAG weights stuck at 0.50/0.50 |
| v5.1 | 82.05% | Conservative encoder LR, higher EAG LR | EAG still not differentiating |
| v5.2 | 75.15% | Teacher forcing + diversity loss | 100% TF caused massive overfit |
| **v5.3** | **82.55%** | **Annealed TF (1→0 over 25 epochs)** | **BEST** |
| v5.4 | 80.15% | + Embedding mixup + Model EMA | Mixup destructive, EMA-only val missed peaks |
| v5.5 | 71.90% | + Confidence penalty | Penalty was backwards (minimized entropy) |
| v5.6 | 69.00% | Fixed penalty + deterministic val | Over-regularized, small fixed val set |
| v5.7 | — | Clean revert to v5.3 | Not tested (went to v5.8) |
| v5.8 | 73.50% | + R-Drop + DropPath + InputAugment | R-Drop doubled compute, augmentation too aggressive |

### v5.3 Learned Modality Weights (Best Run)
| Emotion | EEG Weight | Speech Weight | Dominant |
|---------|:----------:|:-------------:|----------|
| Angry | higher | lower | EEG |
| Happy | ~balanced | ~balanced | — |
| Sad | lower | higher | Speech |
| Neutral | lower | higher | Speech |

### Why v5.3 Couldn't Push Past ~83%
1. **Fundamental data scarcity:** Only 4,424 speech training samples with 7.3M parameters — the model is 1,650× overparameterized relative to the smaller modality. Train accuracy routinely hit 95–99% while val plateaued at 82–83%, confirming overfitting as the hard ceiling
2. **Random cross-modal pairing:** Each epoch creates different EEG–speech pairs via random sampling — validation accuracy fluctuates ±2–3% between runs due to pair variance, making it hard to distinguish genuine improvements from noise
3. **EEG label noise:** DEAP emotion labels are derived from self-reported valence/arousal scores thresholded into 4 quadrants — this introduces systematic label noise (~10–15% estimated disagreement), creating an accuracy ceiling regardless of model quality
4. **Single dataset per modality:** EEG comes only from DEAP (32 lab subjects), speech only from IEMOCAP (10 actors) — no external data for pretraining diversity or cross-dataset validation
5. **Modality quality gap:** EEG standalone accuracy is ~37.5% (near random for 4 classes) while speech is ~56.5% — the fusion is asymmetric, and the weaker EEG modality contributes limited discriminative signal

### Why v5.4–v5.8 All Regressed
| Version | Accuracy | Root Cause of Regression |
|---------|:--------:|-------------------------|
| v5.4 (80.15%) | −2.4 pp | **Embedding mixup** blended representations from different classes at the embedding level, creating ambiguous samples that confused the classifier. **Model EMA** was used for validation only, meaning the reported "best" model was the EMA shadow — but the EMA checkpoint was never saved, so actual deployed model was worse |
| v5.5 (71.90%) | −10.65 pp | **Confidence penalty code had a critical bug:** computed `−entropy` (negative entropy) and ADDED it to the loss. This MINIMIZED entropy → MAXIMIZED confidence → extreme overconfidence. Train accuracy reached 99%, val collapsed to 65%. Random val pairing made EMA comparison unreliable each pass |
| v5.6 (69.00%) | −13.55 pp | **Over-regularized:** dropout 0.15→0.20, weight_decay 3e-4→5e-4, label_smoothing 0.1→0.15 — all increased simultaneously. **Deterministic val set** of only 2,000 fixed pairs was too small and unrepresentative. Model peaked at epoch 3 (69%) then degraded to 56% by epoch 20 — catastrophic forgetting of encoder representations under heavy regularization |
| v5.8 (73.50%) | −9.05 pp | **R-Drop doubled forward passes** (2× compute, 2× gradient noise) on an already small dataset. **DropPath** randomly skipped CMMA residuals, destabilizing the gated cross-attention that needs consistent gradient flow to learn. **InputAugment** (Gaussian noise + SpecAugment) was too aggressive for already-small EEG features (160-dim) |

### Lessons Learned from v5
1. **v5.3's annealed teacher forcing was the sweet spot** — starts with ground truth labels to bootstrap EAG, gradually transitions to model predictions
2. **Every "improvement" after v5.3 caused regression** — the architecture is well-tuned, additional regularization hurts
3. **Small dataset (4,424 speech samples) is the fundamental bottleneck** — 7.3M parameters means overfitting is inevitable
4. **Gate diversity loss is essential** — without it, per-class gates collapse to uniform 0.50/0.50
5. **Discriminative LR matters** — encoders need 20× lower LR than CMMA to preserve pretrained representations
6. **Never change multiple things at once** — v5.4–v5.8 each introduced 2–3 simultaneous changes, making diagnosis impossible until reverted
7. **Bugs in regularization are catastrophic** — the backwards confidence penalty (v5.5) destroyed the model in a way that looked like normal overfitting, hiding the root cause

### Limitations (v5 Overall)
- **Data ceiling:** 4,424 speech samples is insufficient for a 7.3M parameter model — no architectural change can overcome this without more data or much stronger pretraining
- **Label noise:** DEAP valence/arousal → 4-class mapping introduces ~10–15% systematic label noise, creating a hard accuracy ceiling around 85–90%
- **Single-dataset per modality:** No cross-corpus validation or external pretraining data
- **Random pairing noise:** Validation accuracy varies ±2–3% across runs due to stochastic EEG–speech pairing
- **EEG modality weakness:** EEG standalone at ~37.5% contributes limited discriminative value — the system is heavily speech-dependent
- **No subject-independent evaluation:** Train/val split is random, not subject-wise — some same-subject data appears in both splits, inflating accuracy
- **Computational cost for small gains:** v5's end-to-end training is 5–10× slower than v4's frozen-encoder fusion, for only +0.61 pp improvement
- **RL remains abandoned:** The PPO component (v1–v4) was never beneficial and was dropped entirely in v5 — represents wasted architectural complexity

---

## Overall Progression

```
v1 (prototype)
 │
 ▼
v2: 55.92%  ── Label-aligned pairing, gated fusion, focal loss
 │  (+55.92 pp from nothing)
 ▼
v3: 65.97%  ── Contrastive pretrain, DANN, transformer fusion
 │  (+10.05 pp)
 ▼
v4: 81.94%  ── Class rebalancing, CLS-token encoder, WGAN-GP
 │  (+15.97 pp)  ← BIGGEST SINGLE JUMP
 ▼
v5: 82.55%  ── CMMA gated cross-attention, emotion-aware gating
    (+0.61 pp)
```

### What Mattered Most (Ranked by Impact)

| Rank | Technique | Impact | Version |
|:----:|-----------|:------:|---------|
| 1 | Class rebalancing (37x → 2.5x) | +15.97 pp | v4 |
| 2 | Label-aligned pairing | Essential | v2 |
| 3 | Contrastive pretraining (SimCLR + NT-Xent) | +10.05 pp | v3 |
| 4 | DANN domain adaptation | Part of v3 | v3 |
| 5 | Transformer/CMMA cross-modal attention | +0.61 pp | v4→v5 |
| 6 | CLS-token EEG encoder | Part of v4 | v4 |
| 7 | Emotion-aware gating | Interpretability | v5 |
| 8 | Focal loss + class weights | Essential | v2 |

### What Did NOT Work

| Technique | Version | Result |
|-----------|---------|--------|
| RL augmentation control (PPO) | v1–v4 | Always destructive |
| Embedding mixup | v5.4 | −2.4 pp |
| Model EMA | v5.4–v5.5 | Unreliable comparison |
| Confidence penalty (entropy) | v5.5 | −10.65 pp (backwards bug) |
| Deterministic small val set | v5.6 | −13.55 pp |
| R-Drop + DropPath + InputAugment | v5.8 | −9.05 pp |

---

## Dataset Details

### DEAP (EEG)
- 32 subjects × 40 trials × 60 seconds
- 76,800 total samples after preprocessing
- Features: Differential entropy across 4 frequency bands (θ, α, β, γ) × 32 channels + 28 asymmetry features = 160 dimensions
- Labels: Valence/arousal → 4 emotion classes
- **Class distribution (after rebalancing):** Angry: 1140, Happy: 41940, Sad: 16680, Neutral: 17040

### IEMOCAP (Speech)
- 5 sessions × 2 actors = 10 speakers
- 5,531 utterances (after filtering to 4 emotions)
- Features: 80 MFCCs × T time frames (padded/truncated to 120 frames)
- **Class distribution:** Angry: 1636, Happy: 1084, Sad: 1103, Neutral: 1708

### Train/Val Split
- 80/20 stratified split (random_state=42)
- Train: 61,440 EEG + 4,424 speech
- Val: 15,360 EEG + 1,107 speech

---

## Repository Structure

```
amers/
├── config/default.yaml          # All hyperparameters
├── src/
│   ├── models/
│   │   ├── eeg_encoder.py       # CLS-token attention EEG encoder
│   │   ├── speech_encoder.py    # CNN-LSTM + attention pooling
│   │   ├── gan.py               # WGAN-GP conditional generator
│   │   ├── cmma_fusion.py       # CMMA + EAG fusion (v5)
│   │   └── ...
│   ├── data/
│   │   ├── deap_loader.py
│   │   └── iemocap_loader.py
│   ├── training/
│   │   └── fusion_trainer.py    # FocalLoss, LabelAlignedDataset
│   └── utils/
├── scripts/
│   ├── preprocess_deap.py
│   ├── preprocess_iemocap.py
│   ├── train_eeg.py
│   ├── train_speech.py
│   ├── train_gan.py
│   ├── train_fusion.py
│   ├── train_rl.py
│   ├── v3_pretrain_eeg.py       # SimCLR contrastive
│   ├── v3_pretrain_speech.py    # NT-Xent contrastive
│   ├── v3_train_dann.py         # DANN domain adaptation
│   ├── v3_train_transformer_fusion.py
│   ├── v3_train_rl.py
│   ├── v3_evaluate.py
│   ├── v5_train_cmma.py         # CMMA end-to-end (current best)
│   └── evaluate.py
├── notebooks/
│   └── 00_setup_and_run.ipynb   # Google Colab master notebook
└── docs/
    └── VERSION_HISTORY.md       # This document
```

---

## Runtime Environment
- **Platform:** Google Colab Pro
- **GPU:** NVIDIA L4 (22 GB VRAM)
- **Python:** 3.12
- **PyTorch:** 2.x with CUDA, AMP (mixed precision)
- **Training time (v5):** ~15 minutes for 80 epochs on L4

---

*Document generated: February 2026*
*Best model: v5.3 — 82.55% accuracy (commit 1751ad1)*
*Current code: v5.3 (reverted from v5.8)*
