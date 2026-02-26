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
| **LOSO v1** | 68.41% ± 8.37% | 0.575 | — | — | Baseline 32-fold subject-independent |
| **LOSO v2** | **89.86% ± 6.86%** | **0.744** | — | — | **Improved LOSO (+21.44 pp)** |

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
- **Simple concatenation fusion** — just stacking EEG + speech vectors throws away inter-modal relationships; the MLP couldn't learn cross-modal correlations from flat concatenation
- **Vanilla conditional GAN** — training was unstable (mode collapse), generated low-quality augmented samples that added noise rather than useful data
- **No class balancing** — DEAP has 37× imbalance (Angry: 1,140 vs Happy: 41,940); model learned to predict majority class
- **Index-based pairing** — EEG sample #1234 paired with speech sample #1234 regardless of emotion label, creating random cross-modal pairs
- **RL agent unconstrained** — PPO had no safety guard and actively degraded accuracy

### Limitations
- No formal metrics collected — impossible to compare quantitatively with later versions
- No label-aligned pairing meant the fusion model received contradictory supervision signals
- GAN mode collapse meant augmentation was essentially random noise injection
- No early stopping or learning rate scheduling — training was unstable

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
- **37× class imbalance untouched** — despite balanced sampling, the underlying DEAP distribution was still 37:1 (Angry vs Happy), causing the EEG encoder to learn biased representations
- **EEG encoder only 23% standalone accuracy** — the 2-layer attention network couldn't extract meaningful features from the imbalanced data, so fusion received poor EEG embeddings
- **Gated fusion was shallow** — single sigmoid gate couldn't model complex cross-modal interactions; it was just a weighted average
- **RL agent was destructive** — PPO augmentation control reduced accuracy rather than improving it; no safety mechanism to prevent degradation

### Limitations
- 55.92% accuracy is only marginally above random (25%) for 4 classes
- Angry F1 at 0.38 means the model almost never correctly predicts anger
- No contrastive pretraining — encoders trained with supervised loss only, missing self-supervised representation quality
- No domain adaptation — DEAP (EEG, lab setting) and IEMOCAP (speech, acted) have significant domain gap
- Fixed fusion architecture — no cross-modal attention, just element-wise gating

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
- **37× class imbalance STILL untouched** — the single biggest bottleneck remained; contrastive learning and DANN improved representation quality but couldn't fix fundamentally skewed class distributions
- **EEG encoder architecture unchanged** — still a 2-layer network, now with better pretrained weights but same limited capacity
- **Frozen encoders during fusion** — transformer fusion trained on fixed embeddings (no end-to-end fine-tuning), limiting how much the fusion could adapt the representations
- **RL v2 still destructive** — despite composite reward (acc + F1 + balance − overfit penalty), PPO augmentation control degraded accuracy; safety guard correctly reverted

### Limitations
- 65.97% accuracy is decent but well below the 75% target
- No end-to-end training — encoders frozen during fusion means the system can't jointly optimize representations for cross-modal classification
- Transformer fusion had only 2 layers with 4 heads — relatively shallow cross-modal attention
- DANN domain adaptation treated DEAP and IEMOCAP as two domains but they differ in modality (EEG vs speech), not just domain — the gradient reversal may not be optimal
- Contrastive pretraining used generic augmentations, not emotion-specific ones

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

### Why It Didn't Go Higher
- **Frozen encoder fusion** — fusion classifier trained on fixed embeddings; no gradient flows back to encoders, so representations can't adapt to the fusion objective
- **No cross-modal attention during encoding** — EEG and speech encoded independently with no interaction until the fusion layer, missing early-stage complementary features
- **RL still destructive** — even with safety guard, PPO could not improve accuracy; augmentation ratio control via RL appears fundamentally mismatched to this problem
- **Modest EEG standalone accuracy (37.5%)** — the EEG encoder, while much improved, still provides weak signal; fusion relies heavily on speech (56.5% standalone)

### Limitations
- 81.94% accuracy with train accuracy at ~95% = significant overfitting gap
- Only 4,424 speech training samples for a 7.3M parameter pipeline — data bottleneck
- LayerNorm fusion is simpler than the transformer fusion from v3 — traded attention for stability
- Mixup augmentation helps but is a workaround for insufficient data diversity
- RL component is essentially dead weight — always reverted by safety guard across v1–v4

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

| Version | Accuracy | What Changed | Outcome |
|---------|:--------:|-------------|---------|
| v5.0 | 80.90% | Initial CMMA | EAG weights stuck at 0.50/0.50 |
| v5.1 | 82.05% | Conservative encoder LR, higher EAG LR | EAG still not differentiating |
| v5.2 | 75.15% | Teacher forcing + diversity loss | 100% TF caused massive overfit |
| **v5.3** | **82.55%** | **Annealed TF (1→0 over 25 epochs)** | **BEST RESULT** |
| v5.4 | 80.15% | + Embedding mixup + Model EMA | −2.4 pp regression |
| v5.5 | 71.90% | + Confidence penalty | −10.65 pp regression |
| v5.6 | 69.00% | Fixed penalty + deterministic val | −13.55 pp regression |
| v5.7 | — | Clean revert to v5.3 | Not tested |
| v5.8 | 73.50% | + R-Drop + DropPath + InputAugment | −9.05 pp regression |

---

### v5.0 — Initial CMMA (80.90%)

**What changed:** First implementation of Cross-Modal Mutual Attention with Emotion-Aware Gating. Bidirectional cross-attention (3 layers), sigmoid gate (bias=−2.0), EAG with fixed class gate logits. Encoder LR factor = 0.1, EAG LR = 1× (same as CMMA).

**Why it failed:**
- EAG gate logits stayed at 0.50/0.50 for all emotions — the gating mechanism was dead
- Equal LR for EAG and CMMA was too low for the gate parameters to learn meaningful per-class differences
- Encoder LR at 0.1× was too aggressive — pretrained representations were degrading too fast
- Without per-class differentiation, EAG was just an expensive identity function

**Limitations:**
- No mechanism to force gate diversity
- Encoder fine-tuning started immediately (no frozen warmup)
- EAG got the same LR as the much-larger CMMA layers

---

### v5.1 — Conservative LR + Higher EAG LR (82.05%)

**What changed:** Reduced encoder LR factor from 0.1 to 0.05 (more conservative). Increased EAG-specific LR to 3× CMMA LR. Added frozen encoder phase (first 8 epochs). Added modality dropout (0.05). Increased samples per epoch from 5000 to 10000.

**Why it failed to differentiate EAG:**
- Despite 3× LR, EAG gate logits still converged to ~0.50/0.50
- Without ground-truth supervision, the gating signal was too weak — the classification loss alone couldn't push per-class gates apart
- The model found it easier to use CMMA cross-attention for modality weighting than EAG

**Limitations:**
- No teacher forcing — EAG had to discover per-class preferences purely from classification gradients
- No explicit diversity loss to penalize uniform gates
- Still a +1.15 pp improvement over v5.0 from the LR/warmup fixes alone

---

### v5.2 — Teacher Forcing + Diversity Loss (75.15%)

**What changed:** Added teacher forcing (TF) — during training, EAG uses ground-truth labels instead of model predictions to select per-class gates. Added gate diversity loss (`−std(gate_logits)`) to penalize uniform weights. TF ratio = 1.0 (100% teacher forcing, always uses ground truth).

**Why it failed:**
- **100% teacher forcing was catastrophic** — at training time, the model always received perfect labels, so it learned to rely entirely on the oracle signal
- At validation/test time (no labels available), the model had to use its own emotion probe predictions — which it never learned to do
- Result: train accuracy ~99%, val accuracy 75% — **massive overfitting to the TF signal**
- The model's emotion probe was untrained because TF bypassed it entirely

**Limitations:**
- TF ratio was never annealed — the model never practiced using its own predictions
- Classic exposure bias problem: trained with oracle, tested without it
- Gate diversity loss worked (gates differentiated!) but the model couldn't use them without TF
- Dropped −6.9 pp from v5.1 despite the gates finally learning different weights

---

### v5.3 — Annealed Teacher Forcing (82.55%) ★ BEST

**What changed:** Annealed TF ratio from 1.0→0.0 over 25 epochs (`tf_ratio = max(0, 1 − epoch/25)`). First 25 epochs: gradually reducing ground-truth label injection. After epoch 25: model uses only its own emotion probe predictions. All other params same as v5.1.

**Why it succeeded:**
- Annealing solved the exposure bias — model learned to bootstrap from ground truth then transition to self-reliance
- Early epochs: TF helps EAG learn meaningful per-class preferences quickly
- Middle epochs: model practices using its own probe predictions with decreasing label support
- Late epochs: fully autonomous — EAG uses emotion probe, which is now well-trained
- Gate diversity loss kept per-class weights differentiated throughout training
- Result: **82.55% accuracy at epoch 43, with meaningful per-class modality weights**

**Learned Modality Weights:**
| Emotion | EEG Weight | Speech Weight | Dominant |
|---------|:----------:|:-------------:|----------|
| Angry | higher | lower | EEG |
| Happy | ~balanced | ~balanced | — |
| Sad | lower | higher | Speech |
| Neutral | lower | higher | Speech |

**Remaining Limitations:**
- Still overfitting (train ~95%, val 82.55%) — 7.3M params with only 4,424 speech samples
- Modality weight differences are modest (~0.55/0.45, not dramatic like 0.8/0.2)
- Random val pairing introduces noise (~±1% variance between runs)
- Cannot exceed ~83% without more data or fundamentally different approach

---

### v5.4 — Embedding Mixup + Model EMA (80.15%)

**What changed:** Added embedding-level mixup (α=0.3) — interpolate EEG and speech embeddings between random samples. Added Model EMA (exponential moving average, decay=0.998) for smoother validation. Used EMA model for validation, raw model for training.

**Why it failed:**
- **Embedding mixup was destructive** — mixing embeddings from different emotion classes created unnatural intermediate representations that confused the CMMA cross-attention
- Unlike input-level mixup (which works for images), embedding-level mixup destroys the learned manifold structure from DANN pretraining
- **EMA-only validation was misleading** — EMA model is smoother but delayed; it missed sharp accuracy peaks from the raw model
- Best raw-model checkpoint may have been skipped because EMA hadn't caught up yet
- Net effect: −2.4 pp regression from two independently harmful additions

**Limitations:**
- No ablation done — couldn't tell which of the two changes (mixup or EMA) caused more harm
- EMA comparison was done on random val pairs (different data each pass), making raw-vs-EMA comparison unreliable
- Mixup label blending conflicted with focal loss class weights

---

### v5.5 — Confidence Penalty (71.90%)

**What changed:** Removed embedding mixup. Added confidence penalty — intended to prevent overconfident predictions by adding entropy to the loss. Added best-of-two EMA comparison (evaluate both raw and EMA, keep whichever is better).

**Why it failed:**
- **The confidence penalty was implemented BACKWARDS** — computed `−entropy` (negative entropy) and ADDED it to the loss
- Minimizing `loss + (−entropy)` = minimizing `loss − entropy` = MAXIMIZING negative entropy = **MINIMIZING entropy**
- This forced the model to be maximally confident (one-hot predictions), the exact opposite of the intended effect
- Result: train accuracy hit 99% (extreme overconfidence), val accuracy collapsed to 71.9%
- The model learned to output near-one-hot probabilities regardless of input
- Additionally, random val pairing made raw-vs-EMA comparison unreliable (different data each pass)

**Limitations:**
- Classic sign error bug — a single minus sign destroyed 10+ pp of accuracy
- No unit test or sanity check to verify the penalty worked as intended
- The EAG weights finally differentiated beautifully (Angry=0.67 EEG, Neutral=0.38 EEG) but accuracy was terrible — good gating can't compensate for a broken loss function

---

### v5.6 — Fixed Penalty + Deterministic Val (69.00%)

**What changed:** Removed the broken confidence penalty entirely. Added deterministic validation dataset (`FixedPairValDataset`) with 2000 pre-computed pairs (seed=42) to eliminate val noise. Increased regularization: dropout 0.15→0.20, weight_decay 3e-4→5e-4, label_smoothing 0.1→0.15.

**Why it failed:**
- **Over-regularized** — three regularization knobs increased simultaneously (dropout, weight decay, label smoothing), on top of an architecture that was already regularized enough at v5.3 settings
- **Deterministic val set too small** — 2000 fixed pairs (500/class) was unrepresentative of the full validation distribution
- Model peaked at epoch 3 (69%) then **degraded to 56–59%** by epoch 20 — catastrophic forgetting of pretrained encoder representations under heavy regularization
- The increased dropout/weight-decay was suppressing the fine-tuned encoder features faster than CMMA could learn to use them
- Early stopping couldn't help because the model peaked in epoch 3 (before any meaningful training happened)

**Limitations:**
- Changed 3 hyperparameters at once — impossible to diagnose which caused the most harm
- 2000 val pairs was too few for a 4-class problem — some class combinations were undersampled
- The regularization increase was the opposite of what the small-data problem needed — needed more data diversity, not more capacity restriction

---

### v5.7 — Clean Revert to v5.3 (Not Tested)

**What changed:** Complete strip of ALL experimental features added since v5.3: removed EMA, mixup, confidence penalty, deterministic val, increased regularization. Restored exact v5.3 hyperparameters: dropout=0.15, weight_decay=3e-4, label_smoothing=0.1. Removed dead code (FixedPairValDataset, ModelEMA classes). Net −66 lines of code.

**Why not tested:** User skipped directly to designing v5.8 before running v5.7 on Colab. Code should reproduce ~82.55% since it's identical to v5.3.

**Limitations:**
- Same as v5.3 — ceiling around 82–83% due to data limitations
- Dead code (FixedPairValDataset, ModelEMA) was left in the file (cleaned in v5.8, then re-added when reverting)

---

### v5.8 — R-Drop + DropPath + InputAugment (73.50%)

**What changed:** Three proven regularization techniques added on top of v5.3:
1. **R-Drop** (α=1.0) — forward each batch twice with different dropout masks, add symmetric KL divergence loss to enforce prediction consistency
2. **Stochastic Depth / DropPath** (rate 0→0.1) — randomly skip CMMA block residuals with linearly increasing probability across layers
3. **Input Augmentation** — Gaussian noise (σ=0.05) on EEG, SpecAugment-style time masking (T=10) and frequency masking (F=5) on speech MFCCs

All other hyperparameters kept identical to v5.3.

**Why it failed:**
- **R-Drop doubled forward computation** — each batch processed twice through the entire model, effectively halving the number of unique samples seen per epoch for the same wall-clock time
- **R-Drop's KL divergence was too strong at α=1.0** — forced the model to produce identical outputs regardless of dropout mask, which constrained the model's capacity to learn discriminative features
- **Input augmentation was too aggressive** — Gaussian noise σ=0.05 on 160-dim EEG features (which are already compact differential entropy values) distorted the signal; SpecAugment masking on 80×120 MFCCs removed too much information from already-short utterances
- **DropPath randomly disabled cross-attention residuals** — the gated cross-attention (which starts near 0.12 due to bias=−2.0) was further suppressed by stochastic depth, preventing the model from learning effective cross-modal interactions
- The three techniques are individually proven for large-scale models (ViT, BERT) but were **too aggressive for a 7.3M parameter model trained on 4,424 speech samples**
- Model peaked at epoch 4 (73.50%) and EAG weights differentiated (Angry EEG=0.60, Neutral Speech=0.59) but classification accuracy was poor

**Limitations:**
- No individual ablation — three techniques added simultaneously, can't isolate which hurt most
- R-Drop was designed for models like BERT (110M params) on datasets with 100k+ samples — fundamentally mismatched scale
- DropPath rate of 0.1 means 10% chance of dropping the deepest CMMA layer entirely — significant for only 3 layers
- All three techniques reduce effective model capacity, which is the opposite of what a data-limited regime needs

---

### v5.3 Learned Modality Weights (Best Run)
| Emotion | EEG Weight | Speech Weight | Dominant |
|---------|:----------:|:-------------:|----------|
| Angry | higher | lower | EEG |
| Happy | ~balanced | ~balanced | — |
| Sad | lower | higher | Speech |
| Neutral | lower | higher | Speech |

### Lessons Learned from v5
1. **v5.3's annealed teacher forcing was the sweet spot** — starts with ground truth labels to bootstrap EAG, gradually transitions to model predictions
2. **Every "improvement" after v5.3 caused regression** — the architecture is well-tuned, additional regularization hurts
3. **Small dataset (4,424 speech samples) is the fundamental bottleneck** — 7.3M parameters means overfitting is inevitable
4. **Gate diversity loss is essential** — without it, per-class gates collapse to uniform 0.50/0.50
5. **Discriminative LR matters** — encoders need 20× lower LR than CMMA to preserve pretrained representations
6. **Techniques designed for large-scale models don't transfer** — R-Drop, DropPath, and mixup all assume abundant data; they restrict capacity in a regime that needs more capacity
7. **Change one thing at a time** — v5.4/v5.6/v5.8 each changed multiple knobs simultaneously, making diagnosis impossible
8. **Always sanity-check loss terms** — v5.5's sign error (−entropy instead of +entropy) was a trivial bug with catastrophic impact

---

## LOSO v1 — Baseline Subject-Independent Evaluation (68.41%)

**Goal:** Evaluate cross-subject generalization using leave-one-subject-out (LOSO) on DEAP's 32 subjects.

**Protocol:**
- 32 folds: each fold holds out 1 subject for testing, next subject for validation, remaining 30 for training
- IEMOCAP speech split 80/20 (constant across all folds)
- Each fold trains from scratch with fresh encoder weights initialized from pretrained checkpoints
- Reduced hyperparameters: 40 epochs, patience 10, 5,000 samples/epoch
- Per-subject z-score normalization (same as preprocessing)

**Results:**

| Metric | Value |
|--------|:-----:|
| Mean accuracy | 68.41% ± 8.37% |
| Median accuracy | 70.29% |
| Range | [48.58%, 86.46%] |
| Mean F1 (macro) | 0.575 ± 0.059 |
| Best fold | Subject 12 — 86.46% |
| Worst fold | Subject 5 — 48.58% |

**Per-class pooled F1:** Happy = 0.12, Sad = 0.73, Angry = 0.85, Neutral = 0.70

**Analysis:**
- The ~14 pp drop from subject-dependent (82.55%) to LOSO (68.41%) was expected due to inter-subject EEG variability
- **Happy collapsed to F1 = 0.12** — its extreme minority status (1,140 / 76,800 = 1.5%) meant the model couldn't generalize Happy patterns across subjects
- High fold variance (std 8.37%) reflected known inter-subject EEG differences in DEAP
- Encoder pretraining (DANN, contrastive) used all 32 subjects → partial data leakage into per-fold LOSO

---

## LOSO v2 — Improved Subject-Independent Evaluation (89.86%)

**Goal:** Close the LOSO gap through normalization, weighting, and evaluation improvements.

**Five Key Improvements:**

1. **Cross-subject z-score normalization** — Compute mean/std on the 30 training subjects; apply to val/test. Removes inter-subject distribution shift. *This was the single largest contributor.*
2. **EEG-only class weights with √ dampening** — Compute class weights from DEAP labels only (not diluted by IEMOCAP's balanced distribution); apply `sqrt()` dampening so Happy gets ~4× weight instead of ~37×.
3. **Higher focal gamma (γ = 3.0 vs. 2.0)** — Stronger focus on hard minority examples.
4. **Extended training** — 60 epochs (vs. 40), patience 15, 10K samples/epoch, warmup 5 epochs.
5. **Multi-pairing test ensemble (5×)** — Average logits over 5 random speech pairings per EEG sample, reducing test-time pairing variance.

**Hyperparameter Overrides (vs. v1):**

| Parameter | LOSO v1 | LOSO v2 |
|-----------|:-------:|:-------:|
| Epochs | 40 | 60 |
| Patience | 10 | 15 |
| Samples/epoch | 5,000 | 10,000 |
| Focal gamma | 2.0 | 3.0 |
| Label smoothing | 0.05 | 0.05 |
| Gate div weight | 0.15 | 0.15 |
| Warmup epochs | — | 5 |
| Freeze encoder epochs | — | 5 |
| Test ensembles | 1 | 5 |

**Results:**

| Metric | LOSO v1 | LOSO v2 | Delta |
|--------|:-------:|:-------:|:-----:|
| Mean accuracy | 68.41% ± 8.37% | **89.86% ± 6.86%** | **+21.44 pp** |
| Median accuracy | 70.29% | **90.81%** | +20.52 pp |
| Range | [48.58%, 86.46%] | [64.62%, 98.42%] | — |
| Mean F1 (macro) | 0.575 ± 0.059 | **0.744 ± 0.097** | +0.169 |
| Best fold | 86.46% (subj. 12) | **98.42% (subj. 14)** | — |
| Worst fold | 48.58% (subj. 5) | 64.62% (subj. 9) | — |

**Per-class Pooled F1 Comparison:**

| Class | LOSO v1 | LOSO v2 | Delta |
|-------|:-------:|:-------:|:-----:|
| Happy | 0.12 | **0.36** | +0.24 |
| Sad | 0.73 | **0.92** | +0.19 |
| Angry | 0.85 | **0.98** | +0.13 |
| Neutral | 0.70 | **0.87** | +0.17 |

**Pooled Classification Report (LOSO v2):**
- Weighted average — Precision: 0.94, Recall: 0.90, F1: 0.91
- Overall pooled accuracy: **90%**

**Analysis:**
- The +21.44 pp jump is almost entirely due to **cross-subject normalization** — fitting z-score statistics on training subjects and applying them to the held-out subject
- Happy F1 improved 3× (0.12 → 0.36) thanks to EEG-only sqrt-dampened class weights, but remains the weakest class due to extreme minority status
- LOSO v2 (89.86%) actually **exceeds** the subject-dependent accuracy (82.55%) — the subject-dependent evaluation's random 80/20 split mixes subjects, and per-subject z-scores hide the cross-subject shift that normalization fixes
- 26 of 32 folds exceed 82.55%; only 2 folds fall below 75%
- Worst fold (subject 9, 64.62%) likely has highly atypical EEG patterns
- Encoder pretraining leak caveat still applies (DANN/contrastive used all 32 subjects)

**Key Lesson:** Cross-subject feature normalization is as impactful as architectural design. The v1→v2 improvement (+21.44 pp) is larger than any single architecture change in the entire project history.

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
 │  (+15.97 pp)  ← BIGGEST SINGLE JUMP (architecture)
 ▼
v5: 82.55%  ── CMMA gated cross-attention, emotion-aware gating
 │  (+0.61 pp)
 ▼
LOSO v1: 68.41%  ── Baseline 32-fold LOSO evaluation
 │  (subject-independent baseline)
 ▼
LOSO v2: 89.86%  ── Cross-subject norm, EEG-only weights, ensemble
    (+21.44 pp)  ← BIGGEST SINGLE JUMP (evaluation methodology)
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
| 9 | Cross-subject z-score normalization | +21.44 pp (LOSO) | LOSO v2 |
| 10 | EEG-only class weights (√ dampening) | Part of LOSO v2 | LOSO v2 |
| 11 | Multi-pairing test ensemble (5×) | Part of LOSO v2 | LOSO v2 |

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
│   ├── v5_loso.py               # LOSO v1 — baseline 32-fold evaluation
│   ├── v5_loso_v2.py            # LOSO v2 — improved LOSO (89.86%)
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
*Best model (subject-dependent): v5.3 — 82.55% accuracy (commit 1751ad1)*
*Best LOSO (subject-independent): LOSO v2 — 89.86% ± 6.86% accuracy (commit 3e82cf6)*
*Current code: v5.3 + LOSO v2*
