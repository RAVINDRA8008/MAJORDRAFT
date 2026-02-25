# AMERS Paper — Figure Generation Prompts

> **Use these prompts in Google Gemini / ImageFX / any AI image generator.**
> The enhanced `paper.tex` already includes **TikZ/pgfplots versions** of all 5 figures that compile directly. These prompts are for generating **higher-quality raster alternatives** if you want to replace the TikZ versions with `\includegraphics`.

---

## How to Use

1. Copy the prompt for the figure you want
2. Paste it into your AI image generation tool (Google Gemini, ImageFX, Midjourney, DALL-E, etc.)
3. Save the generated image as `figures/fig1.png` (or `.pdf`)
4. In `paper.tex`, replace the TikZ block with:
   ```latex
   \begin{figure}[t]
       \centering
       \includegraphics[width=\columnwidth]{figures/fig1.png}
       \caption{...keep existing caption...}
       \label{fig:architecture}
   \end{figure}
   ```

---

## Figure 1: System Architecture Diagram

### Prompt (Technical Diagram Style)

```
Create a clean, professional technical architecture diagram for an academic research paper on multimodal emotion recognition. White background with a top-to-bottom flowchart layout.

At the TOP: Two parallel input boxes:
- LEFT: "EEG Input (160-dim)" in a light blue rounded rectangle
- RIGHT: "Speech Input (80 MFCCs × 120)" in a light coral/red rounded rectangle

SECOND ROW: Two encoder boxes connected by downward arrows:
- LEFT: "EEG Encoder (3-Layer Attention + CLS)" in blue
- RIGHT: "Speech Encoder (CNN-BiLSTM-Attn)" in coral/red

Between the encoders, show a small label: "Contrastive + DANN Pretraining"

THIRD ROW: "Tokenizer" boxes for each stream, converting to "8×16 + CLS tokens"

MIDDLE (main focus area): A large stack of 3 identical blocks labeled "CMMA Layer 1", "CMMA Layer 2", "CMMA Layer 3" in a soft steel-blue color. Each block has a label inside: "Self-Attn → Cross-Attn → Gated Residual → FFN". Show bidirectional arrows between the two streams inside each layer (EEG ↔ Speech).

A bracket on the left side labels the stack as "N=3 Bidirectional CMMA Layers"

BELOW THE STACK: "CLS Token Pooling" box

NEXT: "Emotion-Aware Gating (EAG)" box in warm orange, with a small annotation "Per-class α, Annealed TF"

Small side boxes branching from the pooling layer: "Aux EEG Head" and "Aux Speech Head" (shown with dashed lines)

BOTTOM: "MLP Classifier (128→256→128→4)" in green, leading to final output "Emotion: Angry | Happy | Sad | Neutral" in a golden box.

Style: Clean vector-like diagram, no shadows, minimal gradients. Use consistent rounded rectangles. Color scheme: blues for EEG path, coral/reds for speech path, steel-blue for CMMA, warm orange for EAG, green for classifier. Arrows should be clean with arrowheads. Font should be sans-serif, legible at print size. Suitable for an IEEE conference paper.
```

### Alternative Simpler Prompt
```
Professional technical flowchart diagram for a research paper. White background. Top-to-bottom flow showing: EEG Input and Speech Input at top → separate Encoders → Tokenizers → 3 stacked "CMMA Layers" (cross-modal mutual attention, shown as a blue block with bidirectional arrows) → CLS Pooling → "Emotion-Aware Gating" orange block → MLP Classifier → 4-class emotion output. Clean, minimal, vector-style. IEEE conference paper quality. Blue tones for EEG, red tones for speech, neutral for shared components.
```

---

## Figure 2: Learned Modality Weights (Per-Class Bar Chart)

### Prompt
```
Create a clean grouped bar chart for an academic paper. White background.

X-axis has 4 emotion categories: Angry, Happy, Sad, Neutral
Y-axis is "Modality Weight (α)" from 0.0 to 0.7

For each emotion, show two bars side by side:
- Light steel blue bar = "EEG Weight"
- Soft coral/salmon bar = "Speech Weight"

Values:
- Angry: EEG=0.57, Speech=0.43
- Happy: EEG=0.50, Speech=0.50
- Sad: EEG=0.43, Speech=0.57
- Neutral: EEG=0.44, Speech=0.56

Each bar should have its value printed above it. Include a clean legend at the top: "EEG Weight" (blue) and "Speech Weight" (coral). Light gray horizontal grid lines. No background color. Clean sans-serif font. Suitable for an IEEE conference paper single-column figure.
```

---

## Figure 3: Training Curves with TF Ratio

### Prompt
```
Create a dual-axis line chart for a research paper on deep learning training dynamics. White background.

LEFT Y-AXIS (dark blue): "Accuracy (%)" from 20% to 100%
RIGHT Y-AXIS (orange): "Teacher Forcing Ratio (ρ)" from 0.0 to 1.0
X-AXIS: "Epoch" from 0 to 80

Three lines:
1. TRAINING ACCURACY (solid dark blue line with small circle markers): Starts at 28%, rises steeply to 68% by epoch 10, continues rising to 86% by epoch 25, then gradually reaches 95% by epoch 60, plateaus at 96%.

2. VALIDATION ACCURACY (solid dark green line with triangle markers): Starts at 26%, rises to 65% by epoch 10, reaches 80.5% by epoch 25, peaks at 82.55% at epoch 43, then slowly declines to 80.5% by epoch 80.

3. TF RATIO (dashed orange line, no markers): Starts at 1.0, linearly decreases to 0.0 at epoch 25, stays at 0.0 for remaining epochs.

Add a vertical dashed golden line at epoch 43 with annotation "Best: 82.55% (epoch 43)".

Include a legend box with all three lines labeled. Light gray grid. Clean, minimal style suitable for an IEEE paper. The gap between training and validation curves (overfitting gap ~12-13 pp) should be clearly visible.
```

---

## Figure 4: Version Progression Bar Chart

### Prompt
```
Create a single-series bar chart showing performance progression for a research paper. White background.

X-axis labels (bold): v1, v2, v3, v4, v5.3
Y-axis: "Validation Accuracy (%)" from 0 to 100

Bar values:
- v1: 25% (light gray, since it's an estimate / no formal evaluation)
- v2: 55.92% (medium blue)
- v3: 65.97% (medium blue)
- v4: 81.94% (slightly brighter blue)
- v5.3: 82.55% (highlighted with a green border or different shade to mark it as best)

Each bar has its value printed above it in bold.

Add an upward arrow annotation between v3 and v4 bars with the text "+15.97 pp (Class rebalancing)" in green, showing this was the biggest jump.

Light horizontal grid lines. Clean minimal style. Sans-serif font. Suitable for IEEE conference paper single-column width.
```

---

## Figure 5: Confusion Matrix Heatmap

### Prompt
```
Create a 4×4 confusion matrix heatmap for a research paper on emotion classification. White background.

Axis labels (both axes): Angry, Happy, Sad, Neutral
Top label: "Predicted"
Left label: "True"

Matrix values (row = true class, column = predicted class):
Row 1 (Angry):   80%, 8%, 5%, 7%
Row 2 (Happy):   5%, 77%, 4%, 14%
Row 3 (Sad):     3%, 5%, 88%, 4%
Row 4 (Neutral): 4%, 12%, 3%, 81%

Color coding:
- DIAGONAL cells (correct predictions): Green shading, darker green for higher values. 88% should be darkest green, 77% lightest green.
- OFF-DIAGONAL cells (errors): Red/coral shading, darker for higher error values. 14% and 12% (happy↔neutral confusions) should be the darkest off-diagonal cells. Low values (3%, 4%, 5%) should be very light pink/almost white.

Each cell displays its percentage value. Diagonal values should be in bold font.

Clean grid lines separating cells. No color bar needed (values are shown directly). Sans-serif font. Professional academic style suitable for an IEEE conference paper.
```

---

## General Style Tips for All Figures

When generating images for academic papers:

1. **Resolution**: Generate at least 300 DPI. If the tool allows, request 2000×1500 pixels or larger.
2. **Background**: Always white/transparent. Never use dark backgrounds.
3. **Font**: Sans-serif (Arial, Helvetica). Keep text large enough to be readable when the figure is scaled to column width (~3.5 inches / 8.9 cm).
4. **Colors**: Use a consistent, muted palette. Avoid neon/bright colors. Good choices:
   - Blues: `#295A82`, `#4682B4`, `#6699CC`
   - Greens: `#277749`, `#2E8B57`
   - Reds/Corals: `#CC6666`, `#B22222`
   - Gold: `#CDA51D`
   - Grays: `#F5F5F5` (background), `#B4B4B4` (borders)
5. **No watermarks**: If the tool adds watermarks, crop them out.
6. **Vector preference**: If the tool supports SVG/PDF output, prefer that over PNG for sharper printing.

---

## Quick Reference: Which Figures Are Already in the LaTeX

The enhanced `paper.tex` includes **compilable TikZ/pgfplots code** for all 5 figures:

| Figure | Type | In LaTeX? | External Generation Recommended? |
|--------|------|-----------|--------------------------------|
| Fig 1: Architecture | TikZ flowchart | ✅ Yes | Optional (TikZ version is clean) |
| Fig 2: Modality Weights | pgfplots bar chart | ✅ Yes | Optional (pgfplots is publishable) |
| Fig 3: Training Curves | pgfplots dual-axis | ✅ Yes | Optional |
| Fig 4: Version Progression | pgfplots bar chart | ✅ Yes | Optional |
| Fig 5: Confusion Matrix | TikZ heatmap | ✅ Yes | Optional |

**Recommendation**: The TikZ/pgfplots figures compile to **vector graphics** which look crisp at any zoom level, and are self-contained (no external files needed). Only generate external images if you specifically want a different visual style.

---

## Replacing TikZ Figures with External Images

If you generate external images and want to swap them in:

1. Create a `figures/` directory in your LaTeX project
2. Save images as `fig1.png` through `fig5.png`
3. For each figure, replace the `\begin{tikzpicture}...\end{tikzpicture}` block with:

```latex
\includegraphics[width=\columnwidth]{figures/fig1.png}
```

Keep the `\begin{figure}...\end{figure}` wrapper, `\caption{}`, and `\label{}` intact.
