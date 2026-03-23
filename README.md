# ExtractArt — CRNN Art Classifier
### HumanAI @ Google Summer of Code 2026

Classifies WikiArt paintings by **Style**, **Genre**, and **Artist** and detects outlier paintings that don't fit their assigned class.

---

## Requirements
- Google Account + Google Drive (~5 GB free space)
- Google Colab with T4 GPU
- WikiArt CSV files (9 files from ArtGAN repository)

---

## Setup

```
1. Open extractart_colab.ipynb in Google Colab
2. Runtime → Change runtime type → T4 GPU → Save
3. Upload 9 CSV files to Google Drive:
   My Drive → ExtractArt → wikiart_csv
```

---

## Model Architecture

ExtractArt uses a **Convolutional-Recurrent Neural Network (CRNN)** that processes each painting as a sequence of spatial patches, then classifies it across three tasks simultaneously.

### Stage 1 — Feature Extraction (ConvNeXt-Base)
The backbone is **ConvNeXt-Base**, a modern CNN pretrained on ImageNet. Unlike older backbones like ResNet, ConvNeXt uses large 7×7 depthwise convolution kernels which are particularly well suited for art — they capture brushstroke textures, colour gradients, and compositional patterns that smaller 3×3 kernels miss. Each painting is processed into a 7×7 grid of feature maps, each patch representing a 32×32 region of the painting.

### Stage 2 — Sequence Modelling (BiLSTM)
The 49 patch features are passed through a **2-layer Bidirectional LSTM**. This is the recurrent part of the architecture — the LSTM reads the patches as a sequence, learning spatial relationships between different regions of the painting (e.g. how the sky relates to the foreground). Being bidirectional means it reads the sequence both left-to-right and right-to-left, giving each patch context from both directions.

### Stage 3 — Attention Pooling
An **attention mechanism** weighs each of the 49 patch outputs, learning which regions of the painting are most important for classification. Patches with distinctive style features (brushstrokes, lighting) get higher attention weights than uniform background regions.

### Stage 4 — Embedding + Classification Heads
The attended features are compressed into a **512-dimensional L2 normalised embedding** — a compact fingerprint of the painting. Three separate linear classification heads then predict Style, Genre, and Artist simultaneously from this shared embedding.

```
```

---

## Tools and Libraries Used

| Tool | Purpose |
|------|---------|
| **PyTorch** | Deep learning framework — model definition, training loop, GPU acceleration |
| **TorchVision** | Provides ConvNeXt-Base pretrained weights and image transforms |
| **ConvNeXt-Base** | CNN backbone pretrained on ImageNet-1k (2022, Meta AI) |
| **BiLSTM** | Recurrent layer for sequential patch modelling |
| **scikit-learn** | F1 score, classification report, Isolation Forest |
| **scipy** | Mahalanobis distance for outlier detection |
| **Pandas** | Loading and managing CSV label files |
| **PIL / Pillow** | Image loading and preprocessing |
| **tqdm** | Progress bars during training and downloading |
| **Google Colab** | Cloud GPU environment (T4, 16GB VRAM) |
| **Google Drive** | Persistent storage for images, checkpoints, outputs |

---

## Training Details

| Setting | Value | Reason |
|---------|-------|--------|
| Loss | Focal Loss + Label Smoothing | Handles rare artist classes |
| Scheduler | CosineAnnealingWarmRestarts | Periodic LR restarts escape local minima |
| Class weights | Inverse frequency per task | Prevents majority classes dominating |
| Batch size | 8 | Safe for T4 16GB with ConvNeXt-Base |
| Epochs | 25 | Sufficient for convergence on 10k images |
| Early stopping | Patience 15 | Stops if val accuracy plateaus |

---

## Outlier Detection

Three independent signals are combined per painting:

| Signal | Method | Flags when |
|--------|--------|-----------|
| A | Softmax confidence | Max class probability < 0.30 |
| B | Mahalanobis distance | Embedding > mean + 2σ from class centre |
| C | Isolation Forest | Global anomaly in embedding space |

Paintings flagged by 2 or more signals are ranked as strong outliers and saved to `outliers_*.csv`.

---

## Running the Notebook

| Cell | What it does |
|------|-------------|
| 1 | Verify GPU |
| 2 | Mount Google Drive |
| 3 | Verify CSV files |
| 4 | Install scipy |
| 5 | Test WikiArt URLs |
| 6 | Download images |
| 7 | Define model classes |
| 8 | Train |
| 9 | Evaluate |
| 10 | Extract embeddings + detect outliers |

---

## Configurable Settings

**Cell 6 — Images to download**
```python
MAX_IMAGES = 10000   # ~5 GB recommended
MAX_IMAGES = 3000    # ~2 GB quick test
```

**Cell 8 — Training**
```python
EPOCHS     = 25   # use 2 first to test pipeline
BATCH_SIZE = 8    # keep at 8 for T4
```

---

## Outputs

```
ExtractArt/checkpoints/multitask_best.pth    best model weights
ExtractArt/features/*_embeddings.npy         512-d painting embeddings
ExtractArt/checkpoints/outliers_style.csv    style outliers
ExtractArt/checkpoints/outliers_genre.csv    genre outliers
ExtractArt/checkpoints/outliers_artist.csv   artist outliers
```



## After Every Runtime Restart

```
Cell 2 → remount Drive
Cell 4 → reinstall scipy
Cell 7 → redefine model classes
Cell 8 → retrain
```
