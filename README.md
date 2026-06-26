# MCFA-Net: Multi-scale Cross-domain Frequency-Aware Network for Brain Tumour Classification

MCFA-Net is a deep learning framework for multi-class brain tumour classification from MRI images. The framework is built upon **EfficientNet-B3** and integrates spatial and frequency-domain feature learning, adaptive multi-scale feature aggregation, and prototype-guided latent representation learning.

The repository also includes complete ablation studies, statistical significance testing, Grad-CAM visualization, and publication-quality result generation.

---

# Architecture

```
Input MRI
     │
     ▼
EfficientNet-B3 Backbone
     │
     ▼
Dual-Domain Attention Fusion (DDAF)
     │
     ▼
Adaptive Multi-scale Pyramid Pooling (AMPP)
     │
     ▼
Hybrid Prototype Contrastive Bottleneck (HPCB)
     │
     ▼
Classifier
```

---

# Implemented Models

The framework automatically trains the following models.

| Model | Description |
|--------|-------------|
| EfficientNet_B3 | Baseline EfficientNet-B3 |
| MCFA_no_DDAF | AMPP + HPCB only |
| MCFA_no_HPCB | DDAF + AMPP only |
| MCFA_no_AMPP | DDAF + HPCB only |
| MCFA_Net_EfficientNet_B3 | Full proposed MCFA-Net |

---

# Main Components

## 1. Dual-Domain Attention Fusion (DDAF)

- Spatial feature extraction using EfficientNet-B3
- Frequency representation using:
  - Haar Wavelet decomposition
  - FFT magnitude spectrum
- Learnable attention gate
- Dynamic fusion of spatial and frequency features

---

## 2. Adaptive Multi-scale Pyramid Pooling (AMPP)

Extracts contextual information at multiple spatial scales.

Pooling scales:

- 1 × 1
- 2 × 2
- 3 × 3
- 6 × 6

A learnable attention mechanism dynamically weights each scale.

---

## 3. Hybrid Prototype Contrastive Bottleneck (HPCB)

The bottleneck consists of

- learnable class prototypes
- prototype-aware latent encoding
- variational latent sampling
- prototype contrastive learning

Loss components include

- Cross-Entropy Loss
- KL Divergence
- Prototype Contrastive Loss

---

# Training Configuration

| Parameter | Value |
|------------|-------|
| Backbone | EfficientNet-B3 |
| Image Size | 224 × 224 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Batch Size | 16 |
| Epochs | 60 |
| Early Stopping | 12 epochs |
| Latent Dimension | 512 |
| Prototype Dimension | 256 |
| Label Smoothing | 0.10 |
| MixUp Alpha | 0.4 |
| Random Seeds | 42–47 |

---

# Data Augmentation

Training images undergo

- Random Resized Crop
- Horizontal Flip
- Vertical Flip
- Random Rotation
- Color Jitter
- Random Grayscale
- ImageNet Normalization

Validation and testing use deterministic resizing and normalization.

---

# Dataset Structure

```
Training/         (Kaggle training directory acquired from - https://www.kaggle.com/datasets/mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images )
    Glioma/
    meningioma/
    pituitary/
    no tumor/

Test/         (Mendeley testing directory acquired from- https://data.mendeley.com/datasets/zwr4ntf94j/1 )
    Glioma/  
    meningioma/
    pituitary/
    no tumor/
```

Update the following paths before training:

```python
TRAIN_PATH = "path/to/Training/"
TEST_PATH  = "path/to/Test/"
```

---

# Installation

Install the required packages.

```bash
pip install torch torchvision timm
pip install PyWavelets
pip install numpy pandas scipy
pip install matplotlib seaborn
pip install scikit-learn
```

---

# Running the Code

Simply execute

```bash
python MCFA_Net_main.py
```

The script automatically

- trains all models
- evaluates each model
- performs six independent runs
- generates Grad-CAM visualizations
- computes statistical significance tests
- exports publication-quality figures

---

# Output Directory

```
MCFA_Net_RESULTS_2026_final_imp/

├── EfficientNet_B3/
│   ├── metrics/
│   ├── curves/
│   └── gradcam/
│
├── MCFA_no_DDAF/
├── MCFA_no_HPCB/
├── MCFA_no_AMPP/
├── MCFA_Net_EfficientNet_B3/
│
├── all_results.csv
├── summary_aggregated.csv
├── ablation_summary.csv
├── wilcoxon_n6.csv
├── mcnemar_pooled.csv
├── superiority_table.csv
├── ablation_superiority.csv
├── boxplot_all_metrics.png
├── barplot_acc.png
├── barplot_f1_macro.png
├── barplot_precision_macro.png
├── barplot_recall_macro.png
├── barplot_auc_macro.png
├── cohens_d_heatmap.png
└── mcnemar_pvalue_heatmap.png
```

---

# Evaluation Metrics

The framework reports

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Macro AUC
- Confusion Matrix
- ROC Curves
- Classification Report

---

# Statistical Analysis

The repository includes automated statistical evaluation.

## Wilcoxon Signed-Rank Test

Performed across six independent random seeds.

Outputs

```
wilcoxon_n6.csv
```

---

## McNemar Test

Performed using pooled predictions across all seeds.

Outputs

```
mcnemar_pooled.csv
```

---

## Effect Size

Cohen's d is computed for pairwise model comparisons.

Outputs

```
cohens_d_heatmap.png
```

---

# Visualization

The repository automatically generates

- Training curves
- Validation curves
- Confusion matrices
- ROC curves
- Grad-CAM heatmaps
- Metric boxplots
- Mean ± standard deviation bar plots
- McNemar significance heatmaps
- Cohen's d heatmaps

---

# Reproducibility

The experiments use fixed random seeds

```
42
43
44
45
46
47
```

The code enables deterministic CUDA execution where supported to improve experimental reproducibility.

---



# Hardware

Experiments were conducted on a Dell Alienware m16 laptop with:

* GPU: NVIDIA GeForce RTX [4060] ([8] GB)
* CPU: Intel Core [Intel Core Ultra 9 185H ]
* RAM: [16] GB

Addtional Support (HPC):

* NVIDIA H200 GPUs (HPC cluster)



Training was performed using CUDA-enabled PyTorch.

---

# Runtime

The full experimental run (all models across all seeds) requires approximately 4 days on the specified hardware.

Runtime may vary depending on GPU capability.

A single model run (one seed) typically takes several hours.

---

# Citation

If you use this repository in your research, please cite the corresponding publication once available.







