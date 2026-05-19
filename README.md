# MCFA-Net: Multi-scale Cross-domain Frequency-Aware Network

## Overview

MCFA-Net is a deep learning framework for brain tumour MRI classification that integrates cross-domain frequency-aware representation learning, adaptive multi-scale contextual aggregation, and prototype-conditioned latent representation modelling within a unified architecture.

The framework is designed to address three major limitations commonly observed in medical image classification systems:

1. Weak integration between spatial and frequency-domain evidence.
2. Loss of contextual information across varying tumour scales.
3. Poorly structured latent representations with limited class separability.

The proposed framework incorporates three synergistic modules:

* Dual-Domain Attention Fusion (DDAF)
* Adaptive Multi-scale Pyramid Pooling (AMPP)
* Hierarchical Prototype Contrastive Bottleneck (HPCB)

EfficientNet-B3 is employed as the backbone feature extractor.

---

#  Novelty Assessment

From a strict SCIE-level evaluation perspective, the strongest novelty of MCFA-Net is not the use of EfficientNet, wavelets, FFTs, or prototype learning individually. The novelty arises from the structured interaction between:

* frequency-aware fusion,
* adaptive multi-scale contextual aggregation,
* and prototype-conditioned variational latent modelling.

The most scientifically significant contribution is the:

> prototype-conditioned variational latent representation formulation within the HPCB module.

This component moves beyond conventional feature engineering and attempts explicit latent space structuring using:

* learnable class prototypes,
* stochastic latent sampling,
* contrastive prototype optimisation,
* and similarity-guided contextual aggregation.

This gives the framework stronger representation-learning depth than standard architectural medical imaging papers.

---

# Main Contributions

## 1. Dual-Domain Attention Fusion (DDAF)

The DDAF module integrates:

* spatial-domain features,
* wavelet-domain representations,
* and Fourier-domain representations.

Unlike fixed fusion mechanisms, DDAF employs a learnable input-aware gating strategy that dynamically weights spatial and frequency evidence.

### Key Characteristics

* Simultaneous spatial-frequency feature integration.
* Learnable attention-based fusion.
* Input-dependent feature weighting.
* Wavelet + FFT complementary representation learning.

### Scientific Importance

Most existing medical image classification frameworks either:

* operate only in the spatial domain, or
* use frequency-domain processing as an auxiliary branch.

MCFA-Net instead performs adaptive cross-domain evidence fusion.

---

## 2. Adaptive Multi-scale Pyramid Pooling (AMPP)

The AMPP module captures tumour characteristics across multiple spatial scales using:

* 1×1 pooling,
* 2×2 pooling,
* 3×3 pooling,
* and 6×6 pooling.

An attention-based weighting mechanism dynamically determines which spatial scale contributes most strongly for a given sample.

### Key Characteristics

* Adaptive multi-scale contextual aggregation.
* Input-aware scale weighting.
* Dynamic pooling importance estimation.
* Improved tumour size and location sensitivity.

### Scientific Importance

Conventional global average pooling discards scale-specific contextual information.
AMPP attempts to preserve discriminative spatial structure through adaptive scale selection.

---

## 3. Hierarchical Prototype Contrastive Bottleneck (HPCB)

The HPCB module is the core novelty of MCFA-Net.

The module introduces:

* learnable class prototypes,
* prototype similarity estimation,
* variational latent sampling,
* and contrastive latent optimisation.

The latent distribution is conditioned using prototype-guided contextual aggregation.

### Key Characteristics

* Prototype-conditioned latent representation learning.
* Variational bottleneck formulation.
* Structured latent space organisation.
* Intra-class compactness enhancement.
* Inter-class separability improvement.
* Prototype-guided stochastic sampling.

### Scientific Importance

Most prototype-learning methods use deterministic embeddings.
Most variational bottleneck methods lack explicit class structure.

MCFA-Net combines:

* prototype learning,
* contrastive learning,
* and variational inference

within a unified latent representation framework.

This is the primary source of conceptual novelty in the proposed architecture.

---

# Architecture Pipeline

Input MRI
→ EfficientNet-B3 Backbone
→ DDAF (Spatial + Frequency Fusion)
→ AMPP (Adaptive Multi-scale Aggregation)
→ HPCB (Prototype-conditioned Variational Bottleneck)
→ Classification Head

---

# Ablation Models

The framework includes three ablation variants:

| Model        | Purpose                                                    |
| ------------ | ---------------------------------------------------------- |
| MCFA_no_DDAF | Evaluates AMPP + HPCB without frequency-aware fusion       |
| MCFA_no_HPCB | Evaluates DDAF + AMPP without prototype bottleneck         |
| MCFA_no_AMPP | Evaluates DDAF + HPCB without adaptive multi-scale pooling |

These ablations isolate the contribution of each proposed component.

---

# Dataset Organisation and Alignment

## Dataset Structure

Organise the dataset exactly as follows:

```text
Training/
    glioma/
    meningioma/
    pituitary/
    notumor/

Test/
    glioma/
    meningioma/
    pituitary/
    notumor/
```

This folder alignment is mandatory for compatibility with the PyTorch ImageFolder pipeline used in MCFA-Net.

Each subdirectory corresponds to a single tumour category, and all MRI images belonging to that category must be stored within the associated folder.

This alignment ensures:

* deterministic class indexing,
* label consistency across training and testing,
* reproducible category mapping,
* and stratified sampling compatibility.

---

## Data Alignment and Preprocessing

All MRI images are aligned into a unified processing pipeline prior to training.

### Image Standardisation

All images are:

* converted to RGB format,
* resized to 224 × 224 pixels,
* normalised using ImageNet statistics,
* and processed using identical augmentation pipelines.

### Training Augmentation Pipeline

The training set includes:

* random resized cropping,
* horizontal flipping,
* vertical flipping,
* random rotation,
* colour jittering,
* grayscale augmentation,
* and Mixup regularisation.

### Validation and Testing Alignment

Validation and testing images are processed using deterministic resizing and normalisation without stochastic augmentation.

This prevents train-test leakage and ensures fair evaluation.

---

## Stratified Dataset Splitting

The training dataset is further divided into:

* training subset,
* and validation subset

using stratified sampling.

This ensures:

* balanced class distribution,
* reduced sampling bias,
* and statistically consistent evaluation across seeds.

Independent runs are performed using six random seeds:

42, 43, 44, 45, 46, and 47.

This multi-seed alignment improves reproducibility and robustness assessment.

---

# Experimental Setup

## Hardware

* Dell Alienware m16
* Intel Core Ultra 9 185H
* NVIDIA RTX 4060 (8 GB VRAM)
* 16 GB RAM
* Additional experiments on NVIDIA H200 HPC GPUs

## Software

* Python 3.10
* PyTorch
* Torchvision
* timm
* NumPy
* pandas
* SciPy
* scikit-learn
* PyWavelets
* matplotlib
* seaborn

## Training Configuration

| Parameter       | Value                         |
| --------------- | ----------------------------- |
| Image Size      | 224 × 224                     |
| Batch Size      | 16                            |
| Optimizer       | AdamW                         |
| Learning Rate   | 1 × 10⁻⁴                      |
| Weight Decay    | 1 × 10⁻²                      |
| Epochs          | 60                            |
| Early Stopping  | 12 epochs                     |
| Scheduler       | Cosine Annealing Warm Restart |
| Label Smoothing | 0.10                          |
| Mixup Alpha     | 0.4                           |

---

# Statistical Validation

The framework includes rigorous statistical evaluation across six independent runs.

## Statistical Methods

* Wilcoxon Signed-Rank Test
* McNemar Test
* Cohen’s d Effect Size Analysis

## Purpose

These analyses evaluate:

* statistical significance,
* prediction consistency,
* robustness across seeds,
* and practical effect magnitude.

This substantially strengthens scientific validity compared with single-run reporting.

---

# Output Directory Structure

All experimental outputs are automatically organised into structured directories for reproducibility, statistical analysis, and publication-ready visualisation generation.

The outputs are saved as follows:

```text
MCFA_Net_RESULTS_2026_final_imp/
│
├── EfficientNet_B3/
├── MCFA_no_DDAF/
├── MCFA_no_HPCB/
├── MCFA_no_AMPP/
├── MCFA_Net_EfficientNet_B3/
│
│   ├── metrics/
│   │   ├── test_metrics_seedXX.csv
│   │   ├── cm_seedXX.csv
│   │   ├── roc_data_seedXX.csv
│   │   └── history_seedXX.csv
│   │
│   ├── curves/
│   │   └── curves_seedXX.png
│   │
│   └── gradcam/
│       ├── glioma/
│       ├── meningioma/
│       ├── pituitary/
│       └── notumor/
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

## Output Alignment Description

### metrics/

Stores:

* classification metrics,
* confusion matrices,
* ROC curve numerical values,
* and training history logs.

### curves/

Stores publication-quality:

* loss curves,
* and accuracy curves.

### gradcam/

Stores GradCAM explainability visualisations organised class-wise.

Each GradCAM image is saved according to:

* tumour category,
* seed number,
* and image index.

### Root-Level Statistical Files

The root directory stores:

* Wilcoxon signed-rank statistical analysis,
* pooled McNemar statistical testing,
* Cohen’s d effect size comparisons,
* ablation summaries,
* and superiority analysis tables.

This structured output alignment improves:

* reproducibility,
* traceability,
* experimental transparency,
* and manuscript preparation.

---

# Outputs

The implementation generates:

* confusion matrices,
* ROC curves,
* GradCAM visualisations,
* statistical comparison tables,
* effect size heatmaps,
* and ablation analysis reports.

---

# Strengths

## Major Strengths

* Strong representation-learning formulation.
* Cross-domain spatial-frequency modelling.
* Structured latent space optimisation.
* Comprehensive ablation analysis.
* Multi-run statistical validation.
* Large effect size reporting.
* Explainability using GradCAM.

---

# Limitations

## Strict Reviewer-Level Limitations

The following concerns may still be raised by strict reviewers:

1. The framework is architecturally complex.
2. Computational overhead may be higher than lightweight baselines.
3. Theoretical justification of prototype-conditioned variational sampling must be mathematically rigorous.
4. External dataset generalisation should be validated.
5. Clinical interpretability remains limited.
6. Multi-centre robustness is not yet demonstrated.

These limitations should be acknowledged transparently in publication.

---

# Strict SCIE-Level Positioning

MCFA-Net should be positioned as:

> a structured latent representation learning framework for medical image classification.

It should NOT be framed merely as:

* another hybrid CNN architecture,
* or a simple frequency-enhanced EfficientNet variant.

The strongest scientific contribution is:

> prototype-conditioned variational latent organisation.

This is the conceptual identity of the work.

---

# Citation

If this framework is used in academic work, please cite the associated manuscript.

---


