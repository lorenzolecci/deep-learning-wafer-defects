<div align="center">

# Deep Learning for Wafer Defect Classification

**A reproducible study of domain-specific CNNs, model compression, data augmentation, optimizer selection, and transfer learning for nine-class wafer-map classification.**

[![Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/lorenzolecci/deep-learning-wafer-defects)
[![LinkedIn](https://img.shields.io/badge/Lorenzo_Lecci-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lorenzo-lecci-793789297/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)](https://opensource.org/license/mit)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep_Learning-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Metrics-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square)](https://matplotlib.org/)
[![scikit-image](https://img.shields.io/badge/scikit--image-Image_Processing-4C72B0?style=flat-square)](https://scikit-image.org/)
[![VisualKeras](https://img.shields.io/badge/VisualKeras-Architecture_Diagrams-6C3483?style=flat-square)](https://github.com/paulgavrikov/visualkeras)

</div>

---

## Overview

Wafer defect maps provide a spatial view of manufacturing failures in semiconductor production. Their classification is challenging because the dataset is highly imbalanced, several classes are defined by subtle spatial differences, and the maps contain discrete values rather than natural-image intensities.

This project develops a controlled and reproducible deep-learning workflow for classifying nine wafer-map patterns from the **WM-811K** dataset:

`center`, `donut`, `edge-loc`, `edge-ring`, `loc`, `near-full`, `none`, `random`, and `scratch`.

The work compares:

- a full custom CNN;
- a parameter-efficient intermediate CNN;
- initialization, regularization, augmentation, and optimizer choices;
- frozen and fine-tuned MobileNetV2 transfer learning;
- domain-safe augmentation against arbitrary-angle rotations.

The main result is that a compact domain-specific CNN offers the strongest balance between predictive performance, efficiency, and reproducibility.

## Key results

| Validation-selected model | Parameters | Test accuracy | Test balanced accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|
| **Full CNN — Nadam** | 897,673 | **95.68%** | **87.32%** | **83.14%** |
| **Intermediate CNN — SGD + Nesterov** | 505,449 | **95.80%** | **85.33%** | **82.29%** |
| **MobileNetV2 — fine-tuned** | 2,269,513 | 81.03% | 67.14% | 59.46% |

The intermediate CNN uses **43.7% fewer parameters** than the full CNN while retaining approximately **99% of its test Macro F1**.

MobileNetV2 contains more than twice the parameters of the full CNN but performs substantially worse. This suggests that ImageNet features transfer poorly to discrete, monochromatic wafer maps, where absolute position and geometric structure are more important than natural-image texture.

Detailed result files:

- [Full CNN test results](project/cnn_outputs/large_test_results.csv)
- [Intermediate CNN test results](project/cnn_outputs_intermediate/large_test_results.csv)
- [MobileNetV2 test results](project/cnn_outputs_mobilenetv2/large_test_results.csv)

## Why this problem is difficult

The labeled dataset contains **172,950 wafer maps**, but the class distribution is extremely uneven:

- `none`: 147,431 samples;
- `edge-ring`: 9,680 samples;
- `near-full`: 149 samples.

Ordinary accuracy is therefore misleading. A classifier biased toward `none` can achieve a high accuracy while failing on manufacturing defects.

For this reason, the project uses:

- **Macro F1** as the primary model-selection metric;
- **balanced accuracy** as a secondary metric;
- per-class precision, recall, and confusion matrices for diagnosis;
- ordinary accuracy only as a descriptive metric.

## Data preprocessing

Wafer maps have variable native dimensions and use discrete values to encode the wafer background, functioning dies, and defective dies.

Nearest-neighbor interpolation resizes every map to `56 × 56` without introducing artificial intermediate values. A channel dimension is then added, and the model rescales the discrete states from `{0, 1, 2}` to `{0, 0.5, 1}`.

<p align="center">
  <img
    src="presentation_outputs\figures\preprocessing\scratch_augmentation_comparison.png"
    alt="Wafer map preprocessing pipeline"
    width="100%"
  >
</p>

### Data protocol

| Component | Design |
|---|---|
| Training partition | 65% pool, capped at 3,000 samples per class |
| Validation partition | 15%, natural class distribution |
| Test partition | 20%, natural class distribution |
| Image size | 56 × 56 × 1 |
| Resize method | Nearest-neighbor |
| Random seed | 42 |
| Primary metric | Validation Macro F1 |
| Final test size | 34,590 wafer maps |

The training cap reduces majority-class dominance without altering validation or test distributions. Samples removed by the cap are not moved into the other partitions.

## CNN architectures

Both custom CNNs follow the same overall structure: convolutional feature extraction, Batch Normalization, spatial downsampling, a dense representation, dropout, and a nine-class softmax classifier.

### Intermediate CNN

The intermediate model reduces the convolutional widths to `24 → 48 → 96` and uses a 96-unit dense representation. It contains 505,449 parameters, approximately 43.7% fewer than the full model.

<p align="center">
  <img
    src="presentation_outputs\figures\intermediate_cnn_architecture_shapes_presentation.png"
    alt="Intermediate CNN architecture"
    width="100%"
  >
</p>

### Full CNN

The full model uses convolutional widths of `32 → 64 → 128`, followed by a 128-unit dense layer. It contains 897,673 parameters.

<p align="center">
  <img
    src="presentation_outputs\figures\cnn_architecture_shapes_presentation.png"
    alt="Full CNN architecture"
    width="100%"
  >
</p>

The controlled comparison keeps preprocessing, data partitions, regularization, optimizer selection, and evaluation unchanged across the two architectures.

## Controlled experiments

The notebooks isolate one design choice at a time.

| Experiment | Question |
|---|---|
| Baseline | How does a standard CNN perform without additional optimization? |
| He initialization | Does initialization designed for ReLU improve convergence? |
| Arbitrary rotations | Do generic image augmentations preserve wafer-map semantics? |
| Safe flips | Can orientation-invariant augmentation improve generalization without interpolation? |
| L2 regularization | Does weight decay improve minority-class performance? |
| Optimizer study | How do Adam, Nadam, RMSprop, and SGD with Nesterov compare? |
| Architecture reduction | How much capacity can be removed without losing performance? |
| Transfer learning | Do ImageNet-pretrained MobileNetV2 features transfer to wafer maps? |

## Data augmentation: a domain-specific decision

Wafer maps contain discrete values:

```text
0 = background
1 = functioning die
2 = defective die
```

Arbitrary-angle rotations require interpolation and create artificial intermediate pixel values. They can also blur narrow structures such as scratches.

The project deliberately retains this transformation as a negative experiment and compares it with horizontal and vertical flips, which preserve the discrete values. The results show that augmentation should be selected according to the physical meaning of the data rather than copied from natural-image pipelines.

## Transfer learning findings

MobileNetV2 was evaluated in two stages:

1. frozen ImageNet backbone with a new classification head;
2. fine-tuning of the final backbone layers with BatchNormalization kept frozen.

The input pipeline maps the three discrete states from `0, 1, 2` to `-1, 0, 1` before repeating the single channel three times.

Fine-tuning improves Macro F1 over the frozen backbone, but the model remains weaker than the custom CNNs. Confusion matrices show particular difficulty with spatially dependent classes such as `center`, `edge-loc`, `loc`, and `scratch`.

This result is useful in its own right: a larger pretrained network is not automatically the best model for highly structured industrial data.

## Reproducibility and auditability

The project includes technical checks beyond setting a random seed:

- Python, NumPy, and TensorFlow random states are reset before each model;
- deterministic TensorFlow operations are enabled when available;
- training, validation, and test indices are saved;
- the same split hashes are verified across architectures;
- dataset SHA-256 hashes are recorded;
- each model uses a distinct checkpoint path;
- checkpoint files are reloaded before evaluation;
- stored metrics are checked against recalculated metrics;
- test data is excluded from model selection.

Audit files are available under:

```text
project/cnn_outputs/audit/
project/cnn_outputs_intermediate/audit/
project/cnn_outputs_mobilenetv2/audit/
```

## Repository structure

```text
deep-learning-wafer-defects/
├── assets/
│   └── images/
│       ├── architecture_full.png
│       ├── architecture_intermediate.png
│       └── preprocessing_pipeline.png
├── datasets/
│   ├── LSWMD.pkl
│   └── Dataset.pkl
├── project/
│   ├── CNNs_full.ipynb
│   ├── CNNs_intermediate.ipynb
│   ├── utils.py
│   ├── cnn_outputs/
│   ├── cnn_outputs_intermediate/
│   └── cnn_outputs_mobilenetv2/
├── scripts/
│   ├── Cleaned_dataset_creation.py
│   ├── Presentation_graphs.ipynb
│   └── Wafer_preprocessing_and_augmentation_images.ipynb
├── .gitignore
└── README.md
```

Dataset pickle files and trained weight files are excluded from version control.

## Getting started

### Clone the repository

```bash
git clone https://github.com/lorenzolecci/deep-learning-wafer-defects.git
cd deep-learning-wafer-defects
```

### Create an environment

```bash
python -m venv .venv
```

Activate it and install the required libraries:

```bash
pip install tensorflow numpy pandas scikit-learn scikit-image matplotlib seaborn tqdm jupyter pillow visualkeras
```

A GPU-compatible TensorFlow installation is recommended but not required.

### Prepare the dataset

Obtain `LSWMD.pkl` from the WM-811K dataset and place it in:

```text
datasets/LSWMD.pkl
```

Then create the cleaned dataset:

```bash
python scripts/Cleaned_dataset_creation.py
```

The generated file is:

```text
datasets/Dataset.pkl
```

### Run the experiments

Launch Jupyter:

```bash
jupyter lab
```

Recommended order:

1. [Full CNN experiments](project/CNNs_full.ipynb)
2. [Intermediate CNN experiments](project/CNNs_intermediate.ipynb)
3. [Presentation and model comparison](scripts/Presentation_graphs.ipynb)

Supporting files:

- [Shared experiment utilities](project/utils.py)
- [Dataset preparation script](scripts/Cleaned_dataset_creation.py)
- [Preprocessing and augmentation visualizations](scripts/Wafer_preprocessing_and_augmentation_images.ipynb)

## Skills demonstrated

This project showcases practical experience with:

- convolutional neural network design;
- transfer learning and fine-tuning;
- class-imbalanced classification;
- controlled ablation studies;
- data augmentation for non-natural images;
- hyperparameter and optimizer comparison;
- model compression and efficiency analysis;
- reproducible machine-learning experiments;
- checkpoint and dataset auditing;
- confusion-matrix and per-class error analysis;
- technical documentation and visual communication.

## Limitations and future work

The current experiments use a stratified image-level split. A lot-level split would provide a stricter estimate of generalization to unseen production lots.

Promising next steps include:

- a compact residual CNN with explicit coordinate channels;
- multi-scale or spatial-attention modules for localized defects;
- lot-aware validation;
- probability calibration;
- inference benchmarking and lightweight deployment;
- evaluation on additional wafer-map datasets.

## Authors

| Author | Links |
|---|---|
| **Lorenzo Lecci** | [GitHub](https://github.com/lorenzolecci) · [LinkedIn](https://www.linkedin.com/in/lorenzo-lecci-793789297/) |
| **Teodoro Pacetti** | Project co-author |

## License

This project is released under the **MIT License**.

---

<div align="center">

**Built as a reproducible deep-learning study for semiconductor wafer-map classification.**

[View the repository](https://github.com/lorenzolecci/deep-learning-wafer-defects) · [Connect on LinkedIn](https://www.linkedin.com/in/lorenzo-lecci-793789297/)

</div>
