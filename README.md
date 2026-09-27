# Blood Group Detection from Fingerprints

A deep-learning project that classifies fingerprint images into one of eight ABO/Rh blood groups:

**A+, A−, B+, B−, AB+, AB−, O+, O−**

The project uses a hybrid **CNN + Graph Neural Network (GNN)** architecture. A Flask web application is included for making predictions from uploaded fingerprint images or webcam captures.

> **Research and educational project only.** There is no established biological relationship between fingerprint patterns and blood groups. The results of this project should not be used for medical diagnosis or treatment decisions.

---

## Overview

The model combines image-based feature extraction with graph-based learning.

A fingerprint image is first passed through an **EfficientNet-B3** backbone to extract spatial feature maps. These feature maps are then converted into a graph, where each spatial region becomes a node. A Graph Attention Network processes the graph and learns relationships between different regions of the fingerprint.

The final representation is passed through an MLP classifier to predict one of the eight blood groups.

### Model Pipeline

```text
Fingerprint Image (224 × 224)
            │
            ▼
    EfficientNet-B3
    ImageNet Pretrained
            │
            ▼
   7 × 7 × 1536 Feature Map
            │
            ▼
      Graph Conversion
     49 Nodes + k-NN Edges
            │
            ▼
 Graph Attention Network (GAT)
   Multi-head GATConv Layers
   BatchNorm + ELU + Dropout
            │
            ▼
    Global Mean + Max Pooling
            │
            ▼
          MLP Head
      512 → 256 → 128 → 8
            │
            ▼
      Blood Group Class
```

---

## Test-Time Augmentation

During inference, the model uses **Test-Time Augmentation (TTA)**.

Each test image is evaluated in seven different versions:

1. Original image
2. Horizontal flip
3. Vertical flip
4. +10° rotation
5. −10° rotation
6. Brightness adjustment
7. Contrast adjustment

The predicted class probabilities from all seven versions are averaged to produce the final prediction.

---

## Results

The test set contains **1,600 images**, with **200 images per class**.

Using seven-augmentation TTA, the model achieved:

### Overall Accuracy

**79.31%**

### Per-Class Accuracy

| Blood Group | Accuracy |
| ----------- | -------: |
| A+          |    90.5% |
| A−          |    89.5% |
| AB+         |    86.0% |
| AB−         |    97.0% |
| B+          |    59.5% |
| B−          |    69.0% |
| O+          |    84.0% |
| O−          |    59.0% |

The most common classification errors were:

* **O− → AB−**
* **B+ → AB+ / AB−**

Confusion matrices and other evaluation images are available in:

```text
static/images/
```

---

## Dataset

The project uses a fingerprint-based blood group dataset.

The dataset is **not included in this repository**. Images should be arranged using the standard `ImageFolder` structure used by torchvision.

```text
dataset/
├── train/
│   ├── A+/
│   ├── A-/
│   ├── AB+/
│   ├── AB-/
│   ├── B+/
│   ├── B-/
│   ├── O+/
│   └── O-/
│
└── test/
    ├── A+/
    ├── A-/
    ├── AB+/
    ├── AB-/
    ├── B+/
    ├── B-/
    ├── O+/
    └── O-/
```

---

## Tech Stack

### Machine Learning

* Python
* PyTorch
* torchvision
* EfficientNet-B3
* efficientnet-pytorch
* PyTorch Geometric
* Graph Attention Networks (GAT)
* scikit-learn

### Data and Visualization

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Pillow

### Web Application

* Flask
* HTML
* CSS
* JavaScript

The web application supports:

* Image upload
* Webcam fingerprint capture
* Model prediction
* Prediction statistics

---

## Project Structure

```text
├── app.py
├── config.py
├── model.py
│
├── train.py
├── train_recovery_92plus.py
├── train_to_90_phase2.py
│
├── tta_ensemble.py
├── ensemble_model.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── images/
│
├── requirements.txt
└── README.md
```

### Main Files

| File                       | Description                                                           |
| -------------------------- | --------------------------------------------------------------------- |
| `app.py`                   | Flask application for prediction                                      |
| `config.py`                | Paths, hyperparameters, and class configuration                       |
| `model.py`                 | EfficientNet backbone, graph conversion, GAT layers, and hybrid model |
| `train.py`                 | Main model training loop                                              |
| `train_recovery_92plus.py` | Fine-tuning from a saved checkpoint using a lower learning rate       |
| `train_to_90_phase2.py`    | Fine-tuning with stronger augmentation and AdamW                      |
| `tta_ensemble.py`          | Test-Time Augmentation evaluation                                     |
| `ensemble_model.py`        | Multi-checkpoint ensemble evaluation                                  |
| `templates/index.html`     | Web application interface                                             |
| `static/`                  | CSS, JavaScript, and evaluation images                                |

---

## Model Architecture

### 1. EfficientNet-B3

The input fingerprint image is resized to:

```text
224 × 224
```

An ImageNet-pretrained EfficientNet-B3 extracts a spatial feature map of approximately:

```text
7 × 7 × 1536
```

### 2. Feature Map to Graph

The `7 × 7` spatial feature map is converted into a graph.

Each spatial cell becomes a node:

```text
7 × 7 = 49 nodes
```

Edges are created using a **k-nearest-neighbor graph with k = 8**.

This allows the model to process relationships between different spatial regions of the fingerprint.

### 3. Graph Attention Network

The graph is processed using multiple GAT layers with:

* Multi-head attention
* Batch normalization
* ELU activation
* Dropout

### 4. Global Pooling

The graph representation is summarized using:

* Global mean pooling
* Global max pooling

The pooled representations are combined before being passed to the classifier.

### 5. Classification Head

The final MLP contains the following layers:

```text
512 → 256 → 128 → 8
```

The final eight outputs correspond to:

```text
A+
A-
B+
B-
AB+
AB-
O+
O-
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/varshreddyy/<repository-name>.git
cd <repository-name>
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Some scripts also require:

```bash
pip install seaborn tqdm
```

---

## PyTorch Geometric

PyTorch Geometric dependencies such as `torch-scatter` and `torch-sparse` need to match your installed PyTorch and CUDA versions.

If installation fails, follow the official PyTorch Geometric installation instructions:

https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

For CPU-only environments, make sure the installed PyTorch Geometric packages are compatible with the installed PyTorch version.

---

## Training

After preparing the dataset, run:

```bash
python train.py
```

The best model checkpoint is saved as:

```text
models/best_cnn_gnn_model.pth
```

### Default Training Configuration

The default configuration in `config.py` includes:

| Parameter     |           Default |
| ------------- | ----------------: |
| Image Size    |         224 × 224 |
| Batch Size    |                16 |
| Epochs        |                50 |
| Learning Rate |            0.0005 |
| Optimizer     |              Adam |
| Scheduler     | ReduceLROnPlateau |
| Device        |               CPU |

The default settings are intended to work on CPU-based systems.

If you have access to a GPU, the batch size and other training parameters can be increased depending on available GPU memory.

---

## Fine-Tuning

The repository contains additional training scripts for continuing training from an existing checkpoint.

### Recovery Fine-Tuning

```bash
python train_recovery_92plus.py
```

This script continues training from a saved checkpoint using a lower learning rate.

### Phase 2 Training

```bash
python train_to_90_phase2.py
```

This version uses stronger augmentation and the AdamW optimizer for further fine-tuning.

---

## Evaluation

To evaluate the model using Test-Time Augmentation:

```bash
python tta_ensemble.py
```

The script evaluates the model using multiple augmented versions of each test image and averages the resulting probabilities.

For multi-checkpoint ensemble evaluation:

```bash
python ensemble_model.py
```

---

## Running the Web Application

The Flask application requires a trained model checkpoint:

```text
models/best_cnn_gnn_model.pth
```

Start the application with:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

The application provides an interface for uploading a fingerprint image or capturing one using a webcam.

---

## Limitations

This project is intended for **research and educational purposes**.

There is currently no established biological mechanism showing that fingerprint patterns can reliably determine a person's ABO/Rh blood group. Therefore, the model's predictions should not be interpreted as medically valid blood group identification.

The reported accuracy is specific to the dataset and experimental setup used in this project. Performance may change significantly with:

* Different fingerprint datasets
* Different image quality
* Different fingerprint acquisition devices
* Different populations
* Changes in preprocessing
* Class imbalance
* Dataset bias

The model should not be used for:

* Blood transfusion decisions
* Medical diagnosis
* Patient identification
* Clinical testing
* Emergency medical decisions

Actual blood group determination should be performed using established laboratory methods.

---

## Future Improvements

Possible areas for further experimentation include:

* Testing on an independent external dataset
* Improving class balance
* Evaluating different CNN backbones
* Experimenting with different graph construction methods
* Comparing GAT with other GNN architectures
* Hyperparameter tuning
* Cross-validation
* Improving the web application's prediction interface
* Comparing the hybrid CNN-GNN model against CNN-only and GNN-only baselines

---

## Disclaimer

This repository is a **research and educational experiment**.

The model's predictions are not a substitute for laboratory blood typing and should not be used for medical or clinical decisions.

---


