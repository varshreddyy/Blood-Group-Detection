Blood Group Detection from Fingerprints

A deep-learning experiment that classifies fingerprint images into one of eight ABO/Rh blood groups (A+, A−, B+, B−, AB+, AB−, O+, O−). It uses a hybrid CNN + Graph Neural Network model, and a Flask web app serves the predictions.

Research/educational project only. There is no established biological link between fingerprint patterns and blood group. This model must not be used for medical decisions. See Limitations.

How it works
Fingerprint image (224×224)
        │
        ▼
EfficientNet-B3 (ImageNet-pretrained)  →  7×7×1536 feature map
        │
        ▼
Feature map → graph: 49 nodes (one per cell), k-NN edges (k=8)
        │
        ▼
Graph Attention Network (multi-head GATConv layers, BatchNorm, ELU, Dropout)
        │
        ▼
Global mean + max pooling
        │
        ▼
MLP head (→512 → 256 → 128 → 8)  →  blood group

At inference, predictions use test-time augmentation (TTA). Each image is scored in 7 variants (original, horizontal/vertical flip, ±10° rotation, brightness, contrast), and the class probabilities are averaged.

Results

The 8-class test set has 1,600 images (200 per class). With 7-augmentation TTA, the model reaches 79.31% accuracy.

Class	Accuracy		Class	Accuracy
A+	90.5%		B+	59.5%
A−	89.5%		B−	69.0%
AB+	86.0%		O+	84.0%
AB−	97.0%		O−	59.0%

The most common confusions are O− predicted as AB− and B+ predicted as AB+/AB−. Confusion matrices are in static/images/.

Tech stack
Model: PyTorch, torchvision, efficientnet-pytorch, PyTorch Geometric (GATConv, knn_graph)
Evaluation: scikit-learn, matplotlib, seaborn
Web app: Flask, Pillow, vanilla HTML/CSS/JS (file upload + webcam capture)
Project structure
├── app.py                     # Flask app: /, /predict, /stats
├── config.py                  # Paths, hyperparameters, class list
├── model.py                   # EfficientNet backbone, CNN→graph converter, GAT, hybrid model
├── train.py                   # Base training loop
├── train_recovery_92plus.py   # Fine-tuning from checkpoint (lower LR)
├── train_to_90_phase2.py      # Fine-tuning with heavier augmentation (AdamW)
├── tta_ensemble.py            # TTA evaluation with 3/5/7 augmentations
├── ensemble_model.py          # Multi-checkpoint ensemble evaluation
├── templates/index.html
└── static/                    # CSS, JS, confusion-matrix images
Setup

The repository does not include the dataset or the trained weights.

bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install seaborn tqdm        # used by scripts, not yet in requirements.txt

PyTorch Geometric's torch-scatter / torch-sparse wheels must match your PyTorch and CUDA versions. If the install fails, see the PyG installation guide.

Dataset layout

Arrange the images in torchvision ImageFolder format:

dataset/
├── train/
│   ├── A+/  ├── A-/  ├── AB+/ ├── AB-/
│   ├── B+/  ├── B-/  ├── O+/  └── O-/
└── test/
    └── (same 8 folders)
Usage
bash
# Train (saves best checkpoint to models/best_cnn_gnn_model.pth)
python train.py

# Optional: continue fine-tuning from the checkpoint
python train_recovery_92plus.py
python train_to_90_phase2.py

# Evaluate with test-time augmentation
python tta_ensemble.py

# Run the web app (requires models/best_cnn_gnn_model.pth)
python app.py                   # http://localhost:5000

Training defaults (in config.py): batch size 16, 50 epochs, learning rate 5e-4, Adam, ReduceLROnPlateau. The defaults are tuned for CPU. On a GPU, raise the batch size.
