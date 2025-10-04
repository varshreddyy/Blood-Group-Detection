import torch

class Config:
    # Dataset paths
    TRAIN_DIR = 'dataset/train'
    TEST_DIR = 'dataset/test'
    MODEL_PATH = 'models/best_cnn_gnn_model.pth'
    
    # Training parameters
    BATCH_SIZE = 16  # Reduced for CPU (use 32 if you have GPU)
    EPOCHS = 50
    LEARNING_RATE = 0.0005
    IMG_SIZE = 224
    NUM_CLASSES = 8
    
    # CNN parameters (EfficientNet)
    EFFICIENTNET_VERSION = 'efficientnet-b3'
    CNN_FEATURE_DIM = 1536  # EfficientNet-B3 output dimension
    
    # GNN parameters
    GNN_HIDDEN_DIM = 512
    GNN_OUTPUT_DIM = 256
    GNN_NUM_LAYERS = 3
    GRAPH_K_NEIGHBORS = 8
    
    # Graph construction
    FEATURE_MAP_SIZE = 7  # 7x7 feature map from EfficientNet
    NUM_GRAPH_NODES = 49  # 7x7 = 49 nodes
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Blood groups
    BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    
    # Flask config
    UPLOAD_FOLDER = 'static/uploads'
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
