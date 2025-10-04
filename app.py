from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os
from model import HybridCNN_GNN
from config import Config

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load model
device = torch.device('cpu')
checkpoint_path = 'models/best_cnn_gnn_model.pth'

print("Loading model...")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Create model instance
model = HybridCNN_GNN(num_classes=Config.NUM_CLASSES).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded! Base accuracy: {checkpoint['accuracy']:.2f}%")

# TTA Transforms (7 augmentations for 79.31% accuracy)
TTA_TRANSFORMS = [
    # Original
    transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Rotate right
    transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomRotation((10, 10)),  # Fixed: tuple format
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Brightness
    transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Vertical flip
    transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Rotate left
    transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomRotation((-10, -10)),  # Fixed: tuple format
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Contrast
    transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ColorJitter(contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
]

# Blood group labels
BLOOD_GROUPS = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Model statistics (with TTA)
MODEL_STATS = {
    'accuracy': 79.31,  # TTA accuracy
    'base_accuracy': checkpoint['accuracy'],
    'epoch': checkpoint['epoch'],
    'total_params': sum(p.numel() for p in model.parameters()),
    'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    'tta_enabled': True,
    'tta_augmentations': len(TTA_TRANSFORMS)
}

@app.route('/')
def index():
    return render_template('index.html', stats=MODEL_STATS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        if 'file' not in request.files and 'image' not in request.form:
            return jsonify({'error': 'No image provided'}), 400
        
        # Handle file upload or webcam capture
        if 'file' in request.files:
            file = request.files['file']
            image = Image.open(file.stream).convert('RGB')
        else:
            # Webcam image (base64)
            image_data = request.form['image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # TTA Prediction (7 augmentations)
        predictions = []
        with torch.no_grad():
            for transform in TTA_TRANSFORMS:
                img_tensor = transform(image).unsqueeze(0).to(device)
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                predictions.append(probs)
        
        # Average probabilities across all augmentations
        avg_probs = torch.stack(predictions).mean(dim=0)
        confidence, predicted = torch.max(avg_probs, 1)
        
        # Get results
        blood_group = BLOOD_GROUPS[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Get all probabilities
        all_probs = {
            BLOOD_GROUPS[i]: float(avg_probs[0][i]) * 100 
            for i in range(len(BLOOD_GROUPS))
        }
        
        return jsonify({
            'success': True,
            'blood_group': blood_group,
            'confidence': round(confidence_score, 2),
            'all_probabilities': all_probs,
            'model_accuracy': 79.31,
            'tta_enabled': True,
            'tta_augmentations': len(TTA_TRANSFORMS)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    cm_path = 'static/images/confusion_matrix_best.png'
    has_cm = os.path.exists(cm_path)
    
    # Check for TTA confusion matrices
    tta_cm_paths = {
        '7aug': 'models/tta_confusion_matrix_7aug.png',
        '5aug': 'models/tta_confusion_matrix_5aug.png',
        '3aug': 'models/tta_confusion_matrix_3aug.png'
    }
    
    tta_cms = {k: os.path.exists(v) for k, v in tta_cm_paths.items()}
    
    return jsonify({
        'accuracy': MODEL_STATS['accuracy'],
        'base_accuracy': MODEL_STATS['base_accuracy'],
        'epoch': MODEL_STATS['epoch'],
        'total_params': MODEL_STATS['total_params'],
        'trainable_params': MODEL_STATS['trainable_params'],
        'tta_enabled': MODEL_STATS['tta_enabled'],
        'tta_augmentations': MODEL_STATS['tta_augmentations'],
        'has_confusion_matrix': has_cm,
        'tta_confusion_matrices': tta_cms
    })

if __name__ == '__main__':
    # Copy confusion matrices to static folder
    if os.path.exists('models/confusion_matrix_best.png'):
        os.makedirs('static/images', exist_ok=True)
        import shutil
        shutil.copy('models/confusion_matrix_best.png', 'static/images/')
    
    # Copy TTA confusion matrices
    for aug in ['3aug', '5aug', '7aug']:
        src = f'models/tta_confusion_matrix_{aug}.png'
        if os.path.exists(src):
            os.makedirs('static/images', exist_ok=True)
            import shutil
            shutil.copy(src, f'static/images/tta_confusion_matrix_{aug}.png')
    
    print("\n" + "="*70)
    print("🩸 Blood Group Detection System (TTA Enhanced)")
    print("="*70)
    print(f"✓ Base Model Accuracy: {MODEL_STATS['base_accuracy']:.2f}%")
    print(f"✓ TTA Enhanced Accuracy: {MODEL_STATS['accuracy']:.2f}%")
    print(f"✓ Improvement: +{MODEL_STATS['accuracy'] - MODEL_STATS['base_accuracy']:.2f}%")
    print(f"✓ Epochs Trained: {MODEL_STATS['epoch']}")
    print(f"✓ Total Parameters: {MODEL_STATS['total_params']:,}")
    print(f"✓ TTA Augmentations: {MODEL_STATS['tta_augmentations']}")
    print("="*70)
    print("\n🌐 Open browser: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000)
