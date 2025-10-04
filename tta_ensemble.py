import torch
import torch.nn as nn
from model import HybridCNN_GNN
from config import Config
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import os

class TTAEnsemble:
    """Test-Time Augmentation Ensemble"""
    
    def __init__(self, model_path, num_augmentations=5, device='cpu'):
        self.device = torch.device(device)
        self.num_aug = num_augmentations
        
        print(f"Loading model: {model_path}")
        self.model = HybridCNN_GNN(num_classes=Config.NUM_CLASSES).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.base_accuracy = checkpoint.get('accuracy', 0)
        print(f"✓ Model loaded! Base accuracy: {self.base_accuracy:.2f}%\n")
        
        # TTA transforms
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ColorJitter(brightness=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        ][:num_augmentations]
    
    def predict_tta(self, image_pil):
        predictions = []
        
        with torch.no_grad():
            for transform in self.tta_transforms:
                img_tensor = transform(image_pil).unsqueeze(0).to(self.device)
                output = self.model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                predictions.append(probs)
        
        avg_probs = torch.stack(predictions).mean(dim=0)
        final_pred = avg_probs.argmax(dim=1)
        
        return final_pred, avg_probs
    
    def evaluate(self, test_dir):
        all_preds = []
        all_labels = []
        
        class_names = sorted(os.listdir(test_dir))
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        print(f"Evaluating with {self.num_aug} augmentations per image...")
        
        image_count = 0
        error_count = 0
        
        for class_name in tqdm(class_names, desc="Classes"):
            class_dir = os.path.join(test_dir, class_name)
            
            # Skip if not a directory
            if not os.path.isdir(class_dir):
                continue
                
            class_idx = class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                
                # Skip if not a file or not an image
                if not os.path.isfile(img_path):
                    continue
                
                # Only process image files
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    continue
                
                try:
                    image = Image.open(img_path).convert('RGB')
                    pred, probs = self.predict_tta(image)
                    
                    all_preds.append(pred.item())
                    all_labels.append(class_idx)
                    image_count += 1
                    
                except Exception as e:
                    error_count += 1
                    # Silently skip errors
                    pass
        
        print(f"✓ Processed {image_count} images ({error_count} errors skipped)")
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        cm = confusion_matrix(all_labels, all_preds)
        
        return accuracy, cm, class_names


def run_tta_ensemble():
    model_path = 'models/best_cnn_gnn_model.pth'
    
    print("="*70)
    print("TEST-TIME AUGMENTATION (TTA) ENSEMBLE")
    print("="*70 + "\n")
    
    results = {}
    
    for num_aug in [3, 5, 7]:
        print(f"\n{'='*70}")
        print(f"Testing with {num_aug} augmentations")
        print(f"{'='*70}\n")
        
        tta = TTAEnsemble(model_path, num_augmentations=num_aug)
        accuracy, cm, class_names = tta.evaluate(Config.TEST_DIR)
        
        results[num_aug] = accuracy
        
        print(f"\n✓ TTA Accuracy ({num_aug} aug): {accuracy:.2f}%")
        print(f"  Base model: {tta.base_accuracy:.2f}%")
        print(f"  Improvement: +{accuracy - tta.base_accuracy:.2f}%")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'TTA Ensemble ({num_aug} aug) - {accuracy:.2f}%')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'models/tta_confusion_matrix_{num_aug}aug.png', dpi=300)
        print(f"  ✓ Saved confusion matrix")
    
    print("\n" + "="*70)
    print("TTA RESULTS SUMMARY")
    print("="*70)
    for aug, acc in results.items():
        print(f"  {aug} augmentations: {acc:.2f}%")
    print(f"\nBest TTA result: {max(results.values()):.2f}%")
    print("="*70)


if __name__ == '__main__':
    run_tta_ensemble()
