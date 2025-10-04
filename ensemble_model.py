import torch
import torch.nn as nn
from model import HybridCNN_GNN
from config import Config
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class EnsembleModel:
    """Ensemble of multiple trained models for better accuracy"""
    
    def __init__(self, model_paths, device='cpu'):
        self.device = torch.device(device)
        self.models = []
        self.model_accuracies = []
        
        print("Loading ensemble models...")
        for i, path in enumerate(model_paths):
            try:
                # Load model
                model = HybridCNN_GNN(num_classes=Config.NUM_CLASSES).to(self.device)
                checkpoint = torch.load(path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                self.models.append(model)
                accuracy = checkpoint.get('accuracy', 0)
                self.model_accuracies.append(accuracy)
                
                print(f"  ✓ Model {i+1}: {path.split('/')[-1]} (Acc: {accuracy:.2f}%)")
            except Exception as e:
                print(f"  ✗ Failed to load {path}: {e}")
        
        print(f"\n✓ Loaded {len(self.models)} models successfully!")
        print(f"Average accuracy: {np.mean(self.model_accuracies):.2f}%\n")
    
    def predict(self, x, method='vote'):
        """
        Ensemble prediction methods:
        - 'vote': Majority voting (best for classification)
        - 'average': Average probabilities
        - 'weighted': Weighted by model accuracy
        """
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                probs = torch.nn.functional.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                predictions.append(pred)
                probabilities.append(probs)
        
        if method == 'vote':
            # Majority voting
            predictions_stack = torch.stack(predictions, dim=0)
            final_pred = torch.mode(predictions_stack, dim=0).values
            
        elif method == 'average':
            # Average probabilities
            avg_probs = torch.stack(probabilities, dim=0).mean(dim=0)
            final_pred = avg_probs.argmax(dim=1)
            
        elif method == 'weighted':
            # Weighted by accuracy
            weights = torch.tensor(self.model_accuracies, device=self.device)
            weights = weights / weights.sum()
            
            weighted_probs = sum(p * w for p, w in zip(probabilities, weights))
            final_pred = weighted_probs.argmax(dim=1)
        
        return final_pred
    
    def evaluate(self, test_loader, method='vote'):
        """Evaluate ensemble on test set"""
        all_preds = []
        all_labels = []
        
        print(f"Evaluating ensemble (method: {method})...")
        
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            predictions = self.predict(images, method=method)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        cm = confusion_matrix(all_labels, all_preds)
        
        return accuracy, cm, all_preds, all_labels


def test_ensemble():
    """Test ensemble model and save results"""
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test data
    test_dataset = ImageFolder(root=Config.TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    
    # Model paths (adjust based on what you have)
    model_paths = [
        'models/best_cnn_gnn_model.pth',  # Your best model (74.19%)
        # Add more if you saved them
    ]
    
    # If you only have one model, we'll use it multiple times with different methods
    if len(model_paths) == 1:
        print("⚠️ Only one model found. Using single model with different prediction methods.")
    
    # Create ensemble
    ensemble = EnsembleModel(model_paths)
    
    # Test different ensemble methods
    methods = ['vote', 'average', 'weighted']
    results = {}
    
    print("\n" + "="*70)
    print("ENSEMBLE EVALUATION")
    print("="*70 + "\n")
    
    for method in methods:
        print(f"\n📊 Method: {method.upper()}")
        print("-" * 70)
        
        accuracy, cm, preds, labels = ensemble.evaluate(test_loader, method=method)
        results[method] = {
            'accuracy': accuracy,
            'confusion_matrix': cm
        }
        
        print(f"✓ Ensemble Accuracy ({method}): {accuracy:.2f}%")
        
        # Save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=Config.CLASSES, yticklabels=Config.CLASSES)
        plt.title(f'Ensemble Confusion Matrix ({method}) - {accuracy:.2f}%')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'models/ensemble_confusion_matrix_{method}.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: models/ensemble_confusion_matrix_{method}.png")
    
    # Print summary
    print("\n" + "="*70)
    print("ENSEMBLE SUMMARY")
    print("="*70)
    print(f"\nBest method: {max(results, key=lambda k: results[k]['accuracy']).upper()}")
    print(f"Best accuracy: {max(r['accuracy'] for r in results.values()):.2f}%")
    print(f"\nComparison:")
    for method, result in results.items():
        print(f"  {method:10s}: {result['accuracy']:.2f}%")
    
    return results


if __name__ == '__main__':
    results = test_ensemble()
