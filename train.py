import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from model import create_model
def get_data_loaders():
    """
    Create training and testing data loaders with augmentation.
    """
    # Training augmentation
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), 
                               scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                              saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(Config.TRAIN_DIR, 
                                        transform=train_transform)
    test_dataset = datasets.ImageFolder(Config.TEST_DIR, 
                                       transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                             batch_size=Config.BATCH_SIZE,
                             shuffle=True, 
                             num_workers=4,
                             pin_memory=True)
    
    test_loader = DataLoader(test_dataset, 
                            batch_size=Config.BATCH_SIZE,
                            shuffle=False, 
                            num_workers=4,
                            pin_memory=True)
    
    return train_loader, test_loader, train_dataset.classes


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'  Batch [{batch_idx}/{len(loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Validate model on test set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Blood Group Detection')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model():
    """
    Main training function.
    """
    print(f"Using device: {Config.DEVICE}")
    print(f"Model: Hybrid CNN (EfficientNet-B3) + GNN (GAT)")
    
    # Get data loaders
    train_loader, test_loader, classes = get_data_loaders()
    print(f"\nClasses: {classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Initialize model
    model = create_model(num_classes=Config.NUM_CLASSES, 
                     model_size='b3',  # ADD THIS
                     device=Config.DEVICE)

    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        print(f'\n{"="*60}')
        print(f'Epoch [{epoch+1}/{Config.EPOCHS}]')
        print(f'{"="*60}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, test_loader, criterion, Config.DEVICE
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f'\nEpoch Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            os.makedirs('models', exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'classes': classes,
                'config': {
                    'num_classes': Config.NUM_CLASSES,
                    'img_size': Config.IMG_SIZE,
                    'cnn_feature_dim': Config.CNN_FEATURE_DIM,
                    'gnn_hidden_dim': Config.GNN_HIDDEN_DIM,
                    'gnn_output_dim': Config.GNN_OUTPUT_DIM
                }
            }, Config.MODEL_PATH)
            
            print(f'  ✓ Best model saved! (Acc: {val_acc:.2f}%)')
            
            # Save confusion matrix for best model
            plot_confusion_matrix(val_labels, val_preds, classes,
                                'models/confusion_matrix_best.png')
    
    print(f'\n{"="*60}')
    print(f'Training Complete!')
    print(f'Best Validation Accuracy: {best_accuracy:.2f}%')
    print(f'{"="*60}\n')
    
    # Final evaluation
    print('\nFinal Classification Report:')
    print(classification_report(val_labels, val_preds, 
                               target_names=classes, digits=4))
    
    # Plot training history
    plot_training_history(history)


def plot_training_history(history):
    """
    Plot training and validation metrics.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()


if __name__ == '__main__':
    train_model()
