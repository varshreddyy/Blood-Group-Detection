import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model import HybridCNN_GNN
from config import Config
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

print("="*70)
print("PHASE 2 TRAINING: PUSH TO 90%+")
print("="*70 + "\n")

# Enhanced augmentation for Phase 2
train_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), 
                           scale=(0.8, 1.2), shear=15),
    transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.2))
])

test_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create data loaders
print("Loading datasets with enhanced augmentation...")
train_dataset = ImageFolder(Config.TRAIN_DIR, transform=train_transform)
test_dataset = ImageFolder(Config.TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                          shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                         shuffle=False, num_workers=0, pin_memory=False)

print(f"✓ Train samples: {len(train_dataset)}")
print(f"✓ Test samples: {len(test_dataset)}\n")

# Load model
device = torch.device('cpu')
model = HybridCNN_GNN(num_classes=Config.NUM_CLASSES).to(device)

checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

start_epoch = checkpoint['epoch']
best_accuracy = checkpoint['accuracy']

print(f"✓ Loaded model from Epoch {start_epoch}")
print(f"✓ Starting accuracy: {best_accuracy:.2f}%\n")

# Optimizer with very low LR for fine-tuning
optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.02)

# Aggressive scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.25, patience=2,
    verbose=True, min_lr=1e-9
)

criterion = nn.CrossEntropyLoss()

# Training parameters
PHASE2_EPOCHS = 45
total_epochs = start_epoch + PHASE2_EPOCHS

print(f"{'='*70}")
print(f"Phase 2 Configuration:")
print(f"  Starting from: Epoch {start_epoch} ({best_accuracy:.2f}%)")
print(f"  Additional epochs: {PHASE2_EPOCHS}")
print(f"  Target total: Epoch {total_epochs}")
print(f"  Initial LR: 0.00001")
print(f"  Scheduler: ReduceLROnPlateau (factor=0.25, patience=2)")
print(f"  Target accuracy: 90%+")
print(f"  Estimated time: ~30 hours")
print(f"{'='*70}\n")

# Training function
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    
    for batch_idx, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(test_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_preds, all_labels

# Training loop
print("Starting Phase 2 training...\n")

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

for epoch in range(start_epoch + 1, total_epochs + 1):
    print(f"\n{'='*70}")
    print(f"Epoch [{epoch}/{total_epochs}]")
    print(f"{'='*70}")
    
    # Train
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, 
                                            optimizer, device, epoch)
    
    # Validate
    val_loss, val_acc, preds, labels = validate(model, test_loader, criterion, device)
    
    # Update scheduler
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Print summary
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
    print(f"  Learning Rate: {current_lr:.9f}")
    
    # Save best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': best_accuracy,
            'classes': train_dataset.classes
        }, Config.MODEL_PATH)
        
        # Save confusion matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=Config.CLASSES, yticklabels=Config.CLASSES)
        plt.title(f'Confusion Matrix - Epoch {epoch} ({best_accuracy:.2f}%)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix_best.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  🎯 Best model saved! (Acc: {best_accuracy:.2f}%)")
    else:
        print(f"  Current best: {best_accuracy:.2f}% (Gap: {best_accuracy - val_acc:.2f}%)")
    
    # Progress report every 5 epochs
    if epoch % 5 == 0:
        print(f"\n  📊 Progress Report:")
        print(f"     Epochs completed: {epoch}/{total_epochs}")
        print(f"     Best so far: {best_accuracy:.2f}%")
        print(f"     Target: 90%+")
        print(f"     Remaining: {total_epochs - epoch} epochs")
    
    # Check if target reached
    if best_accuracy >= 90.0:
        print(f"\n🎉 TARGET REACHED! Accuracy: {best_accuracy:.2f}%")
        break

print("\n" + "="*70)
print("PHASE 2 TRAINING COMPLETE!")
print("="*70)
print(f"Final Best Accuracy: {best_accuracy:.2f}%")
print(f"Total Epochs: {epoch}")
print(f"Model saved: {Config.MODEL_PATH}")
print("="*70)

# Save training history plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History - Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training History - Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('models/phase2_training_history.png', dpi=300, bbox_inches='tight')
print(f"✓ Training history saved: models/phase2_training_history.png")
