import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from model import create_model
from train import get_data_loaders, train_epoch, validate, plot_confusion_matrix, plot_training_history

def recovery_training_92plus():
    """
    Optimized recovery training targeting 92%+ accuracy
    Balanced for speed and performance
    """
    
    print("="*70)
    print("RECOVERY MODE: Targeting 92%+ Accuracy")
    print("="*70)
    
    # Load the BEST model (Epoch 5, 49.69%)
    checkpoint = torch.load('models/best_cnn_gnn_model.pth')
    
    model = create_model(
        num_classes=Config.NUM_CLASSES,
        model_size='b3',
        device=Config.DEVICE
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n✓ Loaded best model from Epoch {checkpoint['epoch']} (Acc: {checkpoint['accuracy']:.2f}%)")
    print("✓ Optimized training for 92%+ accuracy with reduced time\n")
    
    # Get data loaders
    train_loader, test_loader, classes = get_data_loaders()
    
    # OPTIMIZED LEARNING RATE (lower for stability)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Was 0.0005
    
    # Aggressive scheduler with early stopping awareness
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True,
        min_lr=1e-6
    )
    
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_accuracy = checkpoint['accuracy']
    
    # OPTIMIZED: 35 epochs (enough for 92%+, saves time)
    TARGET_EPOCHS = 35
    
    print(f"Training Configuration:")
    print(f"  Initial LR: 0.0001 (5x lower, more stable)")
    print(f"  Starting from: {best_accuracy:.2f}%")
    print(f"  Target accuracy: 92%+")
    print(f"  Epochs: {TARGET_EPOCHS} (optimized)")
    print(f"  Estimated time: ~30 hours")
    print(f"  Early stopping: If 92% reached")
    print("="*70 + "\n")
    
    # Early stopping parameters
    no_improvement_count = 0
    early_stop_patience = 8  # Stop if no improvement for 8 epochs AND >92%
    
    # Training loop
    for epoch in range(TARGET_EPOCHS):
        print(f'\n{"="*70}')
        print(f'Epoch [{epoch+1}/{TARGET_EPOCHS}]')
        print(f'{"="*70}')
        
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
        
        # Print summary
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%')
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr < old_lr:
            print(f'  ✓ Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}')
        
        # Save if improved
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            no_improvement_count = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'classes': classes
            }, 'models/best_cnn_gnn_model.pth')
            
            print(f'  🎯 Best model saved! (Acc: {val_acc:.2f}%)')
            plot_confusion_matrix(val_labels, val_preds, classes,
                                'models/confusion_matrix_best.png')
            
            # Check if target reached
            if val_acc >= 92.0:
                print(f'\n{"="*70}')
                print(f'🎉 TARGET ACHIEVED! Accuracy: {val_acc:.2f}% ≥ 92%')
                print(f'{"="*70}')
        else:
            no_improvement_count += 1
            gap = best_accuracy - val_acc
            print(f'  Current best: {best_accuracy:.2f}% (Gap: {gap:.2f}%)')
        
        # Early stopping if we reached 92% and no improvement for a while
        if best_accuracy >= 92.0 and no_improvement_count >= early_stop_patience:
            print(f'\n{"="*70}')
            print(f'EARLY STOPPING: Target 92%+ achieved!')
            print(f'No improvement for {no_improvement_count} epochs.')
            print(f'Final accuracy: {best_accuracy:.2f}%')
            print(f'{"="*70}')
            break
        
        # Progress tracker
        if epoch % 5 == 0 and epoch > 0:
            print(f'\n  📊 Progress Report:')
            print(f'     Epochs completed: {epoch+1}/{TARGET_EPOCHS}')
            print(f'     Best so far: {best_accuracy:.2f}%')
            print(f'     Target: 92%+')
            remaining = TARGET_EPOCHS - epoch - 1
            print(f'     Remaining: {remaining} epochs (~{remaining * 0.85:.1f} hours)')
    
    print(f'\n{"="*70}')
    print(f'TRAINING COMPLETE!')
    print(f'{"="*70}')
    print(f'Final Best Accuracy: {best_accuracy:.2f}%')
    
    if best_accuracy >= 92.0:
        print(f'✅ TARGET ACHIEVED! ({best_accuracy:.2f}% ≥ 92%)')
        print(f'   Ready for publication! 📄')
    elif best_accuracy >= 90.0:
        print(f'🟡 CLOSE! ({best_accuracy:.2f}%)')
        print(f'   Consider fine-tuning or TTA for 92%+')
    else:
        print(f'🔴 TARGET NOT REACHED ({best_accuracy:.2f}%)')
        print(f'   Recommend: Train with EfficientNet-B4 or ensemble')
    
    print(f'{"="*70}\n')
    
    plot_training_history(history, 'models/recovery_training_history.png')
    
    return best_accuracy

if __name__ == '__main__':
    final_acc = recovery_training_92plus()
