import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import time
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import your model classes (assuming they're in separate files)
from FERVT_GNN import FERVT_GNN, LabelSmoothingLoss
from transformer import Transformer

class DataPreprocessor:
    """Enhanced data preprocessing with augmentation strategies"""
    
    def __init__(self, input_size=(64, 64), augment=True):
        self.input_size = input_size
        self.augment = augment
        
    def get_transforms(self, mode='train'):
        """Get data transforms based on mode"""
        
        if mode == 'train' and self.augment:
            transform = transforms.Compose([
                transforms.Resize((72, 72)),  # Slightly larger for random crop
                transforms.RandomCrop(self.input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        return transform

class FER2013DataLoader:
    """Enhanced FER2013 dataset loader with proper splitting and validation"""
    
    def __init__(self, dataset_path, batch_size=32, input_size=(64, 64), 
                 train_split=0.7, val_split=0.15, test_split=0.15, augment=True):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.preprocessor = DataPreprocessor(input_size, augment)
        
        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Emotion labels for FER2013
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust', 
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
    def load_datasets(self):
        """Load and split datasets"""
        
        # Load full dataset
        full_dataset = ImageFolder(
            root=self.dataset_path,
            transform=self.preprocessor.get_transforms('train')
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Apply different transforms to validation and test sets
        val_dataset.dataset.transform = self.preprocessor.get_transforms('val')
        test_dataset.dataset.transform = self.preprocessor.get_transforms('test')
        
        return train_dataset, val_dataset, test_dataset
    
    def get_dataloaders(self, num_workers=4):
        """Get data loaders"""
        
        train_dataset, val_dataset, test_dataset = self.load_datasets()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

class ModelTrainer:
    """Comprehensive model trainer with advanced features"""
    
    def __init__(self, model, device, save_dir='./checkpoints'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
        
    def setup_training(self, learning_rate=1e-4, weight_decay=1e-4, 
                      label_smoothing=0.1, use_scheduler=True):
        """Setup optimizer, loss function, and scheduler"""
        
        # Optimizer with different learning rates for different parts
        backbone_params = list(self.model.backbone.parameters())
        other_params = [p for n, p in self.model.named_parameters() 
                       if not n.startswith('backbone')]
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained backbone
            {'params': other_params, 'lr': learning_rate}
        ], weight_decay=weight_decay)
        
        # Loss function with label smoothing
        self.criterion = LabelSmoothingLoss(classes=7, smoothing=label_smoothing)
        
        # Learning rate scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        else:
            self.scheduler = None
            
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, (data, targets) in enumerate(pbar):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_targets
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=15):
        """Main training loop"""
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, _, _ = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Record history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Check for best model
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Best Val Acc: {best_val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.2e}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return self.best_model_path

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, device, emotion_labels):
        self.model = model
        self.device = device
        self.emotion_labels = emotion_labels
        
    def evaluate(self, test_loader, model_path=None):
        """Comprehensive model evaluation"""
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing')
            for data, targets in pbar:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_predictions, average=None
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for i, emotion in self.emotion_labels.items():
            per_class_metrics[emotion] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        # Overall metrics
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class_metrics': per_class_metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
    
    def plot_confusion_matrix(self, targets, predictions, save_path=None):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.emotion_labels.values()),
                   yticklabels=list(self.emotion_labels.values()))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, trainer, save_path=None):
        """Plot training history"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(trainer.train_losses, label='Train Loss', color='blue')
        ax1.plot(trainer.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(trainer.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(trainer.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

def main():
    """Main training and evaluation pipeline"""
    
    # Configuration
    config = {
        'dataset_path': '/content/drive/MyDrive/FERVT/dataset/archive/CK+48/',
        'batch_size': 32,
        'input_size': (64, 64),
        'learning_rate': 1e-4,
        'epochs': 50,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'use_gnn': True,
        'save_dir': './checkpoints'
    }
    
    print(f"Using device: {config['device']}")
    
    # Data loading
    print("Loading datasets...")
    data_loader = FER2013DataLoader(
        dataset_path=config['dataset_path'],
        batch_size=config['batch_size'],
        input_size=config['input_size']
    )
    
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    
    print(f"Dataset loaded:")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Model initialization
    print("Initializing model...")
    model = FERVT_GNN(device=config['device'], use_gnn=config['use_gnn'])
    
    # Training
    trainer = ModelTrainer(model, config['device'], config['save_dir'])
    trainer.setup_training(learning_rate=config['learning_rate'])
    
    best_model_path = trainer.train(
        train_loader, val_loader, 
        epochs=config['epochs']
    )
    
    # Evaluation
    print("\nEvaluating model...")
    evaluator = ModelEvaluator(model, config['device'], data_loader.emotion_labels)
    
    results = evaluator.evaluate(test_loader, best_model_path)
    
    # Print results
    print(f"\nTest Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Macro Recall: {results['macro_recall']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    
    print(f"\nPer-class metrics:")
    for emotion, metrics in results['per_class_metrics'].items():
        print(f"{emotion:>10}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
              f"F1={metrics['f1']:.3f}, Support={metrics['support']}")
    
    # Visualizations
    evaluator.plot_confusion_matrix(
        results['targets'], results['predictions'],
        save_path=os.path.join(config['save_dir'], 'confusion_matrix.png')
    )
    
    evaluator.plot_training_history(
        trainer,
        save_path=os.path.join(config['save_dir'], 'training_history.png')
    )

if __name__ == "__main__":
    main()