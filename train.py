import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


class SkinLesionDataset(Dataset):
    """Custom Dataset for skin lesion images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except:
            img = Image.new('RGB', (224, 224))
            if self.transform:
                img = self.transform(img)
            return img, label


class SkinDiseaseTrainer:
    def __init__(self, data_dir='appt_data', model_dir='models', use_preprocessed=True, 
                 max_images_per_class=None, batch_size=16):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.use_preprocessed = use_preprocessed  # Use train_pre/test_pre folders
        self.max_images_per_class = max_images_per_class  # Limit images per class (None = unlimited)
        self.img_size = (224, 224)
        self.batch_size = batch_size  # Configurable batch size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique timestamp for each run
        print(f"Using device: {self.device}")
        print(f"Run ID: {self.run_id}")
        if self.max_images_per_class:
            print(f"Max images per class: {self.max_images_per_class}")
        if self.batch_size != 16:
            print(f"Batch size: {self.batch_size}")
        # Initialize mixed precision if available
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            from torch.cuda.amp import autocast
            self.autocast = autocast
        else:
            self.autocast = None
        
    def download_isic_data(self):
        """Prepare data directories"""
        print("=" * 60)
        print("PREPARING SKIN DISEASE DATA")
        print("=" * 60)
        
        if self.use_preprocessed:
            print("\n⚙️  Using preprocessed APPT dataset")
            train_dir = self.data_dir / 'train_pre'
            test_dir = self.data_dir / 'test_pre'
            
            if not train_dir.exists() or not test_dir.exists():
                print(f"\n❌ Preprocessed data not found!")
                print(f"   Expected: {train_dir} and {test_dir}")
                print(f"\n   Run preprocessing first:")
                print(f"   python appt_preprocess.py")
                return False
            print(f"✓ Found preprocessed data")
            print(f"  Train: {train_dir}")
            print(f"  Test: {test_dir}\n")
        else:
            if not self.data_dir.exists():
                self.data_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n📁 Created data directory: {self.data_dir}")
                
            print("\n📥 Note: To use ISIC data, download from:")
            print("   https://www.isic-archive.com/")
            print("   Place images in subdirectories by class:")
            print("   isic_data/melanoma/")
            print("   isic_data/nevus/")
            print("   isic_data/carcinoma/")
            print("   isic_data/keratosis/")
            print("   isic_data/vascular/")
            print("   isic_data/dermatofibroma/")
            print("\n   Proceeding with ready data in isic_data/ folder...\n")
        
        return True
        
    def load_and_preprocess(self):
        """Load images from organized directories"""
        print("\n📊 LOADING DATASET")
        print("-" * 60)
        
        image_paths = []
        labels = []
        class_names = []
        split_info = []  # Track which split each image belongs to
        
        # Get data directories based on mode
        if self.use_preprocessed:
            data_sources = [
                (self.data_dir / 'train_pre', 'train'),
                (self.data_dir / 'test_pre', 'test')
            ]
        else:
            data_sources = [(self.data_dir, None)]
        
        # Get all class directories from first source
        first_source_dir = data_sources[0][0]
        if not first_source_dir.exists():
            return None, None, None, None
        
        class_dirs = sorted([d for d in first_source_dir.iterdir() if d.is_dir()])
        
        if not class_dirs:
            data_name = 'appt_data/train_pre' if self.use_preprocessed else 'isic_data'
            print(f"❌ No class directories found in {data_name}/")
            print("   Please organize images in class subdirectories.")
            return None, None, None, None
        
        class_names = [d.name for d in class_dirs]
        print(f"Found {len(class_names)} classes: {', '.join(class_names)}")
        
        # Load image paths from all sources (train_pre and test_pre)
        for source_dir, source_name in data_sources:
            if not source_dir.exists():
                if self.use_preprocessed:
                    print(f"  ⚠ {source_name} directory not found: {source_dir}")
                continue
            
            print(f"\n  Loading from {source_name}:" if source_name else "")
            
            for class_idx, class_name in enumerate(class_names):
                class_dir = source_dir / class_name
                if not class_dir.exists():
                    continue
                
                # Find images with case-insensitive extension matching
                img_files = [f for f in class_dir.iterdir() 
                            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
                
                # Limit images per class if specified
                if self.max_images_per_class:
                    img_files = img_files[:self.max_images_per_class]
                
                if source_name:
                    print(f"    • {class_name}: {len(img_files)} images ({source_name})")
                else:
                    print(f"  • {class_name}: {len(img_files)} images")
                
                for img_path in img_files:
                    try:
                        # Test if image can be opened
                        img = Image.open(img_path).convert('RGB')
                        image_paths.append(str(img_path))
                        labels.append(class_idx)
                        split_info.append(source_name)  # Track split
                    except:
                        continue
        
        if not image_paths:
            print("❌ No images found!")
            return None, None, None, None
        
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        split_info = np.array(split_info) if split_info else None
        
        print(f"\n✓ Loaded {len(image_paths)} images across {len(class_names)} classes")
        
        return image_paths, labels, class_names, len(class_names), split_info
    
    def build_model(self, num_classes):
        """Build ResNet50 transfer learning model"""
        print("\n🏗️  BUILDING MODEL")
        print("-" * 60)
        
        # Load pre-trained ResNet50
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze base layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace final layer with better architecture
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        model = model.to(self.device)
        print(f"✓ Model built with ResNet50 backbone")
        print(f"  Output: {num_classes} classes")
        
        return model
    
    def train(self, epochs=40):
        """Train the model with progress tracking"""
        print("\n" + "=" * 60)
        print("TRAINING SKIN DISEASE DETECTOR")
        print("=" * 60)
        
        # Check if data exists - different validation based on mode
        if self.use_preprocessed:
            # Check for preprocessed data in train_pre/test_pre
            train_pre_dir = self.data_dir / 'train_pre'
            test_pre_dir = self.data_dir / 'test_pre'
            has_data = (train_pre_dir.exists() and len(list(train_pre_dir.glob('*/*.jpg'))) + len(list(train_pre_dir.glob('*/*.png'))) > 0) or \
                       (test_pre_dir.exists() and len(list(test_pre_dir.glob('*/*.jpg'))) + len(list(test_pre_dir.glob('*/*.png'))) > 0)
            
            if not has_data:
                print("\n❌ Preprocessed APPT data not found!")
                print("\nTo train the model:")
                print("1. Run preprocessing first: python appt_preprocess.py")
                print("2. Make sure raw images are in appt_data/train/ and appt_data/test/")
                print("3. Run this script again")
                return None
        else:
            # Check for ISIC data in root directories
            if not self.data_dir.exists() or not list(self.data_dir.glob('*/*.jpg')) and not list(self.data_dir.glob('*/*.png')):
                print("\n❌ ISIC data not found!")
                print("\nTo train the model:")
                print("1. Download ISIC images from https://www.isic-archive.com/")
                print("2. Organize into folders by disease type in: isic_data/")
                print("3. Run this script again")
                return None
        
        # Load data
        result = self.load_and_preprocess()
        
        if self.use_preprocessed:
            image_paths, labels, class_names, num_classes, split_info = result
        else:
            image_paths, labels, class_names, num_classes = result + (None,)
            split_info = None
        
        if image_paths is None:
            return None
        
        # Split data
        print("\n📋 SPLITTING DATA")
        print("-" * 60)
        
        if self.use_preprocessed and split_info is not None:
            # Use predefined train/test split from preprocessed data
            X_train = image_paths[split_info == 'train']
            y_train = labels[split_info == 'train']
            X_test = image_paths[split_info == 'test']
            y_test = labels[split_info == 'test']
            
            # Further split train into train/val
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        else:
            # Original splitting logic for ISIC data
            X_train, X_temp, y_train, y_temp = train_test_split(
                image_paths, labels, test_size=0.3, random_state=42, stratify=labels)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        
        # Data transforms - balanced augmentation for 85%+ accuracy
        train_transform = transforms.Compose([
            transforms.RandomRotation(30),  # Reasonable rotation
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = SkinLesionDataset(X_train, y_train, train_transform)
        val_dataset = SkinLesionDataset(X_val, y_val, val_transform)
        test_dataset = SkinLesionDataset(X_test, y_test, val_transform)
        
        # Create weighted sampler for training - balances classes during training
        # This allows us to keep all images while ensuring fair class representation
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / (class_counts + 1e-8)  # Inverse frequency weighting
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights.astype(np.float64),
            num_samples=len(y_train),
            replacement=True
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                 sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                               shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                shuffle=False, num_workers=0)
        
        # Build model
        model = self.build_model(num_classes)
        
        # Compute class weights for balanced loss
        class_counts = np.bincount(y_train)
        class_weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(self.device)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        
        # Loss with class weights and label smoothing
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        # SGD optimizer (better for CNNs than Adam)
        optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.95, weight_decay=1e-4, nesterov=True)
        
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
        
        # Training history
        history = []
        best_val_acc = 0
        
        # Train model
        print("\n🚀 TRAINING MODEL (85%+ ACCURACY OPTIMIZED)")
        print("-" * 60)
        print(f"Epochs: {epochs} | Batch Size: {self.batch_size}")
        print(f"Classes: {num_classes} | Train Samples: {len(X_train)}")
        print(f"\n✨ Optimization Strategy:")
        print(f"  • SGD optimizer (momentum=0.95, weight decay=1e-4)")
        print(f"  • Cosine annealing with warm restarts")
        print(f"  • Class-weighted loss + label smoothing (0.1)")
        print(f"  • 2-stage unfreezing (epochs 10, 20)")
        print(f"  • Balanced data augmentation")
        print(f"  • Batch normalization in head layers\n")
        
        for epoch in range(epochs):
            # Simpler unfreezing schedule for 85% accuracy
            if epoch == 10:
                print("\n🔄 STAGE 1: Unfreezing layer3 and layer4")
                for param in model.layer3.parameters():
                    param.requires_grad = True
                for param in model.layer4.parameters():
                    param.requires_grad = True
                print("✓ Layers 3-4 unlocked\n")
            elif epoch == 20:
                print("\n🔄 STAGE 2: Full fine-tuning")
                for param in model.layer2.parameters():
                    param.requires_grad = True
                print("✓ Full fine-tuning active\n")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")
            for images, labels_batch in train_pbar:
                images = images.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()
                
                train_acc = train_correct / train_total
                train_pbar.set_postfix({'loss': f'{train_loss/len(train_loader):.4f}', 
                                       'acc': f'{train_acc*100:.2f}%'})
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]")
                for images, labels_batch in val_pbar:
                    images = images.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels_batch)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()
                    
                    val_acc_current = val_correct / val_total
                    val_pbar.set_postfix({'loss': f'{val_loss/len(val_loader):.4f}', 
                                         'acc': f'{val_acc_current*100:.2f}%'})
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # Save history
            history.append({
                'epoch': epoch + 1,
                'loss': float(train_loss),
                'accuracy': float(train_acc),
                'val_loss': float(val_loss),
                'val_accuracy': float(val_acc)
            })
            
            # Save models with unique run ID and validation accuracy
            val_acc_percent = int(val_acc * 100)
            torch.save(model.state_dict(), str(self.model_dir / f'run{self.run_id}_epoch{epoch+1}_{val_acc_percent}pct.pth'))
            
            # Save best model with unique run ID
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_acc_percent = int(val_acc * 100)
                torch.save(model.state_dict(), str(self.model_dir / f'run{self.run_id}_BEST_{best_val_acc_percent}pct.pth'))
            
            # Save training history
            with open('training_history.json', 'w') as f:
                json.dump(history, f, indent=2)
            
            # Update learning rate (step at every epoch, not based on val_loss)
            scheduler.step()
            
            print()
        
        # Evaluate on test set
        print("\n📊 FINAL EVALUATION")
        print("-" * 60)
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels_batch in test_loader:
                images = images.to(self.device)
                labels_batch = labels_batch.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels_batch.size(0)
                test_correct += (predicted == labels_batch).sum().item()
        
        test_accuracy = test_correct / test_total
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Save final model with unique run ID
        final_acc_percent = int(best_val_acc * 100)
        torch.save(model.state_dict(), str(self.model_dir / f'run{self.run_id}_FINAL_{final_acc_percent}pct.pth'))
        print(f"\n✓ Model saved to: {self.model_dir}/run{self.run_id}_FINAL_{final_acc_percent}pct.pth")
        
        # Save class mapping
        class_mapping = {i: name for i, name in enumerate(class_names)}
        with open(self.model_dir / 'class_mapping.json', 'w') as f:
            json.dump(class_mapping, f)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        
        return model, history, class_names


if __name__ == "__main__":
    # ============ CONFIGURATION OPTIONS ============
    
    # Fast training (15-30 min): Use 300 images per class, larger batch size, fewer epochs
    trainer = SkinDiseaseTrainer(
        data_dir='appt_data',
        model_dir='models',
        use_preprocessed=True,
        max_images_per_class=300,  # Limit to 300 per class for speed
        batch_size=32               # Use larger batch size
    )
    
    # For balanced accuracy/speed (45-60 min): Use 500 images per class
    # trainer = SkinDiseaseTrainer(
    #     data_dir='appt_data',
    #     model_dir='models',
    #     use_preprocessed=True,
    #     max_images_per_class=500,  # 500 per class for balanced training
    #     batch_size=24
    # )
    
    # Full training (100-150 min): Use all images, smaller batch size
    # trainer = SkinDiseaseTrainer(
    #     data_dir='appt_data',
    #     model_dir='models',
    #     use_preprocessed=True,
    #     max_images_per_class=None,  # Use all images
    #     batch_size=16
    # )
    
    # To train with raw ISIC data instead:
    # trainer = SkinDiseaseTrainer(data_dir='isic_data', model_dir='models', use_preprocessed=False)
    
    # Prepare data
    if trainer.download_isic_data():
        # Train model with fewer epochs for speed (20 epochs = ~40-50 min with 300 imgs/class)
        trainer.train(epochs=20)
