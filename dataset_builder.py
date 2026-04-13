#!/usr/bin/env python3
"""
Automated Kaggle Dataset Builder for Skin Cancer Detection
Downloads HAM10000 and ISIC datasets, combines them, and creates balanced classes
"""

import os
import sys
import json
import shutil
import zipfile
from pathlib import Path
import subprocess
from tqdm import tqdm

class KaggleDatassetBuilder:
    def __init__(self, api_key=None, username=None):
        self.home = Path.home()
        self.kaggle_dir = self.home / '.kaggle'
        self.kaggle_json = Path('kaggle.json')
        self.output_dir = Path('isic_data')
        self.temp_dir = Path('temp_datasets')
        self.api_key = api_key
        self.username = username or 'kaggle_user'
        
        # Dataset info
        self.datasets = {
            'isic9': 'nodoubttome/skin-cancer9-classesisic',
            'skin_melanoma': 'hasananwar/melanoma-skin-cancer-dataset-of-10000-images',
            'dermatology': 'shubhamgoel27/dermnet'
        }
    
    def setup_kaggle_api(self):
        """Setup Kaggle API credentials"""
        print("=" * 60)
        print("🔧 SETTING UP KAGGLE API")
        print("=" * 60)
        
        # Try multiple credential sources
        api_key = None
        username = None
        
        # Source 1: Environment variable
        if os.environ.get('KAGGLE_API_TOKEN'):
            api_key = os.environ.get('KAGGLE_API_TOKEN')
            username = os.environ.get('KAGGLE_USERNAME', 'kaggle_user')
            print(f"\n✓ Found API key in KAGGLE_API_TOKEN environment variable")
        
        # Source 2: Provided to constructor
        elif self.api_key:
            api_key = self.api_key
            username = self.username
            print(f"\n✓ Using provided API key")
        
        # Source 3: kaggle.json in current directory
        elif self.kaggle_json.exists():
            print(f"\n✓ Found kaggle.json in current directory")
            try:
                with open(self.kaggle_json, 'r') as f:
                    config = json.load(f)
                    api_key = config.get('key')
                    username = config.get('username')
            except Exception as e:
                print(f"❌ Failed to read kaggle.json: {e}")
                return False
        
        # If we have API key, create kaggle.json
        if api_key and username:
            self.kaggle_dir.mkdir(exist_ok=True)
            print(f"✓ .kaggle directory: {self.kaggle_dir}")
            
            dest_path = self.kaggle_dir / 'kaggle.json'
            
            try:
                # Create kaggle.json config
                config = {
                    "username": username,
                    "key": api_key
                }
                
                with open(dest_path, 'w') as f:
                    json.dump(config, f)
                
                print(f"✓ Created kaggle.json with API credentials")
            except Exception as e:
                print(f"❌ Failed to create kaggle.json: {e}")
                return False
            
            # Set permissions
            try:
                os.chmod(dest_path, 0o600)
                print(f"✓ Set permissions to 600")
            except Exception as e:
                print(f"⚠️  Could not set permissions (Windows may ignore this): {e}")
            
            # Verify Kaggle SDK is available
            try:
                import kaggle
                print(f"✓ Kaggle SDK configured successfully")
                return True
            except ImportError:
                print(f"❌ Kaggle SDK not installed")
                print(f"   Install with: pip install kaggle")
                return False
            except Exception as e:
                print(f"❌ Error verifying Kaggle SDK: {e}")
                return False
        
        else:
            print("\n❌ No API credentials found!")
            print("\nProvide credentials using ONE of these methods:")
            print("\n1️⃣  Environment variable:")
            print("   export KAGGLE_API_TOKEN=YOUR_API_KEY_HERE")
            print("   export KAGGLE_USERNAME=your_kaggle_username")
            print("\n2️⃣  Command line argument:")
            print("   python dataset_builder.py --api-key YOUR_KEY --username your_username")
            print("\n3️⃣  Place kaggle.json in current directory:")
            print("   Download from: https://www.kaggle.com/settings/account")
            print("\n4️⃣  Hardcoded in ~/.kaggle/kaggle.json:")
            print("   See: https://github.com/Kaggle/kaggle-api#api-credentials")
            return False
    
    def check_dataset_exists(self, dataset_name):
        """Check if dataset is already downloaded and extracted"""
        if not self.temp_dir.exists():
            return False
        
        dataset_name = dataset_name.lower()
        
        # Look for directories or files related to the dataset
        for p in self.temp_dir.glob('*'):
            if not p.is_dir():
                continue
            dir_name = p.name.lower()
            
            if 'isic' in dataset_name and ('isic' in dir_name or 'skin' in dir_name):
                return True
            elif 'melanoma' in dataset_name and ('melanoma' in dir_name or 'melano' in dir_name):
                return True
            elif 'dermatology' in dataset_name or 'dermnet' in dataset_name:
                if 'derm' in dir_name or 'dermatology' in dir_name:
                    return True
        
        return False
    
    def download_dataset(self, dataset_name, dataset_path):
        """Download dataset from Kaggle"""
        print(f"\n📥 Downloading {dataset_name.upper()}")
        print("-" * 60)
        
        # Create temp directory
        self.temp_dir.mkdir(exist_ok=True)
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Initialize Kaggle API from credentials file
            api = KaggleApi()
            api.authenticate()
            
            print(f"Downloading: {dataset_path}")
            api.dataset_download_files(dataset_path, path=str(self.temp_dir), unzip=True)
            
            print(f"✓ Downloaded and extracted {dataset_name}")
            return True
        
        except ImportError:
            print(f"❌ Kaggle API not available")
            print(f"   Install with: pip install kaggle")
            return False
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            return False
    
    
    def organize_isic9(self):
        """Organize ISIC 9-class dataset by class"""
        print("\n📂 ORGANIZING ISIC 9-CLASS")
        print("-" * 60)
        
        # Find ISIC dataset directory
        isic_path = None
        for p in self.temp_dir.glob('*'):
            if p.is_dir() and ('isic' in p.name.lower() or 'skin' in p.name.lower()):
                isic_path = p
                break
        
        if not isic_path:
            print("❌ ISIC dataset directory not found")
            return 0
        
        print(f"Found ISIC dataset at: {isic_path.name}")
        
        # Map ISIC class names to standardized names
        isic_mapping = {
            'melanoma': 'melanoma',
            'mel': 'melanoma',
            'nevus': 'nevus',
            'nv': 'nevus',
            'basal': 'basal_cell_carcinoma',
            'bcc': 'basal_cell_carcinoma',
            'basalcell': 'basal_cell_carcinoma',
            'keratosis': 'actinic_keratosis',
            'ak': 'actinic_keratosis',
            'actinic': 'actinic_keratosis',
            'benign': 'benign_keratosis',
            'bkl': 'benign_keratosis',
            'dermatofibroma': 'dermatofibroma',
            'df': 'dermatofibroma',
            'vascular': 'vascular_lesion',
            'vasc': 'vascular_lesion',
            'seborrheic': 'seborrheic_keratosis',
            'sebk': 'seborrheic_keratosis',
            'squamous': 'squamous_cell_carcinoma',
            'scc': 'squamous_cell_carcinoma',
            'lentigo': 'lentigo',
            'solar': 'solar_lentigo',
            'angioma': 'angioma',
            'lipoma': 'lipoma',
            'other': 'other'
        }
        
        total_moved = 0
        
        # Find all subdirectories that might contain images
        # Handle both direct class folders and Train/Test split structures
        search_paths = []
        
        # Check for Train/Test subdirectories first
        has_splits = False
        for subdir in isic_path.iterdir():
            if subdir.is_dir() and subdir.name.lower() in ['train', 'test', 'validation']:
                search_paths.append(subdir)
                has_splits = True
        
        # If no splits found, search directly in isic_path
        if not has_splits:
            search_paths = [isic_path]
        
        # Process all search paths
        for search_path in search_paths:
            for item in search_path.iterdir():
                if not item.is_dir():
                    continue
                
                # Skip split directories in parent directory
                if item.name.lower() in ['train', 'test', 'validation']:
                    continue
                
                # Normalize class name
                class_key = item.name.lower().replace('_', '').replace('-', '')
                
                # Find matching class in mapping
                matched_class = None
                for key, value in isic_mapping.items():
                    if key in class_key or class_key in key:
                        matched_class = value
                        break
                
                if not matched_class:
                    matched_class = item.name.lower().replace(' ', '_')
                
                output_class_dir = self.output_dir / matched_class
                output_class_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy images from this class directory
                image_count = 0
                for img_path in item.rglob('*'):
                    if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        dest = output_class_dir / img_path.name
                        if not dest.exists():
                            try:
                                shutil.copy2(img_path, dest)
                                total_moved += 1
                                image_count += 1
                            except Exception as e:
                                pass
                
                if image_count > 0:
                    print(f"  ✓ {item.name}: {image_count} images")
        
        print(f"✓ Organized ISIC 9-class: {total_moved} images")
        return total_moved
    
    def organize_melanoma(self):
        """Organize Melanoma dataset by class"""
        print("\n📂 ORGANIZING MELANOMA DATASET")
        print("-" * 60)
        
        # Find Melanoma dataset directory
        melanoma_path = None
        for p in self.temp_dir.glob('*'):
            if p.is_dir() and ('melanoma' in p.name.lower() or 'hasananwar' in p.name.lower()):
                melanoma_path = p
                break
        
        if not melanoma_path:
            print("❌ Melanoma dataset directory not found")
            return 0
        
        print(f"Found Melanoma dataset at: {melanoma_path.name}")
        
        # Map Melanoma class names
        melanoma_mapping = {
            'benign': 'benign_melanoma',
            'malignant': 'malignant_melanoma',
            'nevus': 'nevus',
            'nevi': 'nevus',
            'other': 'other'
        }
        
        total_moved = 0
        
        # Melanoma dataset typically has Train/Test structure with class subfolders
        for split_dir in melanoma_path.iterdir():
            if not split_dir.is_dir():
                continue
            
            # Look for class folders within splits
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                # Normalize class name
                class_key = class_dir.name.lower().replace('_', '').replace('-', '')
                
                # Find matching class
                matched_class = None
                for key, value in melanoma_mapping.items():
                    if key in class_key or class_key in key:
                        matched_class = value
                        break
                
                if not matched_class:
                    matched_class = class_dir.name.lower().replace(' ', '_')
                
                output_class_dir = self.output_dir / matched_class
                output_class_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy images
                image_count = 0
                for img_path in class_dir.rglob('*'):
                    if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        dest = output_class_dir / img_path.name
                        if not dest.exists():
                            try:
                                shutil.copy2(img_path, dest)
                                total_moved += 1
                                image_count += 1
                            except Exception as e:
                                pass
                
                if image_count > 0:
                    print(f"  ✓ {class_dir.name}: {image_count} images")
        
        print(f"✓ Organized Melanoma: {total_moved} images")
        return total_moved
    
    def organize_dermnet(self):
        """Organize DermNet dataset by class"""
        print("\n📂 ORGANIZING DERMNET DATASET")
        print("-" * 60)
        
        # Find DermNet dataset directory
        dermnet_path = None
        for p in self.temp_dir.glob('*'):
            if p.is_dir() and ('derm' in p.name.lower() or 'shubham' in p.name.lower()):
                dermnet_path = p
                break
        
        if not dermnet_path:
            print("❌ DermNet dataset directory not found")
            return 0
        
        print(f"Found DermNet dataset at: {dermnet_path.name}")
        
        # Map DermNet class names
        dermnet_mapping = {
            'acne': 'acne',
            'acneiform': 'acne',
            'alopecia': 'alopecia',
            'androgenetic': 'alopecia',
            'angioma': 'angioma',
            'angiokeratoma': 'angiokeratoma',
            'atrophic': 'atrophic_scar',
            'scar': 'scar',
            'basal': 'basal_cell_carcinoma',
            'bcc': 'basal_cell_carcinoma',
            'cellulitis': 'cellulitis',
            'dermatitis': 'dermatitis',
            'dermatofibroma': 'dermatofibroma',
            'df': 'dermatofibroma',
            'eczema': 'eczema',
            'hemangioma': 'hemangioma',
            'melasma': 'melasma',
            'melanoma': 'melanoma',
            'mel': 'melanoma',
            'mole': 'mole',
            'nevus': 'nevus',
            'nevi': 'nevus',
            'rosacea': 'rosacea',
            'seborrheic': 'seborrheic_keratosis',
            'sebk': 'seborrheic_keratosis',
            'squamous': 'squamous_cell_carcinoma',
            'scc': 'squamous_cell_carcinoma',
            'tinea': 'tinea',
            'urticaria': 'urticaria',
            'vitiligo': 'vitiligo',
            'wart': 'wart',
            'other': 'other'
        }
        
        total_moved = 0
        
        # DermNet has class folders at top level
        for class_dir in dermnet_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            # Normalize class name
            class_key = class_dir.name.lower().replace('_', '').replace('-', '')
            
            # Find matching class
            matched_class = None
            for key, value in dermnet_mapping.items():
                if key in class_key or class_key in key:
                    matched_class = value
                    break
            
            if not matched_class:
                matched_class = class_dir.name.lower().replace(' ', '_')
            
            output_class_dir = self.output_dir / matched_class
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            image_count = 0
            for img_path in class_dir.rglob('*'):
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest = output_class_dir / img_path.name
                    if not dest.exists():
                        try:
                            shutil.copy2(img_path, dest)
                            total_moved += 1
                            image_count += 1
                        except Exception as e:
                            pass
            
            if image_count > 0:
                print(f"  ✓ {class_dir.name}: {image_count} images")
        
        print(f"✓ Organized DermNet: {total_moved} images")
        return total_moved
    
    def balance_dataset(self):
        """Semi-balance dataset - keep all images, minimal removal for extreme outliers"""
        print("\n⚖️  SEMI-BALANCING DATASET")
        print("-" * 60)
        
        min_images = float('inf')
        max_images = 0
        class_counts = {}
        
        # Count images per class
        for class_dir in self.output_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            count = len(images)
            class_counts[class_dir.name] = count
            
            min_images = min(min_images, count)
            max_images = max(max_images, count)
        
        # Semi-balance: only cap very large classes, keep most images for training
        # Use a soft cap at 400 images per class to provide balanced sampling during training
        target_per_class = 400
        
        print(f"Class distribution (semi-balanced, target: {target_per_class}/class):")
        total_removed = 0
        
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            class_dir = self.output_dir / class_name
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            print(f"  • {class_name}: {count} images", end="")
            
            # Only remove images if class is very large (over target)
            if len(images) > target_per_class:
                to_remove = len(images) - target_per_class
                # Shuffle before removing to ensure random subset
                import random
                random.shuffle(images)
                for img in images[target_per_class:]:
                    img.unlink()
                    total_removed += 1
                print(f" → {target_per_class} (removed {to_remove})")
            else:
                print()
        
        print(f"\n✓ Semi-balanced dataset: removed {total_removed} excess images")
        print(f"  Target per class: {target_per_class} images")
        print(f"  Note: Final balancing will be done during training via weighted sampling")
        
        return target_per_class
    
    def get_dataset_stats(self):
        """Print dataset statistics"""
        print("\n📊 DATASET STATISTICS")
        print("-" * 60)
        
        total_images = 0
        class_counts = {}
        
        for class_dir in sorted(self.output_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            count = len(images)
            class_counts[class_dir.name] = count
            total_images += count
        
        print(f"\n✓ Total Classes: {len(class_counts)}")
        print(f"✓ Total Images: {total_images}")
        print(f"✓ Images per class:")
        
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            print(f"    • {class_name}: {count}")
        
        avg_per_class = total_images / len(class_counts) if class_counts else 0
        print(f"\n✓ Average per class: {avg_per_class:.0f}")
        
        return total_images, len(class_counts)
    
    def cleanup(self):
        """Clean up temporary files"""
        print("\n🧹 CLEANING UP TEMPORARY FILES")
        print("-" * 60)
        
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                print(f"✓ Removed temporary directory")
            except Exception as e:
                print(f"⚠️  Could not remove temp directory: {e}")
    
    def run(self):
        """Run the complete pipeline"""
        print("\n" + "🎯 " * 20)
        print("KAGGLE DATASET BUILDER - AUTOMATED PIPELINE")
        print("🎯 " * 20 + "\n")
        
        # Step 1: Setup Kaggle API
        if not self.setup_kaggle_api():
            print("\n❌ Failed to setup Kaggle API")
            return False
        
        # Step 2: Download datasets
        print("\n" + "=" * 60)
        print("📥 DOWNLOADING DATASETS")
        print("=" * 60)
        
        download_success = {}
        
        # Download ISIC 9-class
        if self.check_dataset_exists('isic9'):
            print(f"\n✓ ISIC 9-class already downloaded (skipping)")
            download_success['isic9'] = True
        else:
            download_success['isic9'] = self.download_dataset('ISIC-9', self.datasets['isic9'])
        
        # Download Melanoma dataset
        if self.check_dataset_exists('melanoma'):
            print(f"\n✓ Melanoma dataset already downloaded (skipping)")
            download_success['melanoma'] = True
        else:
            download_success['melanoma'] = self.download_dataset('Melanoma', self.datasets['skin_melanoma'])
        
        # Download Dermatology dataset
        if self.check_dataset_exists('dermatology'):
            print(f"\n✓ Dermatology dataset already downloaded (skipping)")
            download_success['dermatology'] = True
        else:
            download_success['dermatology'] = self.download_dataset('Dermatology', self.datasets['dermatology'])
        
        if not any(download_success.values()):
            print("\n❌ Failed to download any datasets")
            return False
        
        # Step 3: Organize datasets
        print("\n" + "=" * 60)
        print("📂 ORGANIZING DATASETS")
        print("=" * 60)
        
        self.output_dir.mkdir(exist_ok=True)
        
        total_organized = 0
        isic_count = 0
        melanoma_count = 0
        dermnet_count = 0
        
        if download_success.get('isic9', False):
            isic_count = self.organize_isic9()
        
        if download_success.get('melanoma', False):
            melanoma_count = self.organize_melanoma()
        
        if download_success.get('dermatology', False):
            dermnet_count = self.organize_dermnet()
        
        total = isic_count + melanoma_count + dermnet_count
        if total == 0:
            print("\n❌ No images were organized")
            return False
        
        # Step 4: Balance dataset
        print("\n" + "=" * 60)
        print("⚖️  BALANCING DATASET")
        print("=" * 60)
        
        self.balance_dataset()
        
        # Step 5: Dataset statistics
        total_images, num_classes = self.get_dataset_stats()
        
        # Step 6: Cleanup
        self.cleanup()
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"📊 Classes: {num_classes}")
        print(f"📊 Total images: {total_images}")
        print(f"\n✓ Dataset ready for training!")
        print(f"✓ Expected training time: ~50-70 minutes (25 epochs)")
        print("\nNext steps:")
        print("  1. python train.py")
        print("  2. python app.py")
        print("  3. Open http://localhost:5000")
        
        return True


if __name__ == "__main__":
    # Parse command line arguments
    api_key = None
    username = None
    
    if len(sys.argv) > 1:
        # Check for --api-key and --username arguments
        for i, arg in enumerate(sys.argv[1:]):
            if arg == '--api-key' and i + 2 < len(sys.argv):
                api_key = sys.argv[i + 2]
            elif arg == '--username' and i + 2 < len(sys.argv):
                username = sys.argv[i + 2]
    
    # Try environment variables if not provided via args
    if not api_key:
        api_key = os.environ.get('KAGGLE_API_TOKEN')
    if not username:
        username = os.environ.get('KAGGLE_USERNAME')
    
    builder = KaggleDatassetBuilder(api_key=api_key, username=username)
    success = builder.run()
    
    sys.exit(0 if success else 1)
