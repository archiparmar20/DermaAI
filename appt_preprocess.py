#!/usr/bin/env python3
"""
APPT Dataset Preprocessor
- Removes dark (near-black) background areas from images
- Crops or removes images with excessive dark regions
- Augments data to balance class distribution
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json
import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class APPTPreprocessor:
    def __init__(self, root_dir='appt_data', dark_threshold=50, dark_ratio_threshold=0.6):
        """
        Args:
            root_dir: Root directory containing train/ and test/ subdirectories
            dark_threshold: Pixel value threshold for "dark" (0-255), default 50 = near black
            dark_ratio_threshold: If dark pixels > this ratio, remove image entirely
        """
        self.root_dir = Path(root_dir)
        self.train_dir = self.root_dir / 'train'
        self.test_dir = self.root_dir / 'test'
        self.train_pre_dir = self.root_dir / 'train_pre'
        self.test_pre_dir = self.root_dir / 'test_pre'
        
        self.dark_threshold = dark_threshold
        self.dark_ratio_threshold = dark_ratio_threshold
        
        # Track statistics
        self.stats = {
            'processed': defaultdict(int),
            'removed': defaultdict(int),
            'cropped': defaultdict(int),
            'skipped': defaultdict(int),
            'augmented': defaultdict(int)
        }
        
    def setup_output_dirs(self):
        """Create output directories maintaining class structure"""
        for src_split, dst_split in [('train', self.train_pre_dir), ('test', self.test_pre_dir)]:
            src_dir = self.root_dir / src_split
            
            if src_dir.exists():
                for class_name in os.listdir(src_dir):
                    class_path = src_dir / class_name
                    if class_path.is_dir():
                        output_class_path = dst_split / class_name
                        output_class_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output directories created: {self.train_pre_dir}, {self.test_pre_dir}")
    
    def detect_dark_regions(self, image_array):
        """
        Detect dark (near-black) regions in image.
        
        Returns:
            mask: Boolean array where True = dark pixel
            dark_ratio: Ratio of dark pixels to total pixels
        """
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        
        # Create mask: True where pixels are darker than threshold
        mask = gray < self.dark_threshold
        dark_ratio = mask.sum() / mask.size
        
        return mask, dark_ratio
    
    def find_largest_contour(self, mask):
        """
        Find the largest contiguous non-dark region.
        Returns bounding box (x, y, w, h) of the largest content area.
        """
        # Invert mask: True = content (non-dark)
        content_mask = ~mask
        
        # Find contours
        contours, _ = cv2.findContours(content_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        return (x, y, w, h)
    
    def crop_image(self, image_array, bbox):
        """Crop image to bounding box with small margin"""
        x, y, w, h = bbox
        # Add small margin
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image_array.shape[1] - x, w + 2*margin)
        h = min(image_array.shape[0] - y, h + 2*margin)
        
        return image_array[y:y+h, x:x+w]
    
    def preprocess_image(self, image_path, class_name, split):
        """
        Preprocess single image.
        
        Returns:
            (processed_image, action) where action is 'processed', 'cropped', 'removed', 'skipped'
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                self.stats['skipped'][class_name] += 1
                return None, 'skipped'
            
            # Detect dark regions
            dark_mask, dark_ratio = self.detect_dark_regions(image)
            
            # If dark ratio < 5%, skip preprocessing (no significant dark areas)
            if dark_ratio < 0.05:
                self.stats['skipped'][class_name] += 1
                return image, 'skipped'
            
            # If dark ratio > threshold, remove image entirely
            if dark_ratio > self.dark_ratio_threshold:
                self.stats['removed'][class_name] += 1
                return None, 'removed'
            
            # Otherwise, crop to remove dark regions
            bbox = self.find_largest_contour(dark_mask)
            if bbox is None:
                self.stats['removed'][class_name] += 1
                return None, 'removed'
            
            cropped = self.crop_image(image, bbox)
            
            # Ensure minimum size (e.g., 64x64)
            if cropped.shape[0] < 64 or cropped.shape[1] < 64:
                self.stats['removed'][class_name] += 1
                return None, 'removed'
            
            self.stats['cropped'][class_name] += 1
            return cropped, 'cropped'
            
        except Exception as e:
            print(f"  ⚠ Error processing {image_path}: {e}")
            self.stats['skipped'][class_name] += 1
            return None, 'skipped'
    
    def process_split(self, split_name):
        """Process train or test split"""
        src_dir = self.root_dir / split_name
        dst_dir = self.train_pre_dir if split_name == 'train' else self.test_pre_dir
        
        if not src_dir.exists():
            print(f"⚠ Source directory not found: {src_dir}")
            return
        
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*60}")
        
        # Iterate through classes
        classes = sorted([d for d in os.listdir(src_dir) if (src_dir / d).is_dir()])
        
        for class_name in classes:
            class_src = src_dir / class_name
            class_dst = dst_dir / class_name
            
            # Find images with case-insensitive extension matching
            all_files = os.listdir(class_src)
            image_files = [f for f in all_files
                          if os.path.isfile(os.path.join(class_src, f)) and
                          os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp']]
            
            if not image_files:
                continue
            
            print(f"\n{class_name}: ", end='', flush=True)
            
            for img_name in tqdm(image_files, leave=False):
                img_path = class_src / img_name
                processed, action = self.preprocess_image(img_path, class_name, split_name)
                
                if processed is not None:
                    # Save processed image
                    output_path = class_dst / img_name
                    cv2.imwrite(str(output_path), processed)
                    self.stats['processed'][class_name] += 1
            
            # Print summary for this class
            total_input = len(image_files)
            processed_count = self.stats['processed'][class_name]
            removed_count = self.stats['removed'][class_name]
            cropped_count = self.stats['cropped'][class_name]
            
            print(f"  Input: {total_input} | Processed: {processed_count} | "
                  f"Cropped: {cropped_count} | Removed: {removed_count} | "
                  f"Skipped: {self.stats['skipped'][class_name]}")
    
    def augment_data(self):
        """
        Augment data to balance class distribution.
        """
        print(f"\n{'='*60}")
        print("Augmenting data to balance classes")
        print(f"{'='*60}")
        
        for split_name, src_dir in [('Train', self.train_pre_dir), ('Test', self.test_pre_dir)]:
            if not src_dir.exists():
                continue
            
            print(f"\n{split_name} split:")
            
            # Count images per class
            class_counts = {}
            for class_name in os.listdir(src_dir):
                class_path = src_dir / class_name
                if class_path.is_dir():
                    count = len([f for f in os.listdir(class_path) 
                               if os.path.isfile(os.path.join(class_path, f)) and
                               os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
                    class_counts[class_name] = count
            
            if not class_counts:
                print("  No classes found")
                continue
            
            # Find median class size as target
            median_count = int(np.median(list(class_counts.values())))
            max_count = max(class_counts.values())
            target_count = int(max_count * 0.9)  # Target 90% of max
            
            print(f"  Class distribution: min={min(class_counts.values())}, "
                  f"median={median_count}, max={max_count}")
            print(f"  Target count per class: {target_count}")
            
            # Define augmentation transforms
            transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
            
            total_augmented = 0
            
            # Augment underrepresented classes
            for class_name, count in sorted(class_counts.items()):
                if count >= target_count:
                    continue
                
                class_path = src_dir / class_name
                images_needed = target_count - count
                
                # Get existing images
                existing_images = sorted([f for f in os.listdir(class_path) 
                                        if os.path.isfile(os.path.join(class_path, f)) and
                                        os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
                
                if not existing_images:
                    continue
                
                print(f"  {class_name}: {count} -> {target_count} (+{images_needed})", end=' ', flush=True)
                
                augmented_count = 0
                for i in range(images_needed):
                    # Cycle through existing images
                    src_img_name = existing_images[i % len(existing_images)]
                    src_img_path = class_path / src_img_name
                    
                    try:
                        # Load and augment
                        img = Image.open(src_img_path).convert('RGB')
                        augmented = transform(img)
                        
                        # Save with new name
                        base_name, ext = os.path.splitext(src_img_name)
                        aug_name = f"{base_name}_aug_{i}{ext}"
                        aug_path = class_path / aug_name
                        augmented.save(str(aug_path))
                        
                        augmented_count += 1
                        self.stats['augmented'][class_name] += 1
                        total_augmented += 1
                        
                    except Exception as e:
                        print(f"  ⚠ Augmentation error for {src_img_name}: {e}")
                
                print(f"✓ ({augmented_count})")
            
            # Print final statistics for this split
            print(f"\n  Final class distribution for {split_name} split:")
            for class_name in sorted(os.listdir(src_dir)):
                class_path = src_dir / class_name
                if class_path.is_dir():
                    final_count = len([f for f in os.listdir(class_path) 
                                      if os.path.isfile(os.path.join(class_path, f)) and
                                      os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
                    print(f"    {class_name}: {final_count}")
    
    def print_summary(self):
        """Print preprocessing summary"""
        print(f"\n{'='*60}")
        print("PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        
        total_processed = sum(self.stats['processed'].values())
        total_removed = sum(self.stats['removed'].values())
        total_cropped = sum(self.stats['cropped'].values())
        total_skipped = sum(self.stats['skipped'].values())
        total_augmented = sum(self.stats['augmented'].values())
        
        print(f"\nImages Processed: {total_processed}")
        print(f"Images Removed (>60% dark): {total_removed}")
        print(f"Images Cropped (<60% dark): {total_cropped}")
        print(f"Images Skipped (<5% dark): {total_skipped}")
        print(f"Images Augmented: {total_augmented}")
        
        # Per-class breakdown
        print(f"\nPer-class breakdown:")
        print(f"{'Class':<30} {'Processed':<12} {'Removed':<12} {'Cropped':<12}")
        print("-" * 66)
        
        for class_name in sorted(self.stats['processed'].keys()):
            processed = self.stats['processed'][class_name]
            removed = self.stats['removed'][class_name]
            cropped = self.stats['cropped'][class_name]
            print(f"{class_name:<30} {processed:<12} {removed:<12} {cropped:<12}")
    
    def run(self):
        """Run complete preprocessing pipeline"""
        print("\n" + "="*60)
        print("APPT DATASET PREPROCESSING")
        print("="*60)
        print(f"Root directory: {self.root_dir}")
        print(f"Dark threshold: {self.dark_threshold}")
        print(f"Dark ratio threshold: {self.dark_ratio_threshold*100:.0f}%")
        
        # Setup output directories
        self.setup_output_dirs()
        
        # Process both splits
        self.process_split('train')
        self.process_split('test')
        
        # Augment data to balance classes
        self.augment_data()
        
        # Print summary
        self.print_summary()
        
        print(f"\n✓ Preprocessing complete!")
        print(f"  Train preprocessed: {self.train_pre_dir}")
        print(f"  Test preprocessed: {self.test_pre_dir}")


if __name__ == '__main__':
    # Create preprocessor and run
    preprocessor = APPTPreprocessor(
        root_dir='appt_data',
        dark_threshold=50,        # Pixel values < 50 are considered dark (0-255 scale)
        dark_ratio_threshold=0.6  # Remove if > 60% dark
    )
    
    preprocessor.run()
