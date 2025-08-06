#!/usr/bin/env python3
"""
Dataset Setup Script for Plastic Type Classification
Author: GURUPRASHATH R (RA2311003020078)
"""

import os
import shutil
import zipfile
from pathlib import Path

# Plastic types
PLASTIC_TYPES = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS"]

def create_dataset_structure():
    """Create the required dataset directory structure"""
    print("♻️ Creating PlasticNet Dataset Structure")
    print("=" * 50)
    
    # Create main dataset directory
    os.makedirs("dataset", exist_ok=True)
    
    # Create subdirectories for each plastic type
    for plastic_type in PLASTIC_TYPES:
        os.makedirs(f"dataset/{plastic_type}", exist_ok=True)
        print(f"✅ Created: dataset/{plastic_type}/")
    
    print("\n📁 Dataset structure created successfully!")
    print("dataset/")
    for plastic_type in PLASTIC_TYPES:
        print(f"  ├── {plastic_type}/")
    print("      └── (place your images here)")

def extract_kaggle_dataset(zip_path):
    """Extract and organize Kaggle dataset"""
    print(f"\n📦 Extracting dataset from: {zip_path}")
    
    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found: {zip_path}")
        return False
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to temporary directory
            temp_dir = "temp_extract"
            zip_ref.extractall(temp_dir)
            
            print("✅ Dataset extracted successfully!")
            
            # Organize files
            organize_extracted_files(temp_dir)
            
            # Clean up
            shutil.rmtree(temp_dir)
            return True
            
    except Exception as e:
        print(f"❌ Error extracting dataset: {e}")
        return False

def organize_extracted_files(extract_dir):
    """Organize extracted files into proper structure"""
    print("🔄 Organizing files...")
    
    # Look for common Kaggle dataset structures
    possible_structures = [
        # Structure 1: Direct class folders
        extract_dir,
        # Structure 2: Nested in a main folder
        os.path.join(extract_dir, "plastic_dataset"),
        os.path.join(extract_dir, "dataset"),
        os.path.join(extract_dir, "data"),
        # Structure 3: Train/test split
        os.path.join(extract_dir, "train"),
        os.path.join(extract_dir, "data", "train"),
    ]
    
    source_dir = None
    for structure in possible_structures:
        if os.path.exists(structure):
            # Check if it contains our plastic type folders
            has_plastic_types = any(
                os.path.exists(os.path.join(structure, plastic_type)) 
                for plastic_type in PLASTIC_TYPES
            )
            if has_plastic_types:
                source_dir = structure
                break
    
    if source_dir is None:
        print("⚠️  Could not automatically detect dataset structure.")
        print("Please manually organize your images into the following structure:")
        print_dataset_instructions()
        return
    
    print(f"📂 Found dataset in: {source_dir}")
    
    # Copy files to proper structure
    total_files = 0
    for plastic_type in PLASTIC_TYPES:
        source_folder = os.path.join(source_dir, plastic_type)
        target_folder = f"dataset/{plastic_type}"
        
        if os.path.exists(source_folder):
            # Copy all image files
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            files_copied = 0
            
            for filename in os.listdir(source_folder):
                if filename.lower().endswith(image_extensions):
                    source_file = os.path.join(source_folder, filename)
                    target_file = os.path.join(target_folder, filename)
                    shutil.copy2(source_file, target_file)
                    files_copied += 1
            
            print(f"📁 {plastic_type}: {files_copied} images copied")
            total_files += files_copied
    
    print(f"\n✅ Dataset organization completed! Total images: {total_files}")

def print_dataset_instructions():
    """Print instructions for manual dataset setup"""
    print("\n📋 Manual Dataset Setup Instructions:")
    print("=" * 50)
    print("1. Download 'Dataset for Visual Plastic Type Recognition' from Kaggle")
    print("   - Kaggle Link: https://www.kaggle.com/datasets/harshitkandoi/plastic-type-recognition")
    print("   - Author: Harshit Kandoi")
    print()
    print("2. Extract the downloaded zip file")
    print()
    print("3. Organize images into the following structure:")
    print("   dataset/")
    for plastic_type in PLASTIC_TYPES:
        print(f"   ├── {plastic_type}/")
        print(f"   │   ├── {plastic_type.lower()}_image1.jpg")
        print(f"   │   ├── {plastic_type.lower()}_image2.jpg")
        print(f"   │   └── ... (all {plastic_type} images)")
    print()
    print("4. Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff")
    print()
    print("5. Run this script again to verify the setup")

def verify_dataset():
    """Verify that the dataset is properly organized"""
    print("\n🔍 Verifying dataset...")
    
    total_images = 0
    missing_folders = []
    
    for plastic_type in PLASTIC_TYPES:
        folder_path = f"dataset/{plastic_type}"
        if not os.path.exists(folder_path):
            missing_folders.append(plastic_type)
            continue
        
        # Count images in folder
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(image_extensions)]
        
        print(f"📁 {plastic_type}: {len(images)} images")
        total_images += len(images)
    
    if missing_folders:
        print(f"\n❌ Missing folders: {', '.join(missing_folders)}")
        return False
    
    if total_images == 0:
        print("\n❌ No images found in dataset!")
        return False
    
    print(f"\n✅ Dataset verification completed!")
    print(f"📊 Total images: {total_images}")
    print(f"🎯 Plastic types: {len(PLASTIC_TYPES)}")
    
    return True

def main():
    print("♻️ PlasticNet Dataset Setup")
    print("=" * 60)
    print("This script helps you set up the dataset for plastic type classification.")
    print()
    
    # Create directory structure
    create_dataset_structure()
    
    # Check for existing dataset
    if verify_dataset():
        print("\n🎉 Dataset is ready for training!")
        print("🚀 You can now run: python train_model.py")
        return
    
    # Look for zip files
    zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
    
    if zip_files:
        print(f"\n📦 Found zip files: {', '.join(zip_files)}")
        choice = input("Would you like to extract one of these files? (y/n): ").lower()
        
        if choice == 'y':
            if len(zip_files) == 1:
                zip_path = zip_files[0]
            else:
                print("Available zip files:")
                for i, zip_file in enumerate(zip_files, 1):
                    print(f"  {i}. {zip_file}")
                choice = int(input("Select file number: ")) - 1
                zip_path = zip_files[choice]
            
            if extract_kaggle_dataset(zip_path):
                verify_dataset()
                print("\n🎉 Dataset setup completed!")
                print("🚀 You can now run: python train_model.py")
            else:
                print_dataset_instructions()
        else:
            print_dataset_instructions()
    else:
        print_dataset_instructions()

if __name__ == "__main__":
    main() 