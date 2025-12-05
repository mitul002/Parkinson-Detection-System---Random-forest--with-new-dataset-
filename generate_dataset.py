"""
Generate features dataset for Parkinson's Disease Detection
This script processes all images and generates a CSV file with extracted features
"""

import os
import pandas as pd
from feature_extraction import SpiralFeatureExtractor
from tqdm import tqdm

def process_dataset(data_dir, output_csv):
    """Process all images in the dataset and save features to CSV"""
    print("Initializing feature extractor...")
    extractor = SpiralFeatureExtractor(use_cnn=True)
    
    rows = []
    # Process each class folder
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"\nProcessing {label} images...")
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc=f"Extracting features for {label}"):
            img_path = os.path.join(class_dir, img_file)
            features = extractor.extract_features(img_path)
            
            if features is not None:
                # Get handcrafted features
                handcrafted = features['handcrafted_features']
                # Add CNN features with prefix
                for i, cnn_feat in enumerate(features['cnn_features']):
                    handcrafted[f'cnn_feat_{i}'] = cnn_feat
                    
                # Add metadata
                handcrafted['Image'] = img_file
                handcrafted['Label'] = label
                rows.append(handcrafted)
    
    # Create DataFrame and save to CSV
    print("\nCreating dataset...")
    df = pd.DataFrame(rows)
    
    # Reorder columns to put Image and Label first
    cols = ['Image', 'Label'] + [col for col in df.columns if col not in ['Image', 'Label']]
    df = df[cols]
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nDataset saved to: {output_csv}")
    print(f"Total samples: {len(df)}")
    print("\nClass distribution:")
    print(df['Label'].value_counts())
    
    return df

if __name__ == "__main__":
    # Use the DATASET folder in model directory
    data_dir = "DATASET"  # Folder containing class subfolders
    output_csv = "spiral_feature_dataset.csv"
    
    df = process_dataset(data_dir, output_csv)