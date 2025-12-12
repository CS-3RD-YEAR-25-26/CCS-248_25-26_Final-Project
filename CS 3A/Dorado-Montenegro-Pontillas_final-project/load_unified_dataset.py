"""
Unified Dataset Loader for Filipino Hate Speech Detection
Combines multiple Filipino hate speech datasets into a single strong dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class UnifiedDatasetLoader:
    """Load and combine all Filipino hate speech datasets"""
    
    def __init__(self, base_path='.'):
        self.base_path = base_path
        self.combined_data = None
        
    def load_hatespeech_dataset(self):
        """Load the hatespeech/ folder dataset (Twitter election data)"""
        train = pd.read_csv(os.path.join(self.base_path, 'hatespeech/train.csv'), lineterminator='\n')
        valid = pd.read_csv(os.path.join(self.base_path, 'hatespeech/valid.csv'), lineterminator='\n')
        test = pd.read_csv(os.path.join(self.base_path, 'hatespeech/test.csv'), lineterminator='\n')
        
        # Combine all splits
        combined = pd.concat([train, valid, test], ignore_index=True)
        combined['source'] = 'twitter_election'
        
        print(f"✓ Loaded hatespeech dataset: {len(combined)} samples")
        print(f"  - Hate speech: {combined['label'].sum()}")
        print(f"  - Non-hate: {len(combined) - combined['label'].sum()}")
        
        return combined
    
    def load_tiktok_dataset(self):
        """Load the filipino-tiktok-hatespeech-main/ dataset"""
        train = pd.read_csv(os.path.join(self.base_path, 'filipino-tiktok-hatespeech-main/data/train.csv'))
        valid = pd.read_csv(os.path.join(self.base_path, 'filipino-tiktok-hatespeech-main/data/valid.csv'))
        test = pd.read_csv(os.path.join(self.base_path, 'filipino-tiktok-hatespeech-main/data/test.csv'))
        
        # Clean column names (remove trailing whitespace/carriage returns)
        train.columns = train.columns.str.strip()
        valid.columns = valid.columns.str.strip()
        test.columns = test.columns.str.strip()
        
        # Combine all splits
        combined = pd.concat([train, valid, test], ignore_index=True)
        combined['source'] = 'tiktok'
        
        print(f"✓ Loaded TikTok dataset: {len(combined)} samples")
        print(f"  - Hate speech: {combined['label'].sum()}")
        print(f"  - Non-hate: {len(combined) - combined['label'].sum()}")
        
        return combined
    
    def load_english_cyberbullying_dataset(self):
        """Load the English cyberbullying tweets dataset and convert to binary labels"""
        df = pd.read_csv(os.path.join(self.base_path, 'cyberbullying_tweets.csv'))
        
        # Rename columns to match our format
        df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'original_label'})
        
        # Convert multi-class to binary: not_cyberbullying=0, everything else=1
        df['label'] = (df['original_label'] != 'not_cyberbullying').astype(int)
        df['source'] = 'english_twitter'
        
        # Keep only text, label, and source columns
        df = df[['text', 'label', 'source']]
        
        print(f"✓ Loaded English cyberbullying dataset: {len(df)} samples")
        print(f"  - Cyberbullying: {df['label'].sum()}")
        print(f"  - Non-cyberbullying: {len(df) - df['label'].sum()}")
        
        return df
    
    def load_all_datasets(self):
        """Load and combine all Filipino and English datasets"""
        print("=" * 60)
        print("LOADING ALL HATE SPEECH/CYBERBULLYING DATASETS")
        print("=" * 60)
        
        # Load individual datasets
        hatespeech_df = self.load_hatespeech_dataset()
        tiktok_df = self.load_tiktok_dataset()
        english_df = self.load_english_cyberbullying_dataset()
        
        # Combine all datasets
        self.combined_data = pd.concat([hatespeech_df, tiktok_df, english_df], ignore_index=True)
        
        # Clean the data
        self.combined_data = self._clean_data(self.combined_data)
        
        print("\n" + "=" * 60)
        print("UNIFIED BILINGUAL DATASET STATISTICS")
        print("=" * 60)
        print(f"Total samples: {len(self.combined_data)}")
        print(f"Hate/Cyberbullying samples: {self.combined_data['label'].sum()} ({self.combined_data['label'].mean()*100:.2f}%)")
        print(f"Non-hate samples: {len(self.combined_data) - self.combined_data['label'].sum()} ({(1-self.combined_data['label'].mean())*100:.2f}%)")
        print(f"\nDataset sources:")
        print(self.combined_data['source'].value_counts())
        print(f"\nLanguage distribution:")
        filipino_count = len(self.combined_data[self.combined_data['source'].isin(['twitter_election', 'tiktok'])])
        english_count = len(self.combined_data[self.combined_data['source'] == 'english_twitter'])
        print(f"  Filipino: {filipino_count} samples ({filipino_count/len(self.combined_data)*100:.1f}%)")
        print(f"  English: {english_count} samples ({english_count/len(self.combined_data)*100:.1f}%)")
        print("=" * 60)
        
        return self.combined_data
    
    def _clean_data(self, df):
        """Clean the dataset: remove duplicates, handle missing values"""
        print("\nCleaning dataset...")
        
        # Remove duplicates based on text
        original_len = len(df)
        df = df.drop_duplicates(subset=['text'], keep='first')
        duplicates_removed = original_len - len(df)
        print(f"  - Removed {duplicates_removed} duplicate texts")
        
        # Remove missing values
        df = df.dropna(subset=['text', 'label'])
        print(f"  - Removed missing values")
        
        # Remove empty texts
        df = df[df['text'].str.strip() != '']
        print(f"  - Removed empty texts")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def get_train_val_test_split(self, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """Split the unified dataset into train, validation, and test sets"""
        if self.combined_data is None:
            raise ValueError("No data loaded. Call load_all_datasets() first.")
        
        # Ensure stratified split to maintain class balance
        X = self.combined_data['text'].values
        y = self.combined_data['label'].values
        
        # First split: train and temp (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=train_size, stratify=y, random_state=random_state
        )
        
        # Second split: val and test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_ratio, stratify=y_temp, random_state=random_state
        )
        
        print("\n" + "=" * 60)
        print("TRAIN/VAL/TEST SPLIT")
        print("=" * 60)
        print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  - Hate: {y_train.sum()}, Non-hate: {len(y_train) - y_train.sum()}")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  - Hate: {y_val.sum()}, Non-hate: {len(y_val) - y_val.sum()}")
        print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"  - Hate: {y_test.sum()}, Non-hate: {len(y_test) - y_test.sum()}")
        print("=" * 60)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save_unified_dataset(self, output_path='unified_bilingual_hatespeech.csv'):
        """Save the unified dataset to a CSV file"""
        if self.combined_data is None:
            raise ValueError("No data loaded. Call load_all_datasets() first.")
        
        self.combined_data.to_csv(output_path, index=False)
        print(f"\n✓ Unified bilingual dataset saved to: {output_path}")
        return output_path


def main():
    """Demo: Load and display unified bilingual dataset statistics"""
    loader = UnifiedDatasetLoader()
    
    # Load all datasets
    unified_data = loader.load_all_datasets()
    
    # Get train/val/test splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.get_train_val_test_split()
    
    # Save unified dataset
    loader.save_unified_dataset()
    
    # Show sample texts
    print("\n" + "=" * 60)
    print("SAMPLE TEXTS FROM UNIFIED BILINGUAL DATASET")
    print("=" * 60)
    print("\nHate Speech/Cyberbullying Examples:")
    hate_samples = unified_data[unified_data['label'] == 1].sample(min(5, len(unified_data[unified_data['label'] == 1])))
    for idx, row in hate_samples.iterrows():
        print(f"  [{row['source']}] {row['text'][:100]}...")
    
    print("\nNon-Hate Examples:")
    non_hate_samples = unified_data[unified_data['label'] == 0].sample(min(5, len(unified_data[unified_data['label'] == 0])))
    for idx, row in non_hate_samples.iterrows():
        print(f"  [{row['source']}] {row['text'][:100]}...")
    print("=" * 60)


if __name__ == '__main__':
    main()
