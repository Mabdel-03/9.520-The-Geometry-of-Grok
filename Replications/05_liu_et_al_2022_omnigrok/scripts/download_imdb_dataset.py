#!/usr/bin/env python3
"""
Script to download the IMDb dataset for Paper 05 (Omnigrok)

This script attempts to download the IMDb dataset from Kaggle.
Requirements:
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials: ~/.kaggle/kaggle.json
   Get API token from: https://www.kaggle.com/settings -> API -> Create New API Token

If Kaggle API is not available, provides manual download instructions.
"""

import os
import sys
from pathlib import Path

def download_imdb_dataset():
    """Download IMDb dataset from Kaggle"""
    
    # Target directory
    target_dir = Path(__file__).parent.parent / "imdb" / "grokking"
    target_file = target_dir / "IMDB Dataset.csv"
    
    # Check if already downloaded
    if target_file.exists():
        print(f"✓ IMDb dataset already exists at: {target_file}")
        return True
    
    print("=" * 70)
    print("IMDb Dataset Download")
    print("=" * 70)
    print()
    
    # Try to use Kaggle API
    try:
        import kaggle
        print("✓ Kaggle package found")
        print("Downloading dataset from Kaggle...")
        print()
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            'lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
            path=str(target_dir),
            unzip=True
        )
        
        if target_file.exists():
            print()
            print("=" * 70)
            print("✓ SUCCESS: IMDb dataset downloaded successfully!")
            print(f"  Location: {target_file}")
            print("=" * 70)
            return True
        else:
            print()
            print("⚠ Download completed but file not found at expected location")
            print(f"  Please check: {target_dir}")
            return False
            
    except ImportError:
        print("✗ Kaggle package not installed")
        print()
        print("Install with: pip install kaggle")
        print()
        manual_download_instructions()
        return False
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print()
        manual_download_instructions()
        return False

def manual_download_instructions():
    """Print manual download instructions"""
    target_dir = Path(__file__).parent.parent / "imdb" / "grokking"
    
    print("=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print()
    print("1. Visit Kaggle:")
    print("   https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    print()
    print("2. Download 'IMDB Dataset.csv'")
    print()
    print("3. Place the file in:")
    print(f"   {target_dir}")
    print()
    print("4. Run the IMDb experiment:")
    print("   sbatch scripts/run_imdb.sh")
    print()
    print("=" * 70)
    print()
    print("ALTERNATIVE: Set up Kaggle API")
    print("=" * 70)
    print()
    print("1. Install Kaggle package:")
    print("   pip install kaggle")
    print()
    print("2. Get API credentials:")
    print("   - Go to: https://www.kaggle.com/settings")
    print("   - Click 'Create New API Token'")
    print("   - Save kaggle.json to: ~/.kaggle/kaggle.json")
    print()
    print("3. Run this script again:")
    print(f"   python {__file__}")
    print()
    print("=" * 70)

if __name__ == "__main__":
    success = download_imdb_dataset()
    sys.exit(0 if success else 1)

