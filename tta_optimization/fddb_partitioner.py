#!/usr/bin/env python3
"""
Simple FDDB Data Partitioner
Creates 10%-90% split for TTA optimization.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json

# Add parent directory to import ellipse_rcnn modules
sys.path.insert(0, '..')


class FDDBSimplePartitioner:
    """
    Simple FDDB dataset partitioner for TTA optimization.
    Creates 10%-90% split (optimization-test) for practical TTA analysis.
    """
    
    def __init__(self, data_root: str = "../data/FDDB", seed: int = 42):
        """
        Initialize the FDDB simple partitioner.
        
        Args:
            data_root: Path to FDDB dataset
            seed: Random seed for reproducible splits
        """
        self.data_root = Path(data_root)
        self.seed = seed
        np.random.seed(seed)
        
        print(f"🗂️  Initializing Simple FDDB partitioner")
        print(f"📁 Data root: {self.data_root}")
        print(f"🎲 Seed: {seed}")
        
        # Find all images
        self.all_images = self._find_all_images()
        
    def _find_all_images(self) -> List[str]:
        """Find all images in the FDDB dataset."""
        print(f"\n🔍 Finding all images in dataset...")
        
        images_dir = self.data_root / "images"
        if not images_dir.exists():
            # Try alternative structure
            possible_dirs = [
                self.data_root / "originalPics",
                self.data_root / "FDDB-pics",
                self.data_root
            ]
            
            for alt_dir in possible_dirs:
                if alt_dir.exists():
                    images_dir = alt_dir
                    break
        
        if not images_dir.exists():
            raise ValueError(f"Could not find images directory in {self.data_root}")
        
        # Find all image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(list(images_dir.rglob(ext)))
        
        # Convert to strings and sort for reproducibility
        all_images = sorted([str(img) for img in all_images])
        
        print(f"📷 Found {len(all_images)} images in {images_dir}")
        
        if len(all_images) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        return all_images
    
    def create_tta_optimization_splits(self, optimization_fraction: float = 0.1) -> Dict[str, List[str]]:
        """
        Create simple TTA optimization splits.
        
        Args:
            optimization_fraction: Fraction for optimization set (default 0.1 = 10%)
            
        Returns:
            Dictionary with split assignments:
            - 'optimization': Small set for TTA parameter optimization
            - 'test': Large set for final evaluation
        """
        print(f"\n📊 Creating simple TTA optimization splits...")
        print(f"   Optimization fraction: {optimization_fraction:.1%}")
        print(f"   Test fraction: {1-optimization_fraction:.1%}")
        
        # Shuffle images for random split
        shuffled_images = self.all_images.copy()
        np.random.shuffle(shuffled_images)
        
        # Calculate split point
        total_images = len(shuffled_images)
        split_point = int(total_images * optimization_fraction)
        
        # Create splits
        optimization_images = shuffled_images[:split_point]
        test_images = shuffled_images[split_point:]
        
        splits = {
            "optimization": optimization_images,
            "test": test_images
        }
        
        print(f"\n📈 Split sizes:")
        print(f"   Optimization: {len(optimization_images)} images ({len(optimization_images)/total_images:.1%})")
        print(f"   Test: {len(test_images)} images ({len(test_images)/total_images:.1%})")
        print(f"   Total: {total_images} images")
        
        return splits
    
    def save_splits(self, splits: Dict[str, List[str]], 
                   optimization_fraction: float = 0.1,
                   output_dir: str = "fddb_splits"):
        """Save the simple splits to files with metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save splits
        with open(output_path / "fddb_splits.json", 'w', encoding='utf-8') as f:
            json.dump(splits, f, indent=2)
        
        # Save metadata
        metadata = {
            "optimization_fraction": optimization_fraction,
            "test_fraction": 1.0 - optimization_fraction,
            "optimization_size": len(splits["optimization"]),
            "test_size": len(splits["test"]),
            "total_images": len(splits["optimization"]) + len(splits["test"]),
            "data_root": str(self.data_root),
            "seed": self.seed,
            "split_type": "simple_random"
        }
        
        with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Create human-readable summary
        self._create_summary(splits, metadata, output_path)
        
        print(f"\n💾 Simple splits saved to: {output_path}")
    
    def _create_summary(self, splits: Dict[str, List[str]], 
                       metadata: Dict[str, Any], output_path: Path):
        """Create a human-readable summary of the splits."""
        summary_path = output_path / "Split_Summary.md"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# Simple FDDB Splits for TTA Optimization\n\n")
            f.write(f"**Generated:** {np.datetime64('now')}\n")
            f.write(f"**Dataset:** {self.data_root}\n")
            f.write(f"**Split Type:** Simple Random Split\n")
            f.write(f"**Seed:** {self.seed}\n\n")
            
            # Overview
            f.write("## 📊 Split Overview\n\n")
            f.write("Simple 10%-90% split for practical TTA optimization:\n")
            f.write(f"- **Optimization Set:** {metadata['optimization_size']} images ({metadata['optimization_fraction']:.1%})\n")
            f.write(f"- **Test Set:** {metadata['test_size']} images ({metadata['test_fraction']:.1%})\n\n")
            
            f.write("| Split | Images | Percentage |\n")
            f.write("|-------|--------|-----------|\n")
            f.write(f"| Optimization | {metadata['optimization_size']} | {metadata['optimization_fraction']:.1%} |\n")
            f.write(f"| Test | {metadata['test_size']} | {metadata['test_fraction']:.1%} |\n")
            f.write(f"| **Total** | **{metadata['total_images']}** | **100.0%** |\n\n")
            
            # Usage
            f.write("## 🚀 Usage\n\n")
            f.write("```python\n")
            f.write("import json\n")
            f.write("with open('fddb_splits.json') as f:\n")
            f.write("    splits = json.load(f)\n")
            f.write("    \n")
            f.write("optimization_images = splits['optimization']  # TTA parameter optimization\n")
            f.write("test_images = splits['test']                  # Final evaluation\n")
            f.write("```\n")
        
        print(f"📋 Split summary saved to: {summary_path}")


def main():
    """Main function to create simple FDDB partitions."""
    print("🗂️  SIMPLE FDDB PARTITIONING FOR TTA OPTIMIZATION")
    print("="*50)
    print("Creating 10%-90% split for practical TTA analysis")
    
    # Configuration
    DATA_ROOT = "../data/FDDB"
    SEED = 42
    OPTIMIZATION_FRACTION = 0.1  # 10% for optimization, 90% for testing
    
    # Create partitioner
    partitioner = FDDBSimplePartitioner(DATA_ROOT, SEED)
    
    # Create simple splits
    print(f"\n" + "="*50)
    print("CREATING SIMPLE TTA OPTIMIZATION SPLITS")
    splits = partitioner.create_tta_optimization_splits(OPTIMIZATION_FRACTION)
    
    # Save splits
    partitioner.save_splits(splits, OPTIMIZATION_FRACTION)
    
    print(f"\n🎉 Simple FDDB partitioning completed!")
    print(f"📊 Splits saved to: fddb_splits/")
    print(f"\n✅ Ready for TTA optimization!")


if __name__ == "__main__":
    main()
