#!/usr/bin/env python3
"""
Dataset Download and Preparation Script
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This script downloads the BraTS2021 dataset using Kaggle API and prepares it for federated learning.
"""

import os
import shutil
import zipfile
import argparse
from pathlib import Path
from typing import List, Tuple

import kaggle
import nibabel as nib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class BraTSDatasetDownloader:
    """BraTS Dataset Downloader and Preprocessor"""

    def __init__(self, config: dict):
        self.config = config
        self.data_dir = Path(config['dataset']['data_dir'])
        self.processed_dir = Path(config['dataset']['processed_dir'])
        self.modalities = config['dataset']['modalities']
        self.target_size = config['dataset']['target_size']

        # Setup Kaggle API
        self._setup_kaggle()

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _setup_kaggle(self):
        """Setup Kaggle API credentials"""
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)

        # Copy kaggle.json to proper location
        kaggle_json_src = Path('kaggle.json')
        kaggle_json_dst = kaggle_dir / 'kaggle.json'

        if kaggle_json_src.exists():
            shutil.copy(kaggle_json_src, kaggle_json_dst)
            kaggle_json_dst.chmod(0o600)
            print("✓ Kaggle API credentials configured")
        else:
            raise FileNotFoundError("kaggle.json not found in project root")

    def download_dataset(self):
        """Download BraTS2021 dataset from Kaggle"""
        dataset_name = self.config['dataset']['kaggle_dataset']

        print(f"📥 Downloading {dataset_name} from Kaggle...")

        try:
            kaggle.api.competition_download_files(
                'rsna-miccai-brain-tumor-radiogenomic-classification',
                path=self.data_dir,
                quiet=False
            )
            print("✓ Dataset downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download dataset: {e}")
            print("Please ensure you have accepted the competition rules on Kaggle")
            return False

        return True

    def extract_dataset(self):
        """Extract downloaded zip files"""
        print("📦 Extracting dataset...")

        zip_files = list(self.data_dir.glob("*.zip"))
        if not zip_files:
            print("❌ No zip files found")
            return False

        for zip_file in zip_files:
            print(f"Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

        print("✓ Dataset extracted successfully")
        return True

    def preprocess_data(self):
        """Preprocess MRI data for federated learning"""
        print("🔄 Preprocessing data...")

        # Find all patient directories
        train_dir = self.data_dir / 'train'
        if not train_dir.exists():
            print("❌ Training data directory not found")
            return False

        patient_dirs = list(train_dir.glob("*/"))
        print(f"Found {len(patient_dirs)} patients")

        # Create federated splits
        self._create_federated_splits(patient_dirs)

        # Process each patient
        processed_data = []
        for patient_dir in tqdm(patient_dirs[:100], desc="Processing patients"):  # Limit for demo
            patient_data = self._process_patient(patient_dir)
            if patient_data:
                processed_data.append(patient_data)

        # Save processed data
        self._save_processed_data(processed_data)

        print("✓ Data preprocessing completed")
        return True

    def _create_federated_splits(self, patient_dirs: List[Path]):
        """Create federated data splits simulating different hospitals"""
        print("🏥 Creating federated splits...")

        # Split patients into 4 virtual hospitals
        splits = train_test_split(
            patient_dirs,
            test_size=0.25,
            random_state=42,
            shuffle=True
        )

        hospital_dirs = []
        for i, split in enumerate([splits[0][:len(splits[0])//3],
                                  splits[0][len(splits[0])//3:2*len(splits[0])//3],
                                  splits[0][2*len(splits[0])//3:],
                                  splits[1]]):
            hospital_dir = self.processed_dir / f"hospital_{chr(97+i)}"  # a, b, c, d
            hospital_dir.mkdir(exist_ok=True)

            # Save split information
            split_file = hospital_dir / "patients.txt"
            with open(split_file, 'w') as f:
                for patient_dir in split:
                    f.write(f"{patient_dir.name}\n")

            hospital_dirs.append(hospital_dir)

        print(f"✓ Created {len(hospital_dirs)} hospital splits")
        return hospital_dirs

    def _process_patient(self, patient_dir: Path) -> dict:
        """Process single patient data"""
        try:
            # Load MRI modalities
            images = {}
            for modality in self.modalities:
                file_path = patient_dir / f"{patient_dir.name}_{modality}.nii.gz"
                if file_path.exists():
                    img = nib.load(file_path)
                    images[modality] = img.get_fdata()
                else:
                    return None

            # Load segmentation mask
            seg_path = patient_dir / f"{patient_dir.name}_seg.nii.gz"
            if seg_path.exists():
                seg = nib.load(seg_path)
                mask = seg.get_fdata()
            else:
                return None

            # Normalize and resize
            processed_images = {}
            for modality, img in images.items():
                # Normalize to [0, 1]
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                processed_images[modality] = img.astype(np.float32)

            # Process segmentation mask
            processed_mask = mask.astype(np.int32)

            return {
                'patient_id': patient_dir.name,
                'images': processed_images,
                'mask': processed_mask
            }

        except Exception as e:
            print(f"❌ Error processing {patient_dir.name}: {e}")
            return None

    def _save_processed_data(self, processed_data: List[dict]):
        """Save processed data to disk"""
        print("💾 Saving processed data...")

        for data in processed_data:
            patient_dir = self.processed_dir / data['patient_id']
            patient_dir.mkdir(exist_ok=True)

            # Save images
            for modality, img in data['images'].items():
                np.save(patient_dir / f"{modality}.npy", img)

            # Save mask
            np.save(patient_dir / "mask.npy", data['mask'])

    def get_dataset_stats(self):
        """Get dataset statistics"""
        print("📊 Computing dataset statistics...")

        stats = {
            'total_patients': 0,
            'modalities': self.modalities,
            'image_shape': None,
            'classes': [0, 1, 2, 3],  # Background + tumor classes
            'class_distribution': {}
        }

        processed_patients = list(self.processed_dir.glob("*/"))
        stats['total_patients'] = len(processed_patients)

        if processed_patients:
            # Load first patient to get shape
            sample_img = np.load(processed_patients[0] / "t1.npy")
            stats['image_shape'] = sample_img.shape

            # Compute class distribution
            class_counts = np.zeros(4)
            for patient_dir in processed_patients[:50]:  # Sample for stats
                mask = np.load(patient_dir / "mask.npy")
                unique, counts = np.unique(mask, return_counts=True)
                for cls, count in zip(unique, counts):
                    if cls < 4:
                        class_counts[int(cls)] += count

            total_pixels = class_counts.sum()
            stats['class_distribution'] = {
                f"class_{i}": {
                    'count': int(class_counts[i]),
                    'percentage': float(class_counts[i] / total_pixels * 100)
                }
                for i in range(4)
            }

        return stats


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Download and preprocess BraTS dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--download", action="store_true",
                       help="Download dataset from Kaggle")
    parser.add_argument("--extract", action="store_true",
                       help="Extract downloaded dataset")
    parser.add_argument("--preprocess", action="store_true",
                       help="Preprocess dataset for training")
    parser.add_argument("--stats", action="store_true",
                       help="Compute and display dataset statistics")

    args = parser.parse_args()

    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize downloader
    downloader = BraTSDatasetDownloader(config)

    # Execute requested operations
    if args.download:
        success = downloader.download_dataset()
        if not success:
            return

    if args.extract:
        success = downloader.extract_dataset()
        if not success:
            return

    if args.preprocess:
        success = downloader.preprocess_data()
        if not success:
            return

    if args.stats:
        stats = downloader.get_dataset_stats()
        print("\n📊 Dataset Statistics:")
        print(f"Total Patients: {stats['total_patients']}")
        print(f"Image Shape: {stats['image_shape']}")
        print(f"Modalities: {stats['modalities']}")
        print("Class Distribution:")
        for cls, info in stats['class_distribution'].items():
            count = info.get('count')
            pct = info.get('percentage')
            if count is not None and pct is not None:
                print(f"  {cls}: {count} ({pct:.2f}%)")
            else:
                print(f"  {cls}: {info}")


if __name__ == "__main__":
    main()


