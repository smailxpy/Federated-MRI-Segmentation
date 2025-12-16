#!/usr/bin/env python3
"""
Process BraTS2021 Dataset for Federated Continual Learning
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This script processes the real BraTS2021 dataset (NIfTI format) with proper
3D volumetric data and 4-class tumor segmentation for federated learning.
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class BraTS2021Processor:
    """Process BraTS2021 MRI dataset for federated learning"""

    # BraTS2021 tumor classes
    CLASSES = {
        0: "Background",
        1: "NCR/NET (Necrotic/Non-Enhancing Tumor)",  # Label 1
        2: "ED (Peritumoral Edema)",                   # Label 2
        4: "ET (GD-Enhancing Tumor)"                   # Label 4 (note: no label 3)
    }
    
    # Modalities in BraTS2021
    MODALITIES = ["t1", "t1ce", "t2", "flair"]

    def __init__(self, config: dict):
        self.config = config
        self.raw_data_dir = Path(config['dataset']['data_dir']) / "brats2021"
        self.processed_dir = Path(config['dataset']['processed_dir'])
        self.target_size = tuple(config['dataset']['target_size'])
        self.num_classes = config['dataset']['num_classes']
        
        # Create processed directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 BraTS2021 Processor initialized")
        print(f"   Raw data: {self.raw_data_dir}")
        print(f"   Output: {self.processed_dir}")
        print(f"   Target size: {self.target_size}")

    def process_dataset(self, max_patients: Optional[int] = None) -> dict:
        """Process the entire BraTS2021 dataset"""
        print("\n🔄 Processing BraTS2021 Dataset...")

        # Get all patient directories
        patient_dirs = sorted([
            d for d in self.raw_data_dir.iterdir() 
            if d.is_dir() and d.name.startswith('BraTS2021_')
        ])
        
        total_patients = len(patient_dirs)
        print(f"📁 Found {total_patients} patient directories")
        
        if max_patients:
            patient_dirs = patient_dirs[:max_patients]
            print(f"   Processing first {max_patients} patients")

        # Process each patient
        processed_patients = []
        failed_patients = []
        
        for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
            try:
                patient_data = self._process_patient(patient_dir)
                if patient_data:
                    processed_patients.append(patient_data)
            except Exception as e:
                failed_patients.append((patient_dir.name, str(e)))
                
        print(f"\n✅ Successfully processed {len(processed_patients)} patients")
        if failed_patients:
            print(f"⚠️  Failed to process {len(failed_patients)} patients")

        # Create federated splits (4 hospitals)
        self._create_federated_splits(processed_patients)

        # Save dataset statistics
        stats = self._compute_dataset_stats(processed_patients)
        stats_path = self.processed_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"📊 Statistics saved to {stats_path}")

        return stats

    def _process_patient(self, patient_dir: Path) -> Optional[dict]:
        """Process a single patient's MRI data"""
        patient_id = patient_dir.name
        
        # Load all modalities
        volumes = {}
        for modality in self.MODALITIES:
            nifti_path = patient_dir / f"{patient_id}_{modality}.nii.gz"
            if not nifti_path.exists():
                return None
            
            nifti_img = nib.load(nifti_path)
            volume = nifti_img.get_fdata().astype(np.float32)
            volumes[modality] = volume
            
        # Load segmentation mask
        seg_path = patient_dir / f"{patient_id}_seg.nii.gz"
        if not seg_path.exists():
            return None
            
        seg_img = nib.load(seg_path)
        mask = seg_img.get_fdata().astype(np.int32)
        
        # Get original shape
        original_shape = volumes['t1'].shape
        
        # Normalize volumes
        for modality in self.MODALITIES:
            volumes[modality] = self._normalize_volume(volumes[modality])
        
        # Resize to target size if needed
        if original_shape != self.target_size:
            for modality in self.MODALITIES:
                volumes[modality] = self._resize_volume(volumes[modality], self.target_size)
            mask = self._resize_mask(mask, self.target_size)
        
        # Remap labels: BraTS uses 0,1,2,4 -> we use 0,1,2,3
        mask = self._remap_labels(mask)
        
        # Save processed data
        output_dir = self.processed_dir / patient_id
        output_dir.mkdir(exist_ok=True)
        
        for modality in self.MODALITIES:
            np.save(output_dir / f"{modality}.npy", volumes[modality])
        np.save(output_dir / "mask.npy", mask)
        
        # Calculate tumor statistics for this patient
        tumor_stats = self._compute_tumor_stats(mask)
        
        return {
            'patient_id': patient_id,
            'original_shape': original_shape,
            'target_shape': self.target_size,
            'tumor_stats': tumor_stats,
            'output_dir': str(output_dir)
        }

    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume using z-score normalization on non-zero voxels"""
        # Mask for brain region (non-zero voxels)
        brain_mask = volume > 0
        
        if brain_mask.sum() == 0:
            return volume
        
        # Z-score normalization on brain region
        brain_voxels = volume[brain_mask]
        mean = brain_voxels.mean()
        std = brain_voxels.std()
        
        if std > 0:
            volume[brain_mask] = (volume[brain_mask] - mean) / std
        
        return volume

    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize 3D volume using trilinear interpolation"""
        from scipy.ndimage import zoom
        
        factors = [t / s for t, s in zip(target_size, volume.shape)]
        return zoom(volume, factors, order=1)  # Linear interpolation

    def _resize_mask(self, mask: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize segmentation mask using nearest-neighbor interpolation"""
        from scipy.ndimage import zoom
        
        factors = [t / s for t, s in zip(target_size, mask.shape)]
        return zoom(mask, factors, order=0).astype(np.int32)  # Nearest neighbor

    def _remap_labels(self, mask: np.ndarray) -> np.ndarray:
        """Remap BraTS labels (0,1,2,4) to sequential (0,1,2,3)"""
        remapped = np.zeros_like(mask)
        remapped[mask == 0] = 0  # Background
        remapped[mask == 1] = 1  # NCR/NET
        remapped[mask == 2] = 2  # ED
        remapped[mask == 4] = 3  # ET (label 4 -> 3)
        return remapped

    def _compute_tumor_stats(self, mask: np.ndarray) -> dict:
        """Compute tumor region statistics"""
        unique, counts = np.unique(mask, return_counts=True)
        total_voxels = mask.size
        
        stats = {
            'total_voxels': int(total_voxels),
            'class_counts': {},
            'class_percentages': {}
        }
        
        class_names = ['background', 'ncr_net', 'ed', 'et']
        for i in range(4):
            count = int(counts[unique == i].sum()) if i in unique else 0
            stats['class_counts'][class_names[i]] = count
            stats['class_percentages'][class_names[i]] = round(100 * count / total_voxels, 4)
        
        return stats

    def _create_federated_splits(self, processed_patients: List[dict]):
        """Create federated data splits for 4 virtual hospitals"""
        print("\n🏥 Creating federated splits (4 hospitals)...")
        
        patient_ids = [p['patient_id'] for p in processed_patients]
        
        # Shuffle and split into 4 equal parts
        np.random.seed(42)
        np.random.shuffle(patient_ids)
        
        # Split into 4 hospitals (25% each)
        n = len(patient_ids)
        splits = {
            'hospital_a': patient_ids[0:n//4],
            'hospital_b': patient_ids[n//4:n//2],
            'hospital_c': patient_ids[n//2:3*n//4],
            'hospital_d': patient_ids[3*n//4:]
        }
        
        for hospital_name, patients in splits.items():
            hospital_dir = self.processed_dir / hospital_name
            hospital_dir.mkdir(exist_ok=True)
            
            # Save patient list
            with open(hospital_dir / "patients.txt", 'w') as f:
                for patient_id in patients:
                    f.write(f"{patient_id}\n")
            
            # Create symlinks to patient data (to avoid duplicating data)
            for patient_id in patients:
                src_dir = self.processed_dir / patient_id
                dst_dir = hospital_dir / patient_id
                
                if src_dir.exists() and not dst_dir.exists():
                    # Create hard copies for portability (symlinks can cause issues)
                    dst_dir.mkdir(exist_ok=True)
                    for file in src_dir.glob("*.npy"):
                        np.save(dst_dir / file.name, np.load(file))
            
            print(f"   {hospital_name}: {len(patients)} patients")
        
        print(f"✅ Created 4 hospital splits")

    def _compute_dataset_stats(self, processed_patients: List[dict]) -> dict:
        """Compute comprehensive dataset statistics"""
        print("\n📊 Computing dataset statistics...")
        
        stats = {
            'dataset_name': 'BraTS2021',
            'total_patients': len(processed_patients),
            'modalities': self.MODALITIES,
            'target_shape': list(self.target_size),
            'num_classes': self.num_classes,
            'class_names': ['Background', 'NCR/NET', 'ED', 'ET'],
            'class_distribution': {
                'background': 0,
                'ncr_net': 0,
                'ed': 0,
                'et': 0
            },
            'hospital_splits': {
                'hospital_a': 0,
                'hospital_b': 0,
                'hospital_c': 0,
                'hospital_d': 0
            }
        }
        
        # Aggregate tumor statistics
        for patient in processed_patients:
            tumor_stats = patient.get('tumor_stats', {})
            for class_name, count in tumor_stats.get('class_counts', {}).items():
                if class_name in stats['class_distribution']:
                    stats['class_distribution'][class_name] += count
        
        # Count patients per hospital
        for hospital in ['hospital_a', 'hospital_b', 'hospital_c', 'hospital_d']:
            hospital_dir = self.processed_dir / hospital
            if hospital_dir.exists():
                patients_file = hospital_dir / "patients.txt"
                if patients_file.exists():
                    with open(patients_file) as f:
                        stats['hospital_splits'][hospital] = len(f.readlines())
        
        return stats


def main():
    """Main processing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process BraTS2021 dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--max-patients", type=int, default=None,
                       help="Maximum number of patients to process (for testing)")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only compute and display statistics")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create processor
    processor = BraTS2021Processor(config)
    
    if args.stats_only:
        # Just load existing stats
        stats_path = processor.processed_dir / "dataset_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
        else:
            print("❌ No statistics file found. Run processing first.")
            return
    else:
        # Process dataset
        stats = processor.process_dataset(max_patients=args.max_patients)
    
    # Display statistics
    print("\n" + "="*60)
    print("📊 BraTS2021 Dataset Statistics")
    print("="*60)
    print(f"Total Patients: {stats['total_patients']}")
    print(f"Modalities: {', '.join(stats['modalities'])}")
    print(f"Volume Shape: {stats['target_shape']}")
    print(f"Number of Classes: {stats['num_classes']}")
    print(f"\nClass Names:")
    for i, name in enumerate(stats['class_names']):
        print(f"  {i}: {name}")
    print(f"\nHospital Splits:")
    for hospital, count in stats['hospital_splits'].items():
        print(f"  {hospital}: {count} patients")
    print(f"\nClass Distribution (voxels):")
    for class_name, count in stats['class_distribution'].items():
        print(f"  {class_name}: {count:,}")
    print("="*60)
    print("\n✅ BraTS2021 dataset processing complete!")


if __name__ == "__main__":
    main()



