#!/usr/bin/env python3
"""
Process LGG MRI Segmentation Dataset for Federated Continual Learning
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This script processes the real LGG MRI segmentation dataset (TIF format)
into the format expected by our federated learning pipeline.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml


class LGGDatasetProcessor:
    """Process LGG MRI dataset for federated learning"""

    def __init__(self, config: dict):
        self.config = config
        self.raw_data_dir = Path(config['dataset']['data_dir']) / "lgg-mri-segmentation" / "kaggle_3m"
        self.processed_dir = Path(config['dataset']['processed_dir'])
        self.target_size = config['dataset']['target_size']

        # Create processed directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_dataset(self):
        """Process the entire LGG dataset"""
        print("🔄 Processing LGG MRI Segmentation Dataset...")

        # Load patient metadata
        metadata_file = self.raw_data_dir / "data.csv"
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file)
            print(f"✅ Loaded metadata for {len(metadata)} patients")
        else:
            print("⚠️  No metadata file found, proceeding without it")
            metadata = None

        # Get all patient directories
        patient_dirs = [d for d in self.raw_data_dir.iterdir() if d.is_dir()]
        patient_dirs = [d for d in patient_dirs if d.name.startswith('TCGA_')]

        print(f"📁 Found {len(patient_dirs)} patient directories")

        # Process each patient
        processed_patients = []
        for patient_dir in tqdm(patient_dirs[:50], desc="Processing patients"):  # Limit for demo
            patient_data = self._process_patient(patient_dir)
            if patient_data:
                processed_patients.append(patient_data)

        print(f"✅ Successfully processed {len(processed_patients)} patients")

        # Create federated splits
        self._create_federated_splits(processed_patients)

        # Save dataset statistics
        stats = self._compute_dataset_stats(processed_patients)
        with open(self.processed_dir / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def _process_patient(self, patient_dir: Path) -> dict:
        """Process a single patient"""
        try:
            patient_id = patient_dir.name

            # Get all TIF files for this patient
            tif_files = list(patient_dir.glob("*.tif"))
            image_files = [f for f in tif_files if not f.name.endswith('_mask.tif')]
            mask_files = [f for f in tif_files if f.name.endswith('_mask.tif')]

            if not image_files:
                return None

            # Group files by slice number
            slices_data = {}
            for img_file in image_files:
                # Extract slice number from filename
                # Format: TCGA_<institution>_<patient>_<slice>.tif
                parts = img_file.stem.split('_')
                slice_num = int(parts[-1])

                if slice_num not in slices_data:
                    slices_data[slice_num] = {'image': None, 'mask': None}

                slices_data[slice_num]['image'] = img_file

            # Match masks to slices
            for mask_file in mask_files:
                parts = mask_file.stem.split('_')
                slice_num = int(parts[-2])  # Remove '_mask' suffix

                if slice_num in slices_data:
                    slices_data[slice_num]['mask'] = mask_file

            # Process slices (take middle slice for simplicity)
            if not slices_data:
                return None

            # Select a representative slice (middle one)
            slice_nums = sorted(slices_data.keys())
            middle_slice = slice_nums[len(slice_nums) // 2]

            slice_data = slices_data[middle_slice]

            if slice_data['image'] is None or slice_data['mask'] is None:
                return None

            # Load and process image
            image = self._load_tif_image(slice_data['image'])
            mask = self._load_tif_mask(slice_data['mask'])

            if image is None or mask is None:
                return None

            # Convert to our expected format
            processed_data = self._convert_to_standard_format(image, mask)

            return {
                'patient_id': patient_id,
                'original_slice': middle_slice,
                'images': processed_data['images'],
                'mask': processed_data['mask'],
                'metadata': {
                    'institution': patient_id.split('_')[1],
                    'total_slices': len(slices_data)
                }
            }

        except Exception as e:
            print(f"❌ Error processing {patient_dir.name}: {e}")
            return None

    def _load_tif_image(self, image_path: Path) -> np.ndarray:
        """Load TIF image (3 channels: pre-contrast, FLAIR, post-contrast)"""
        try:
            img = Image.open(image_path)

            # Convert to numpy array
            if img.mode == 'RGB':
                img_array = np.array(img)
            else:
                # Convert to RGB if needed
                img = img.convert('RGB')
                img_array = np.array(img)

            # Normalize to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0

            # Transpose to [C, H, W] format
            img_array = np.transpose(img_array, (2, 0, 1))

            return img_array

        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None

    def _load_tif_mask(self, mask_path: Path) -> np.ndarray:
        """Load TIF mask (binary segmentation)"""
        try:
            mask = Image.open(mask_path)

            # Convert to numpy array
            mask_array = np.array(mask)

            # Ensure binary mask
            mask_array = (mask_array > 0).astype(np.int32)

            # If multi-channel, take first channel
            if mask_array.ndim > 2:
                mask_array = mask_array[:, :, 0]

            return mask_array

        except Exception as e:
            print(f"❌ Error loading mask {mask_path}: {e}")
            return None

    def _convert_to_standard_format(self, image_3ch: np.ndarray, mask_2d: np.ndarray) -> dict:
        """Convert 3-channel TIF format to our 4-modality NIfTI-like format"""

        # Image is [3, H, W] - pre-contrast, FLAIR, post-contrast
        # We need to create 4 modalities: t1, t1ce, t2, flair

        h, w = image_3ch.shape[1], image_3ch.shape[2]

        # Create synthetic modalities based on available data
        modalities = {}

        # FLAIR is available (channel 1)
        modalities['flair'] = image_3ch[1].copy()  # FLAIR

        # Pre-contrast as T1 (channel 0)
        modalities['t1'] = image_3ch[0].copy()  # Pre-contrast

        # Post-contrast as T1CE (channel 2)
        modalities['t1ce'] = image_3ch[2].copy()  # Post-contrast

        # Create synthetic T2 from FLAIR (approximation)
        modalities['t2'] = np.clip(image_3ch[1] * 0.8 + np.random.normal(0, 0.05, (h, w)), 0, 1)

        # Resize to target size if needed
        target_h, target_w, target_d = self.target_size

        for modality in modalities:
            # Resize to target size (2D to 3D by adding depth dimension)
            img_resized = self._resize_image(modalities[modality], (target_h, target_w))
            # Add depth dimension (single slice becomes 3D volume)
            modalities[modality] = np.expand_dims(img_resized, axis=-1).repeat(target_d, axis=-1)

        # Process mask
        mask_resized = self._resize_mask(mask_2d, (target_h, target_w))
        mask_3d = np.expand_dims(mask_resized, axis=-1).repeat(target_d, axis=-1)

        return {
            'images': modalities,
            'mask': mask_3d
        }

    def _resize_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize 2D image to target size"""
        from PIL import Image as PILImage

        # Convert to PIL Image for resizing
        img_pil = PILImage.fromarray((image * 255).astype(np.uint8))
        img_resized = img_pil.resize(target_size, PILImage.BILINEAR)
        img_array = np.array(img_resized).astype(np.float32) / 255.0

        return img_array

    def _resize_mask(self, mask: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize 2D mask to target size"""
        from PIL import Image as PILImage

        # Convert to PIL Image for resizing
        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
        mask_resized = mask_pil.resize(target_size, PILImage.NEAREST)
        mask_array = (np.array(mask_resized) > 127).astype(np.int32)

        return mask_array

    def _create_federated_splits(self, processed_patients: list):
        """Create federated data splits"""
        print("\\n🏥 Creating federated splits...")

        # Split patients into 4 virtual hospitals
        patient_ids = [p['patient_id'] for p in processed_patients]

        # Create stratified split based on institution
        institutions = [p['metadata']['institution'] for p in processed_patients]

        # Simple split for demo (can be improved with stratification)
        splits = train_test_split(
            patient_ids,
            test_size=0.25,
            random_state=42,
            shuffle=True
        )

        hospital_names = ['hospital_a', 'hospital_b', 'hospital_c', 'hospital_d']
        hospital_splits = [splits[0][:len(splits[0])//3],
                          splits[0][len(splits[0])//3:2*len(splits[0])//3],
                          splits[0][2*len(splits[0])//3:],
                          splits[1]]

        for hospital_name, split in zip(hospital_names, hospital_splits):
            hospital_dir = self.processed_dir / hospital_name
            hospital_dir.mkdir(exist_ok=True)

            # Save patient list
            with open(hospital_dir / "patients.txt", 'w') as f:
                for patient_id in split:
                    f.write(f"{patient_id}\\n")

            # Copy patient data to hospital directory
            for patient_id in split:
                patient_data = next((p for p in processed_patients if p['patient_id'] == patient_id), None)
                if patient_data:
                    # Create patient directory
                    patient_dir = hospital_dir / patient_id
                    patient_dir.mkdir(exist_ok=True)

                    # Save processed data
                    for modality, img in patient_data['images'].items():
                        np.save(patient_dir / f"{modality}.npy", img)

                    np.save(patient_dir / "mask.npy", patient_data['mask'])

        print(f"✅ Created {len(hospital_names)} hospital splits")

    def _compute_dataset_stats(self, processed_patients: list) -> dict:
        """Compute dataset statistics"""
        print("\\n📊 Computing dataset statistics...")

        stats = {
            'total_patients': len(processed_patients),
            'modalities': ['t1', 't1ce', 't2', 'flair'],
            'image_shape': self.target_size,
            'classes': [0, 1],  # Background, tumor (simplified)
            'institutions': {},
            'slices_per_patient': []
        }

        # Count patients per institution
        for patient in processed_patients:
            inst = patient['metadata']['institution']
            stats['institutions'][inst] = stats['institutions'].get(inst, 0) + 1
            stats['slices_per_patient'].append(patient['metadata']['total_slices'])

        # Class distribution (approximate)
        total_pixels = 0
        tumor_pixels = 0

        for patient in processed_patients[:10]:  # Sample for stats
            mask = patient['mask']
            total_pixels += mask.size
            tumor_pixels += (mask > 0).sum()

        stats['class_distribution'] = {
            'background': int(total_pixels - tumor_pixels),
            'tumor': int(tumor_pixels)
        }

        return stats


def main():
    """Main processing function"""
    # Load configuration
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Process dataset
    processor = LGGDatasetProcessor(config)
    stats = processor.process_dataset()

    print("\\n📊 Dataset Statistics:")
    print(f"Total Patients: {stats['total_patients']}")
    print(f"Institutions: {len(stats['institutions'])}")
    print(f"Image Shape: {stats['image_shape']}")
    print(f"Modalities: {stats['modalities']}")
    print(f"Class Distribution: {stats['class_distribution']}")

    print("\\n✅ Real LGG dataset processed successfully!")


if __name__ == "__main__":
    main()


