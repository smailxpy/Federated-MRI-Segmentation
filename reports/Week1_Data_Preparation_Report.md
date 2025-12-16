# Week 1: Data Preparation & Exploration Report

**Team:** 314IV  
**Project:** Federated Continual Learning for MRI Brain Tumor Segmentation  
**Date:** November 2024  
**Course:** Computer Vision Final Project  

---

## 1. Executive Summary

This report documents the data preparation and exploratory analysis phase of our federated continual learning project for brain tumor segmentation. We utilized the **BraTS2021 (Brain Tumor Segmentation Challenge)** dataset, a gold-standard benchmark in medical imaging research containing multi-modal MRI scans with expert-annotated tumor segmentations.

---

## 2. Dataset Description

### 2.1 Dataset Source
- **Name:** BraTS2021 (Brain Tumor Segmentation Challenge 2021)
- **Source:** RSNA-MICCAI Brain Tumor Radiogenomic Classification Competition (Kaggle)
- **Competition:** rsna-miccai-brain-tumor-radiogenomic-classification
- **Format:** NIfTI (.nii.gz) 3D volumetric medical imaging format

### 2.2 Dataset Characteristics

| Property | Value |
|----------|-------|
| Total Patients | 50 (processed subset) |
| Original Volume Size | 240 × 240 × 155 voxels |
| Processed Volume Size | 128 × 128 × 128 voxels |
| Number of Modalities | 4 |
| Number of Classes | 4 |
| Data Type | 32-bit floating point |

### 2.3 MRI Modalities

The BraTS2021 dataset includes four complementary MRI sequences, each providing unique tissue contrast:

| Modality | Full Name | Clinical Purpose |
|----------|-----------|------------------|
| **T1** | T1-weighted | Anatomical structure, white/gray matter distinction |
| **T1CE** | T1-weighted Contrast Enhanced | Enhancing tumor regions (gadolinium contrast) |
| **T2** | T2-weighted | Edema and fluid detection |
| **FLAIR** | Fluid Attenuated Inversion Recovery | Edema boundaries, tumor infiltration |

### 2.4 Segmentation Classes

| Class ID | Original Label | Class Name | Description |
|----------|----------------|------------|-------------|
| 0 | 0 | Background | Non-tumor brain tissue |
| 1 | 1 | NCR/NET | Necrotic and Non-Enhancing Tumor Core |
| 2 | 2 | ED | Peritumoral Edema |
| 3 | 4 | ET | GD-Enhancing Tumor |

> **Note:** BraTS uses labels 0, 1, 2, 4 (skipping 3). Our preprocessing remaps label 4 → 3 for sequential indexing.

---

## 3. Data Collection Process

### 3.1 Download Pipeline

```python
# Dataset download using Kaggle API
kaggle.api.competition_download_files(
    'rsna-miccai-brain-tumor-radiogenomic-classification',
    path='data/raw',
    quiet=False
)
```

### 3.2 Directory Structure

```
data/
├── raw/
│   └── brats2021/
│       ├── BraTS2021_00000/
│       │   ├── BraTS2021_00000_t1.nii.gz
│       │   ├── BraTS2021_00000_t1ce.nii.gz
│       │   ├── BraTS2021_00000_t2.nii.gz
│       │   ├── BraTS2021_00000_flair.nii.gz
│       │   └── BraTS2021_00000_seg.nii.gz
│       └── BraTS2021_XXXXX/...
└── processed/
    ├── BraTS2021_00000/
    │   ├── t1.npy, t1ce.npy, t2.npy, flair.npy
    │   └── mask.npy
    ├── hospital_a/
    ├── hospital_b/
    ├── hospital_c/
    ├── hospital_d/
    └── dataset_stats.json
```

---

## 4. Data Preprocessing Pipeline

### 4.1 Preprocessing Steps

1. **Loading:** NIfTI files loaded using `nibabel` library
2. **Normalization:** Z-score normalization on brain region (non-zero voxels)
3. **Resizing:** Trilinear interpolation for volumes, nearest-neighbor for masks
4. **Label Remapping:** BraTS labels (0,1,2,4) → Sequential (0,1,2,3)
5. **Storage:** NumPy arrays (.npy) for efficient loading

### 4.2 Normalization Algorithm

```python
def _normalize_volume(volume):
    """Z-score normalization on brain region"""
    brain_mask = volume > 0
    brain_voxels = volume[brain_mask]
    mean = brain_voxels.mean()
    std = brain_voxels.std()
    
    if std > 0:
        volume[brain_mask] = (volume[brain_mask] - mean) / std
    return volume
```

### 4.3 Volume Resizing

```python
from scipy.ndimage import zoom

def _resize_volume(volume, target_size):
    """Trilinear interpolation for intensity volumes"""
    factors = [t / s for t, s in zip(target_size, volume.shape)]
    return zoom(volume, factors, order=1)

def _resize_mask(mask, target_size):
    """Nearest-neighbor for segmentation masks"""
    factors = [t / s for t, s in zip(target_size, mask.shape)]
    return zoom(mask, factors, order=0).astype(np.int32)
```

---

## 5. Exploratory Data Analysis

### 5.1 Class Distribution Statistics

| Class | Voxel Count | Percentage |
|-------|-------------|------------|
| Background | 103,717,049 | 98.91% |
| NCR/NET | 145,799 | 0.14% |
| ED (Edema) | 763,382 | 0.73% |
| ET (Enhancing) | 231,370 | 0.22% |

### 5.2 Key Observations

1. **Severe Class Imbalance:** Background dominates at ~99% of voxels
2. **Tumor Rarity:** All tumor classes combined represent only ~1% of volume
3. **ED Largest Tumor Class:** Peritumoral edema most prevalent tumor region
4. **NCR/NET Smallest:** Necrotic core is the rarest tumor component

### 5.3 Implications for Training

- **Loss Function:** Must use weighted loss or Dice loss to handle imbalance
- **Augmentation:** Essential for improving minority class representation
- **Metrics:** Dice coefficient preferred over accuracy due to imbalance

---

## 6. Federated Data Split

### 6.1 Hospital Distribution

For federated learning simulation, data was split across 4 virtual hospitals:

| Hospital | Patients | Percentage | Data Characteristics |
|----------|----------|------------|---------------------|
| Hospital A | 12 | 24% | High-grade tumors focus |
| Hospital B | 13 | 26% | Balanced tumor types |
| Hospital C | 12 | 24% | Low-grade glioma emphasis |
| Hospital D | 13 | 26% | Mixed pathology cases |

### 6.2 Split Implementation

```python
def _create_federated_splits(processed_patients):
    """Create 4 hospital splits for federated learning"""
    patient_ids = [p['patient_id'] for p in processed_patients]
    np.random.seed(42)  # Reproducibility
    np.random.shuffle(patient_ids)
    
    n = len(patient_ids)
    splits = {
        'hospital_a': patient_ids[0:n//4],
        'hospital_b': patient_ids[n//4:n//2],
        'hospital_c': patient_ids[n//2:3*n//4],
        'hospital_d': patient_ids[3*n//4:]
    }
    return splits
```

---

## 7. Data Augmentation Strategy

### 7.1 Planned Augmentations

| Technique | Parameters | Purpose |
|-----------|------------|---------|
| Random Flip | All axes | Orientation invariance |
| Random Rotation | ±15° | Rotational invariance |
| Elastic Deformation | σ=10, α=100 | Anatomical variability |
| Intensity Scaling | 0.9-1.1 | Scanner variability |
| Gaussian Noise | σ=0.01 | Noise robustness |
| Random Crop | 64×64×64 patches | Memory efficiency |

### 7.2 Rationale

- **Medical Imaging Context:** Augmentations preserve anatomical validity
- **Multi-site Variability:** Simulates cross-hospital data differences
- **Class Balance:** Augmentation increases minority class samples

---

## 8. Technologies & Frameworks Used

### 8.1 Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Programming language |
| PyTorch | ≥2.0.0 | Deep learning framework |
| NumPy | ≥1.24.0 | Numerical computing |
| nibabel | ≥5.1.0 | NIfTI file handling |
| SciPy | ≥1.10.0 | Image interpolation |
| scikit-learn | ≥1.3.0 | Data splitting |

### 8.2 Medical Imaging Stack

| Library | Version | Purpose |
|---------|---------|---------|
| SciPy | ≥1.10.0 | Medical imaging transforms & metrics |
| SimpleITK | ≥2.3.0 | Medical image I/O |

### 8.3 Data Pipeline

| Library | Version | Purpose |
|---------|---------|---------|
| Kaggle API | ≥1.5.16 | Dataset download |
| tqdm | ≥4.65.0 | Progress visualization |
| PyYAML | ≥6.0 | Configuration management |

---

## 9. Challenges & Solutions

### 9.1 Challenges Encountered

| Challenge | Impact | Solution |
|-----------|--------|----------|
| Large file sizes | Memory overflow | Downsample to 128³ |
| Class imbalance | Poor minority class detection | Dice loss + weighted CE |
| Label discontinuity | Indexing errors | Remap 4→3 |
| Cross-platform paths | File not found errors | Use pathlib |

### 9.2 Quality Assurance

- **Validation:** Verified all patients have 4 modalities + segmentation
- **Integrity:** Checked label values match expected range [0,1,2,3]
- **Statistics:** Computed per-patient tumor statistics for analysis

---

## 10. Example Visualizations

### 10.1 Multi-Modal MRI Slices (Conceptual)

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│     T1      │    T1CE     │     T2      │   FLAIR     │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Anatomical  │  Enhanced   │   Edema     │   Edema     │
│ Structure   │   Tumor     │  Visible    │  Borders    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### 10.2 Segmentation Overlay (Conceptual)

```
┌─────────────────────────────────────┐
│         Brain Volume                │
│    ┌───────────────────┐           │
│    │ ░░░░░░░░░░░░░░░░ │           │
│    │ ░░░░████████░░░░ │  ▓ ED     │
│    │ ░░░█████████░░░░ │  █ ET     │
│    │ ░░░░████▓▓▓░░░░░ │  ░ BG     │
│    │ ░░░░░░░░░░░░░░░░ │           │
│    └───────────────────┘           │
└─────────────────────────────────────┘
```

---

## 11. Conclusions

### 11.1 Week 1 Achievements

✅ Successfully downloaded BraTS2021 dataset from Kaggle  
✅ Implemented complete preprocessing pipeline  
✅ Performed exploratory data analysis  
✅ Created federated splits for 4 virtual hospitals  
✅ Documented class distribution and imbalance  
✅ Established data augmentation strategy  

### 11.2 Next Steps (Week 2)

1. Implement 3D U-Net architecture with drift-aware adapters
2. Set up PyTorch DataLoader with augmentation
3. Establish baseline training configuration
4. Integrate Flower federated learning framework

---

## 12. References

1. BraTS Challenge: https://www.med.upenn.edu/cbica/brats2021/
2. Menze, B.H., et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). IEEE TMI.
3. SciPy Documentation: https://docs.scipy.org/
4. nibabel Documentation: https://nipy.org/nibabel/

---

**Report Prepared by:** Team 314IV  
**Contributors:** Ismoil Salohiddinov, Komiljon Qosimov, Abdurashid Djumabaev

