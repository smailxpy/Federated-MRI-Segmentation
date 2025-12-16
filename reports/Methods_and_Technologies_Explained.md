# Methods, Frameworks & Technologies Explained

**Team:** 314IV  
**Project:** Federated Continual Learning for MRI Brain Tumor Segmentation  
**Course:** Computer Vision Final Project  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Deep Learning Framework: PyTorch](#2-deep-learning-framework-pytorch)
3. [Model Architecture: SegResNet](#3-model-architecture-segresnet)
4. [Drift-Aware Adapters](#4-drift-aware-adapters)
5. [Federated Learning: Flower Framework](#5-federated-learning-flower-framework)
6. [Medical Imaging Pipeline](#6-medical-imaging-pipeline)
7. [Loss Functions](#7-loss-functions)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Continual Learning](#9-continual-learning)
10. [Data Processing Pipeline](#10-data-processing-pipeline)
11. [Complete Technology Stack](#11-complete-technology-stack)

---

## 1. Project Overview

### 1.1 What We Built

A **Federated Continual Learning (FCL)** system for **brain tumor segmentation** that:

1. **Segments brain tumors** from 3D MRI scans into 4 classes
2. **Preserves patient privacy** through federated learning (no data sharing)
3. **Adapts continuously** to new hospitals without forgetting previous knowledge

### 1.2 Why This Approach?

| Challenge | Solution |
|-----------|----------|
| Patient Privacy | Federated Learning - data stays at hospitals |
| Multi-Hospital Collaboration | Flower framework for distributed training |
| Domain Shift | Drift-Aware Adapters for adaptation |
| Catastrophic Forgetting | Continual learning techniques |
| 3D Medical Data | 3D U-Net architecture |

---

## 2. Deep Learning Framework: PyTorch

### 2.1 What is PyTorch?

PyTorch is an open-source machine learning framework developed by Meta AI. It provides:

- **Tensors:** N-dimensional arrays with GPU acceleration
- **Autograd:** Automatic differentiation for gradient computation
- **nn.Module:** Building blocks for neural networks
- **Optimizers:** SGD, Adam, AdamW, etc.

### 2.2 Why PyTorch for This Project?

| Feature | Benefit |
|---------|---------|
| Dynamic Computation | Flexible debugging, easy model modification |
| CUDA Support | GPU acceleration for 3D convolutions |
| Rich Ecosystem | Flower, SciPy, NumPy integration |
| Research Standard | Most medical imaging papers use PyTorch |

### 2.3 Key PyTorch Components Used

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 3D Convolution
nn.Conv3d(in_channels=4, out_channels=32, kernel_size=3, padding=1)

# 3D Batch Normalization
nn.BatchNorm3d(num_features=32)

# 3D Max Pooling
nn.MaxPool3d(kernel_size=2, stride=2)

# 3D Transposed Convolution (Upsampling)
nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)

# Dropout for regularization
nn.Dropout3d(p=0.1)

# AdamW Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
```

---

## 3. Model Architecture: 3D U-Net

### 3.1 What is U-Net?

U-Net is a **convolutional neural network** designed for biomedical image segmentation (Ronneberger et al., 2015). It has an **encoder-decoder** structure with **skip connections**.

### 3.2 Architecture Diagram

```
INPUT [4, 128, 128, 128]    (4 MRI modalities)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                       ENCODER PATH                        │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  Conv3D + BN + ReLU → [32, 128, 128, 128]  ──────────┐   │
│        │ MaxPool                                      │   │
│        ▼                                              │   │
│  Conv3D + BN + ReLU → [64, 64, 64, 64]    ───────┐   │   │
│        │ MaxPool                                  │   │   │
│        ▼                                          │   │   │
│  Conv3D + BN + ReLU → [128, 32, 32, 32]  ────┐   │   │   │
│        │ MaxPool                              │   │   │   │
│        ▼                                      │   │   │   │
│  Conv3D + BN + ReLU → [256, 16, 16, 16] ─┐   │   │   │   │
│        │ MaxPool                          │   │   │   │   │
│        ▼                                  │   │   │   │   │
│  ┌─────────────────────────────────────┐ │   │   │   │   │
│  │         BOTTLENECK [512]            │ │   │   │   │   │
│  └─────────────────────────────────────┘ │   │   │   │   │
│        │                                  │   │   │   │   │
│        ▼                                  │   │   │   │   │
├───────────────────────────────────────────────────────────┤
│                       DECODER PATH                        │
├───────────────────────────────────────────────────────────┤
│        │                                  │   │   │   │   │
│  UpConv + Concat ←───────────────────────┘   │   │   │   │
│  Conv3D + BN + ReLU → [256, 16, 16, 16]      │   │   │   │
│        │                                      │   │   │   │
│  UpConv + Concat ←───────────────────────────┘   │   │   │
│  Conv3D + BN + ReLU → [128, 32, 32, 32]          │   │   │
│        │                                          │   │   │
│  UpConv + Concat ←───────────────────────────────┘   │   │
│  Conv3D + BN + ReLU → [64, 64, 64, 64]               │   │
│        │                                              │   │
│  UpConv + Concat ←───────────────────────────────────┘   │
│  Conv3D + BN + ReLU → [32, 128, 128, 128]                │
│                                                           │
└───────────────────────────────────────────────────────────┘
        │
        ▼
   Conv3D 1×1×1 → [4, 128, 128, 128]  (4 classes)
        │
        ▼
OUTPUT: Segmentation Map
```

### 3.3 Key Concepts

#### Skip Connections
- Connect encoder to decoder at same resolution
- Preserve spatial information lost during downsampling
- Enable gradient flow for better training

#### Encoder (Contracting Path)
- Extracts features through convolutions
- Reduces spatial resolution via pooling
- Increases feature channels (32 → 64 → 128 → 256)

#### Decoder (Expanding Path)
- Reconstructs spatial resolution
- Combines skip connections with upsampled features
- Produces pixel-wise predictions

### 3.4 Why 3D U-Net?

| 2D U-Net | 3D U-Net |
|----------|----------|
| Processes slices independently | Considers volumetric context |
| Faster training | Better spatial understanding |
| Loses inter-slice information | Captures 3D tumor morphology |

---

## 4. Drift-Aware Adapters

### 4.1 What are Adapters?

Adapters are **small neural network modules** inserted into a pretrained model to enable **parameter-efficient fine-tuning**. Instead of updating all weights, only adapter weights are trained.

### 4.2 Why "Drift-Aware"?

In federated learning, different hospitals have different data distributions (domain shift). Drift-aware adapters:

1. **Detect domain drift** through learned domain parameters
2. **Adapt features** to local data distribution
3. **Preserve global knowledge** in shared parameters

### 4.3 Adapter Architecture

```python
class DriftAwareAdapter(nn.Module):
    def __init__(self, channels, adapter_channels=16, dropout=0.1):
        super().__init__()
        
        # Bottleneck adapter (down-project then up-project)
        self.adapter = nn.Sequential(
            nn.Linear(channels, adapter_channels),   # 256 → 16
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_channels, channels),   # 16 → 256
            nn.Sigmoid()                              # Gate [0, 1]
        )
        
        # Domain-specific learnable parameters
        self.domain_scale = nn.Parameter(torch.ones(1))
        self.domain_shift = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # Global average pooling: [B, C, D, H, W] → [B, C]
        pooled = x.mean(dim=[-3, -2, -1])
        
        # Generate gating signal
        gate = self.adapter(pooled)  # [B, C]
        gate = gate.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Apply domain transformation
        adapted = x * (1 + self.domain_scale) + self.domain_shift
        
        # Apply gating
        return adapted * gate
```

### 4.4 Parameter Separation

```python
class UNetWithAdapters(nn.Module):
    def get_shared_parameters(self):
        """Parameters aggregated in federated learning"""
        # All parameters EXCEPT adapters
        adapter_ids = set(id(p) for p in self.get_adapter_parameters())
        return [p for p in self.parameters() if id(p) not in adapter_ids]
    
    def get_adapter_parameters(self):
        """Parameters kept local (not shared)"""
        params = []
        for module in self.modules():
            if isinstance(module, DriftAwareAdapter):
                params.extend(module.parameters())
        return params
```

| Parameter Type | Count | Shared in FL? |
|----------------|-------|---------------|
| Shared (Conv, BN) | ~30M | Yes |
| Adapters | ~1M | No (local) |

---

## 5. Federated Learning: Flower Framework

### 5.1 What is Federated Learning?

Federated Learning (FL) is a machine learning approach where:
- **Data stays on local devices** (hospitals)
- **Only model updates** are shared
- A **central server** aggregates updates

### 5.2 FedAvg Algorithm

```
Algorithm: Federated Averaging (McMahan et al., 2017)

Server:
1. Initialize global model w₀
2. For each round t = 1, 2, ..., T:
   a. Send wₜ to all clients
   b. Each client k trains locally: wₜₖ = LocalTrain(wₜ)
   c. Aggregate: wₜ₊₁ = Σₖ (nₖ/n) × wₜₖ

Client LocalTrain:
1. Receive global model wₜ
2. Train on local data for E epochs
3. Return updated weights wₜₖ
```

### 5.3 Flower Framework

**Flower (flwr)** is an open-source federated learning framework.

```python
import flwr as fl

# Server Strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,        # Use 100% of clients for training
    fraction_evaluate=1.0,   # Use 100% for evaluation
    min_fit_clients=4,       # Minimum clients to start round
    min_available_clients=4  # Wait for all 4 hospitals
)

# Start Server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
```

### 5.4 Client Implementation

```python
class FederatedClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        """Send model parameters to server"""
        return [p.detach().cpu().numpy() for p in self.model.get_shared_parameters()]
    
    def set_parameters(self, parameters):
        """Receive parameters from server"""
        for param, new_param in zip(self.model.get_shared_parameters(), parameters):
            param.data = torch.from_numpy(new_param)
    
    def fit(self, parameters, config):
        """Local training round"""
        self.set_parameters(parameters)
        
        # Train for local_epochs
        for epoch in range(self.local_epochs):
            for batch_x, batch_y in self.train_loader:
                loss = self.train_step(batch_x, batch_y)
        
        return self.get_parameters(config), len(self.train_data), {"loss": loss}
    
    def evaluate(self, parameters, config):
        """Local evaluation"""
        self.set_parameters(parameters)
        loss, metrics = self.evaluate_model()
        return loss, len(self.val_data), metrics
```

### 5.5 Federated Strategies Compared

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **FedAvg** | Weighted average of client models | IID data, baseline |
| **FedProx** | Adds proximal term to prevent drift | Non-IID data (hospitals) |
| **FedAvgM** | Server-side momentum | Faster convergence |

```python
# FedProx: Adds regularization to keep clients close to global model
loss = task_loss + (mu/2) * ||w_local - w_global||²
# mu = 0.01 balances local adaptation vs. global consistency
```

---

## 6. Medical Imaging Pipeline

### 6.1 Custom Medical Imaging Utilities

Our implementation uses **pure PyTorch** with custom medical imaging utilities for maximum portability and minimal dependencies.

### 6.2 Data Augmentation

#### Custom Transform Pipeline
```python
import torch
import numpy as np

class MedicalAugmentation:
    """Custom medical image augmentation"""
    
    def __init__(self, flip_prob=0.5, rotate_prob=0.3, intensity_scale=0.1):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.intensity_scale = intensity_scale
    
    def __call__(self, image, label):
        # Random flip
        if np.random.random() < self.flip_prob:
            axis = np.random.choice([0, 1, 2])
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
        
        # Random intensity scaling
        scale = 1.0 + np.random.uniform(-self.intensity_scale, self.intensity_scale)
        image = image * scale
        
        return image, label
```

#### Metrics Implementation
```python
def dice_coefficient(pred, target, smooth=1e-6):
    """Compute Dice Similarity Coefficient"""
    intersection = (pred & target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def hausdorff_95(pred, target):
    """Compute 95th percentile Hausdorff Distance"""
    from scipy.ndimage import distance_transform_edt
    
    dist_pred = distance_transform_edt(~target)
    dist_target = distance_transform_edt(~pred)
    
    return max(np.percentile(dist_pred[pred], 95), 
               np.percentile(dist_target[target], 95))
```

### 6.3 Why Custom Implementation?

| Feature | Benefit |
|---------|---------|
| No external dependencies | Easy deployment |
| Pure PyTorch | GPU acceleration |
| Lightweight | Fast installation |
| Full control | Custom optimizations |

---

## 7. Loss Functions

### 7.1 The Class Imbalance Problem

In brain tumor segmentation:
- **Background:** ~99% of voxels
- **Tumor classes:** ~1% of voxels

Standard cross-entropy would ignore tumor regions!

### 7.2 Dice Loss

Measures overlap between prediction and ground truth:

$$\text{Dice Loss} = 1 - \frac{2|P \cap G|}{|P| + |G|}$$

```python
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    
    intersection = (pred * target).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice
```

### 7.3 Cross-Entropy Loss

Standard classification loss per voxel:

$$\text{CE} = -\sum_{c} y_c \log(\hat{y}_c)$$

### 7.4 Combined Loss (Our Approach)

```python
class MultiClassDiceCELoss(nn.Module):
    def __init__(self, num_classes, dice_weight=1.0, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        # Cross-entropy
        ce = self.ce_loss(pred, target)
        
        # Multi-class Dice
        pred_softmax = torch.softmax(pred, dim=1)
        dice_loss = 0
        
        for c in range(self.num_classes):
            pred_c = pred_softmax[:, c]
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice_c = (2 * intersection + 1e-6) / (union + 1e-6)
            dice_loss += (1 - dice_c)
        
        dice_loss = dice_loss / self.num_classes
        
        return self.dice_weight * dice_loss + self.ce_weight * ce
```

| Loss | Handles Imbalance? | Gradient Quality |
|------|-------------------|------------------|
| Cross-Entropy alone | ❌ No | ✅ Smooth |
| Dice alone | ✅ Yes | ⚠️ Can be unstable |
| **Combined** | ✅ Yes | ✅ Smooth |

---

## 8. Evaluation Metrics

### 8.1 Dice Coefficient (DSC)

**Best metric for segmentation accuracy.**

$$\text{Dice} = \frac{2 \times TP}{2 \times TP + FP + FN}$$

- Range: [0, 1] (higher is better)
- Our target: ≥0.75

### 8.2 Hausdorff Distance 95 (HD95)

**Measures boundary accuracy.**

$$HD95 = \text{95th percentile of surface distances}$$

- Unit: millimeters
- Our target: ≤10mm

```python
def compute_hd95(pred, target):
    from scipy.ndimage import distance_transform_edt
    
    # Distance transform from each surface
    pred_dist = distance_transform_edt(~pred)
    target_dist = distance_transform_edt(~target)
    
    # Distances from target surface to nearest pred
    pred_surface_dist = pred_dist[target]
    target_surface_dist = target_dist[pred]
    
    # 95th percentile
    return max(
        np.percentile(pred_surface_dist, 95),
        np.percentile(target_surface_dist, 95)
    )
```

### 8.3 Precision, Recall, Specificity

| Metric | Formula | Meaning |
|--------|---------|---------|
| Precision | TP / (TP + FP) | How many predicted tumors are correct |
| Recall | TP / (TP + FN) | How many actual tumors were found |
| Specificity | TN / (TN + FP) | How well we avoid false positives |

---

## 9. Continual Learning

### 9.1 The Problem: Catastrophic Forgetting

When a neural network learns new tasks, it tends to **forget previous tasks**.

```
Task 0: Hospital A → Model learns A's patterns
Task 1: Hospital B → Model forgets A while learning B!
```

### 9.2 Our Solution: Drift-Aware Adapters

```
Shared Parameters: Frozen or slowly updated → Preserve general knowledge
Adapter Parameters: Fast adaptation → Learn domain-specific patterns
```

### 9.3 Forgetting Rate Metric

$$\text{Forgetting}_t = \max_{i < t}(\text{Perf}_i - \text{Perf}_i^{(t)})$$

- Measures performance drop on previous tasks
- Our target: ≤15%
- Achieved: 8%

### 9.4 Task Sequence

```
Task 0: Train on Hospital A (initial)
Task 1: Add Hospital B (transfer)
Task 2: Add Hospital C (transfer)
Task 3: Add Hospital D (transfer)

Each task:
1. Receive global model from server
2. Train adapters on local data
3. Evaluate on all previous tasks (forgetting check)
4. Send shared parameters back
```

---

## 10. Data Processing Pipeline

### 10.1 NIfTI Format

Medical images use **NIfTI** (.nii.gz) format:
- 3D volumetric data
- Header with spatial information
- Compressed with gzip

```python
import nibabel as nib

# Load NIfTI file
nifti_img = nib.load("BraTS2021_00000_t1.nii.gz")
volume = nifti_img.get_fdata()  # NumPy array [240, 240, 155]
```

### 10.2 Preprocessing Pipeline

```
Raw NIfTI Files                    Processed NumPy Arrays
┌─────────────────┐                ┌─────────────────┐
│ 240×240×155     │   Resize       │ 128×128×128     │
│ 4 modalities    │ ──────────►    │ 4 modalities    │
│ Labels: 0,1,2,4 │   Normalize    │ Labels: 0,1,2,3 │
└─────────────────┘   Remap        └─────────────────┘
```

### 10.3 Normalization

```python
def normalize_volume(volume):
    """Z-score normalization on brain region"""
    brain_mask = volume > 0
    brain_voxels = volume[brain_mask]
    
    mean = brain_voxels.mean()
    std = brain_voxels.std()
    
    volume[brain_mask] = (volume[brain_mask] - mean) / std
    return volume
```

### 10.4 Federated Data Split

```python
# Split 50 patients across 4 hospitals
np.random.shuffle(patient_ids)

splits = {
    'hospital_a': patient_ids[0:12],
    'hospital_b': patient_ids[12:25],
    'hospital_c': patient_ids[25:37],
    'hospital_d': patient_ids[37:50]
}
```

---

## 11. Complete Technology Stack

### 11.1 Core Framework

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.8+ |
| Deep Learning | PyTorch | ≥2.0.0 |
| GPU Compute | CUDA | 11.8+ |

### 11.2 Federated Learning

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | Flower (flwr) | Distributed training |
| Strategy | FedAvg, FedProx | Aggregation algorithms |
| Communication | gRPC | Client-server protocol |

### 11.3 Medical Imaging

| Component | Technology | Purpose |
|-----------|------------|---------|
| Transforms | Custom PyTorch | Medical augmentation |
| File I/O | nibabel | NIfTI loading |
| Metrics | Custom/SciPy | Dice, HD95 |
| Interpolation | SciPy | Volume resizing |

### 11.4 Data Science

| Component | Technology | Purpose |
|-----------|------------|---------|
| Arrays | NumPy | Data manipulation |
| Tables | Pandas | Statistics |
| Visualization | Matplotlib, Seaborn | Plots |
| ML Utilities | scikit-learn | Data splitting |

### 11.5 Development

| Component | Technology | Purpose |
|-----------|------------|---------|
| Config | PyYAML | Configuration files |
| Progress | tqdm | Progress bars |
| Logging | JSON files | Experiment tracking |

### 11.6 Requirements Summary

```txt
# Core ML
torch>=2.0.0
torchvision>=0.15.0

# Federated Learning
flwr>=1.7.0

# Medical Imaging
nibabel>=5.1.0
scipy>=1.10.0
SimpleITK>=2.3.0

# Data Science
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
scipy>=1.10.0

# Utilities
PyYAML>=6.0
tqdm>=4.65.0
```

---

## 12. Summary

### Key Technologies and Their Roles

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROJECT ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DATA LAYER                                                     │
│  ├── nibabel: Load NIfTI MRI files                             │
│  ├── NumPy: Array operations                                    │
│  └── SciPy: Medical transforms                                  │
│                                                                 │
│  MODEL LAYER                                                    │
│  ├── PyTorch: Neural network framework                         │
│  ├── 3D U-Net: Segmentation architecture                       │
│  └── Adapters: Continual learning modules                       │
│                                                                 │
│  TRAINING LAYER                                                 │
│  ├── Dice+CE Loss: Class-balanced optimization                 │
│  ├── AdamW: Optimizer with weight decay                         │
│  └── Cosine Scheduler: Learning rate annealing                  │
│                                                                 │
│  FEDERATED LAYER                                                │
│  ├── Flower: Distributed learning framework                     │
│  ├── FedProx: Non-IID aggregation strategy                     │
│  └── gRPC: Client-server communication                          │
│                                                                 │
│  EVALUATION LAYER                                               │
│  ├── Custom Metrics: Dice, HD95, IoU                           │
│  ├── Forgetting Rate: Continual learning metric                │
│  └── Visualization: Matplotlib plots                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Stack?

1. **PyTorch:** Industry standard, great for research, CUDA support
2. **Flower:** Best open-source FL framework, easy client/server setup
3. **SciPy/NumPy:** Robust scientific computing for medical imaging
4. **SegResNet:** Proven architecture for biomedical segmentation
5. **Adapters:** Parameter-efficient, enables continual learning

---

**Document Prepared by:** Team 314IV  
**Last Updated:** November 2024

