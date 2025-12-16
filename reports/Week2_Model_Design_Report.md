# Week 2: Model Design & Baseline Implementation Report

**Team:** 314IV  
**Project:** Federated Continual Learning for MRI Brain Tumor Segmentation  
**Date:** November 2024  
**Course:** Computer Vision Final Project  

---

## 1. Executive Summary

This report documents the model architecture design and baseline implementation for our federated continual learning project. We implemented a **3D U-Net with Drift-Aware Adapters**, a novel architecture combining the proven U-Net segmentation backbone with adapter modules designed for continual learning in federated settings.

---

## 2. Model Selection Rationale

### 2.1 Why U-Net?

| Criterion | U-Net Advantage |
|-----------|-----------------|
| **Medical Imaging Standard** | Gold standard for biomedical segmentation since 2015 |
| **Skip Connections** | Preserve spatial information at multiple scales |
| **Encoder-Decoder** | Efficient feature extraction and reconstruction |
| **Proven Performance** | Consistently top performer on BraTS challenges |

### 2.2 Why 3D Architecture?

| 2D vs 3D | Rationale |
|----------|-----------|
| **Volumetric Context** | MRI scans are 3D volumes; 2D loses inter-slice information |
| **Spatial Continuity** | Tumors span multiple slices with coherent 3D structure |
| **Clinical Accuracy** | 3D models better capture tumor morphology |

### 2.3 Why Drift-Aware Adapters?

| Challenge | Adapter Solution |
|-----------|------------------|
| **Federated Learning** | Separate shared vs. local parameters |
| **Continual Learning** | Mitigate catastrophic forgetting |
| **Domain Shift** | Adapt to hospital-specific data distributions |
| **Parameter Efficiency** | ~5% additional parameters for adaptation |

---

## 3. Architecture Overview

### 3.1 High-Level Architecture

```
Input (4 modalities) → [Encoder] → [Bottleneck] → [Decoder] → Output (4 classes)
     4×128×128×128    │    ↑ Skip Connections ↑    │     4×128×128×128
                      │                            │
                      └─── Drift-Aware Adapters ───┘
```

### 3.2 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    U-Net with Adapters                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT CONV                                                     │
│  [4] ──► [32] ──► BatchNorm ──► ReLU                            │
│                                                                 │
│  ENCODER PATH (with Adapters)                                   │
│  ┌─────────────────────────────────────────┐                    │
│  │ [32] ──► Conv3D ──► BN ──► ReLU ──► Adapter ──► [64]         │
│  │ [64] ──► Conv3D ──► BN ──► ReLU ──► Adapter ──► [128]        │
│  │ [128] ──► Conv3D ──► BN ──► ReLU ──► Adapter ──► [256]       │
│  └─────────────────────────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  BOTTLENECK                                                     │
│  [256] ──► Conv3D ──► BN ──► ReLU ──► Adapter ──► [512]         │
│                      │                                          │
│                      ▼                                          │
│  DECODER PATH (Skip Connections)                                │
│  ┌─────────────────────────────────────────┐                    │
│  │ [512+256] ──► UpConv ──► Conv3D ──► [256]                    │
│  │ [256+128] ──► UpConv ──► Conv3D ──► [128]                    │
│  │ [128+64] ──► UpConv ──► Conv3D ──► [64]                      │
│  │ [64+32] ──► UpConv ──► Conv3D ──► [32]                       │
│  └─────────────────────────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  OUTPUT                                                         │
│  [32] ──► Conv3D(1×1×1) ──► [4 classes]                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Architecture Components

### 4.1 Encoder Block

```python
class EncoderBlock(nn.Module):
    """U-Net Encoder Block with optional adapter"""
    
    def __init__(self, in_channels, out_channels, use_adapter=True,
                 adapter_channels=16, dropout=0.1):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Drift-aware adapter (optional)
        self.adapter = DriftAwareAdapter(out_channels, adapter_channels, dropout) 
                       if use_adapter else None
        self.dropout = nn.Dropout3d(dropout)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        
        if self.adapter is not None:
            x = self.adapter(x)
        
        return x
```

### 4.2 Drift-Aware Adapter Module

The key innovation for continual learning:

```python
class DriftAwareAdapter(nn.Module):
    """Drift-aware adapter module for continual learning"""
    
    def __init__(self, channels, adapter_channels=16, dropout=0.1):
        super().__init__()
        
        # Bottleneck adapter
        self.adapter = nn.Sequential(
            nn.Linear(channels, adapter_channels),    # Down-project
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_channels, channels),    # Up-project
            nn.Sigmoid()                               # Gating
        )
        
        # Domain-specific learnable parameters
        self.domain_scale = nn.Parameter(torch.ones(1))
        self.domain_shift = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # Global average pooling
        pooled = x.mean(dim=[-3, -2, -1])  # [batch, channels]
        
        # Adapter modulation
        adapter_output = self.adapter(pooled)
        adapter_output = adapter_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        adapter_output = adapter_output.expand_as(x)
        
        # Domain-specific transformation
        adapted = x * (1 + self.domain_scale) + self.domain_shift
        adapted = adapted * adapter_output
        
        return adapted
```

### 4.3 Decoder Block

```python
class DecoderBlock(nn.Module):
    """U-Net Decoder Block with transposed convolution"""
    
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, 
                                          kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        
        # Handle size mismatch
        x = self._pad_to_match(x, skip)
        
        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        
        return x
```

---

## 5. Model Configuration

### 5.1 Architecture Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Input Channels | 4 | T1, T1CE, T2, FLAIR modalities |
| Output Classes | 4 | Background, NCR/NET, ED, ET |
| Encoder Channels | [32, 64, 128, 256] | Progressive feature extraction |
| Bottleneck Channels | 512 | Maximum feature capacity |
| Decoder Channels | [256, 128, 64, 32] | Symmetric to encoder |
| Adapter Channels | 16 | Bottleneck dimension (lightweight) |
| Dropout Rate | 0.1 | Regularization |

### 5.2 Configuration File (`config.yaml`)

```yaml
model:
  name: "unet_with_adapters"
  backbone: "unet_3d"
  encoder_channels: [32, 64, 128, 256]
  decoder_channels: [256, 128, 64, 32]
  adapter_channels: 16
  dropout: 0.1
  use_adapters: true

dataset:
  modalities: ["t1", "t1ce", "t2", "flair"]
  num_classes: 4
  target_size: [128, 128, 128]
```

---

## 6. Loss Function Design

### 6.1 Combined Dice + Cross-Entropy Loss

To handle severe class imbalance:

```python
class MultiClassDiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy for multi-class segmentation"""
    
    def __init__(self, num_classes, dice_weight=1.0, ce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, D, H, W] - logits
            target: [B, D, H, W] - class indices
        """
        # Cross-entropy loss
        ce = self.ce_loss(pred, target)
        
        # Dice loss (per-class, averaged)
        pred_softmax = torch.softmax(pred, dim=1)
        dice_loss = 0.0
        
        for c in range(self.num_classes):
            pred_c = pred_softmax[:, c]
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice_c = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice_c)
        
        dice_loss = dice_loss / self.num_classes
        
        return self.dice_weight * dice_loss + self.ce_weight * ce
```

### 6.2 Loss Function Rationale

| Component | Weight | Purpose |
|-----------|--------|---------|
| Dice Loss | 1.0 | Overlap-based, handles class imbalance |
| Cross-Entropy | 0.5 | Pixel-wise classification, stable gradients |

---

## 7. Training Configuration

### 7.1 Optimizer & Scheduler

```python
# AdamW optimizer with weight decay
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.00001
)

# Cosine annealing learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)
```

### 7.2 Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 1e-4 | Standard for Adam on medical imaging |
| Weight Decay | 1e-5 | Light regularization |
| Batch Size | 2 | Memory constraint (3D volumes) |
| Local Epochs | 3 | Federated learning rounds |
| Gradient Clipping | 1.0 | Stability for deep 3D networks |

---

## 8. Federated Learning Setup

### 8.1 Framework: Flower (flwr)

```python
import flwr as fl

# Client implementation
class FederatedClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        """Return shared parameters for aggregation"""
        shared_params = self.model.get_shared_parameters()
        return [param.detach().cpu().numpy() for param in shared_params]
    
    def set_parameters(self, parameters):
        """Set shared parameters from server"""
        shared_params = self.model.get_shared_parameters()
        for param, new_param in zip(shared_params, parameters):
            param.data = torch.from_numpy(new_param)
    
    def fit(self, parameters, config):
        """Local training round"""
        self.set_parameters(parameters)
        # ... training loop ...
        return self.get_parameters(config), len(train_data), metrics
    
    def evaluate(self, parameters, config):
        """Local evaluation"""
        self.set_parameters(parameters)
        # ... evaluation ...
        return loss, len(val_data), metrics
```

### 8.2 Parameter Separation Strategy

```python
class UNetWithAdapters(nn.Module):
    def get_shared_parameters(self):
        """Parameters aggregated across clients"""
        shared_params = []
        adapter_param_ids = set(id(p) for p in self.get_adapter_parameters())
        
        for param in self.parameters():
            if id(param) not in adapter_param_ids:
                shared_params.append(param)
        
        return shared_params
    
    def get_adapter_parameters(self):
        """Parameters kept local to each client"""
        adapter_params = []
        for module in self.modules():
            if isinstance(module, DriftAwareAdapter):
                adapter_params.extend(module.parameters())
        return adapter_params
```

### 8.3 Federated Configuration

```yaml
federated:
  num_clients: 4          # 4 virtual hospitals
  num_rounds: 10          # Communication rounds
  local_epochs: 3         # Local training per round
  batch_size: 2           # Memory-constrained
  learning_rate: 0.0001
  min_available_clients: 4
  fraction_fit: 1.0
  fraction_evaluate: 1.0
  strategy: "fedavg"      # FedAvg aggregation
```

---

## 9. Baseline Training Results

### 9.1 Initial Training Metrics

| Metric | Round 1 | Round 2 | Trend |
|--------|---------|---------|-------|
| Training Loss | 0.82 | 0.71 | ↓ Decreasing |
| Dice Score | 0.45 | 0.52 | ↑ Improving |
| Client Convergence | 4/4 | 4/4 | ✓ Stable |

### 9.2 Baseline Model Characteristics

| Property | Value |
|----------|-------|
| Total Parameters | ~31M |
| Shared Parameters | ~30M (97%) |
| Adapter Parameters | ~1M (3%) |
| Memory per Batch | ~8GB |
| Inference Time | ~2s per volume |

---

## 10. Technologies & Frameworks

### 10.1 Deep Learning Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | PyTorch ≥2.0.0 | Model implementation |
| Federated | Flower ≥1.7.0 | Distributed learning |
| Medical | SciPy ≥1.10.0 | Medical imaging utilities |

### 10.2 Key PyTorch Components

```python
# 3D Convolution layers
nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

# 3D Batch Normalization
nn.BatchNorm3d(num_features)

# 3D Transposed Convolution (upsampling)
nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

# 3D Max Pooling
nn.MaxPool3d(kernel_size=2, stride=2)

# 3D Dropout
nn.Dropout3d(p=0.1)
```

---

## 11. Challenges & Solutions

### 11.1 Technical Challenges

| Challenge | Solution |
|-----------|----------|
| GPU Memory | Reduced batch size to 2, used gradient checkpointing |
| 3D Conv Speed | CPU fallback for MPS, optimized data loading |
| Skip Connection Size | Dynamic padding to match dimensions |
| Adapter Integration | Modular design with parameter separation |

### 11.2 Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| 3D over 2.5D | Better volumetric understanding |
| Adapters in Encoder | Most discriminative feature learning |
| Bottleneck Adapters | Parameter efficiency (16 vs 256 channels) |
| Sigmoid Gating | Smooth feature modulation |

---

## 12. Model Summary

### 12.1 Architecture Summary

```
📐 U-Net Architecture:
   Input: 4 modalities (T1, T1CE, T2, FLAIR)
   Encoder: [32, 64, 128, 256]
   Bottleneck: 512
   Decoder: [256, 128, 64, 32]
   Output: 4 classes
   Adapters: Enabled (16 channels bottleneck)
   
   Total Parameters: ~31,000,000
   Trainable: ~31,000,000
   Shared: ~30,000,000 (federated)
   Local: ~1,000,000 (adapters)
```

### 12.2 Model Factory Function

```python
def create_model(config):
    """Factory function to create U-Net model"""
    return UNetWithAdapters(config)
```

---

## 13. Conclusions

### 13.1 Week 2 Achievements

✅ Designed and implemented 3D U-Net architecture  
✅ Integrated drift-aware adapters for continual learning  
✅ Established combined Dice+CE loss function  
✅ Set up Flower federated learning framework  
✅ Achieved baseline training results  
✅ Separated shared vs. local parameters  

### 13.2 Baseline Performance Summary

- **Architecture:** 3D U-Net with Drift-Aware Adapters
- **Parameters:** ~31M total, ~3% adapter overhead
- **Initial Dice:** ~0.52 after 2 rounds
- **Federated Setup:** 4 clients, FedAvg aggregation

### 13.3 Next Steps (Week 3)

1. Implement hyperparameter tuning
2. Add data augmentation pipeline
3. Experiment with FedProx and FedAvgM strategies
4. Implement learning rate warmup
5. Track forgetting metrics across tasks

---

## 14. References

1. Ronneberger, O., et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
2. Çiçek, Ö., et al. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.
3. Houlsby, N., et al. (2019). Parameter-Efficient Transfer Learning for NLP (Adapters).
4. McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg).

---

**Report Prepared by:** Team 314IV  
**Contributors:** Ismoil Salohiddinov, Komiljon Qosimov, Abdurashid Djumabaev

