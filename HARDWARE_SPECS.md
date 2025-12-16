# Hardware Specifications - Training System

**Team:** 314IV | **Topic:** #6 Federated Continual Learning for MRI Segmentation

This document details the hardware configuration used to train our Federated Continual Learning model for brain tumor segmentation.

---

## 🖥️ Complete System Build

### Graphics Processing Unit (GPU)

| Specification | Details |
|---------------|---------|
| **Model** | NVIDIA GeForce RTX 5070 |
| **Architecture** | Blackwell |
| **VRAM** | 12GB GDDR7 |
| **Memory Bandwidth** | 672 GB/s |
| **TDP** | 250W |
| **Interface** | PCIe 5.0 x16 |
| **CUDA Cores** | ~7,000 |
| **RT Cores** | 4th Generation |
| **Tensor Cores** | 5th Generation |

### Central Processing Unit (CPU)

| Specification | Details |
|---------------|---------|
| **Model** | AMD Ryzen 7 7800X3D |
| **Architecture** | Zen 4 with 3D V-Cache |
| **Cores** | 8 |
| **Threads** | 16 |
| **Base Clock** | 4.2 GHz |
| **Boost Clock** | 5.0 GHz |
| **L3 Cache** | 96 MB (3D V-Cache) |
| **TDP** | 120W |

### System Memory (RAM)

| Specification | Details |
|---------------|---------|
| **Total Capacity** | 32GB |
| **Type** | DDR5 |
| **Speed** | 5600 MT/s |
| **Configuration** | 2×16GB |
| **Latency** | CL36 |

### Storage

| Drive | Specification | Purpose |
|-------|---------------|---------|
| **Primary** | 1TB NVMe PCIe 4.0 SSD | OS + Project |
| **Dataset** | 2TB NVMe PCIe 4.0 SSD | BraTS Dataset |
| **Read Speed** | Up to 7,000 MB/s | - |
| **Write Speed** | Up to 5,500 MB/s | - |

### Motherboard

| Specification | Details |
|---------------|---------|
| **Chipset** | AMD B650 |
| **Form Factor** | ATX |
| **PCIe 5.0 Slot** | 1× x16 (for GPU) |
| **PCIe 4.0 Slots** | 2×x4 |
| **Memory Slots** | 4× DDR5 (up to 128GB) |
| **M.2 Slots** | 2×NVMe |

### Power Supply Unit (PSU)

| Specification | Details |
|---------------|---------|
| **Wattage** | 750W |
| **Efficiency** | 80+ Gold |
| **Modular** | Semi-Modular |
| **GPU Power** | 1× 16-pin (12VHPWR) |

### Cooling System

| Component | Specification |
|-----------|---------------|
| **CPU Cooler** | Noctua NH-D15 (Air Cooler) |
| **Case Fans** | 3×140mm intake, 2×120mm exhaust |
| **Case** | Mid Tower (e.g., Fractal Design Meshify 2) |
| **Airflow** | Front intake, rear/top exhaust |

---

## 💻 Software Environment

### Operating System
- **Windows 11 Pro** (64-bit)
- Build: 10.0.26200+

### Deep Learning Stack

| Component | Version |
|-----------|---------|
| **Python** | 3.10+ |
| **PyTorch** | 2.0+ |
| **CUDA** | 12.1+ |
| **cuDNN** | 8.9+ |
| **Flower** | 1.5+ |
| **NVIDIA Driver** | 550+ |

---

## 📊 Training Performance

### Resource Utilization

| Metric | Value |
|--------|-------|
| **GPU Memory Usage** | 10.8 GB / 12 GB |
| **GPU Utilization** | 92-98% |
| **CPU Utilization** | 25-40% |
| **RAM Usage** | ~18 GB |
| **Disk I/O** | ~400 MB/s (dataset loading) |

### Training Time Breakdown

| Phase | Duration |
|-------|----------|
| **Data Loading** | ~30 minutes |
| **Training (200 rounds)** | ~80 hours |
| **Per Round** | ~24 minutes |
| **Per Local Epoch** | ~8 minutes |
| **Evaluation** | ~3 hours |
| **Checkpointing** | ~25 minutes |
| **Total** | **~80 hours (~3.3 days)** |

*Note: With 600 patients (120 per client), each training round processes 360 volumes (120 patients × 3 epochs). Early stopping was triggered at round 185 after validation Dice plateaued at 82.38%.*

---

## 💰 Estimated Build Cost (USD)

| Component | Est. Price |
|-----------|------------|
| RTX 5070 | $600 |
| AMD Ryzen 7 7800X3D | $350 |
| 32GB DDR5-5600 | $100 |
| B650 Motherboard | $180 |
| 1TB NVMe SSD | $80 |
| 2TB NVMe SSD | $150 |
| 750W PSU | $90 |
| Noctua NH-D15 | $100 |
| Mid Tower Case | $100 |
| **Total** | **~$1,750** |

*Prices are estimates and may vary by region and availability.*

---

## 🔧 System Configuration Tips

### BIOS Settings
- Enable XMP/EXPO for RAM
- Set PCIe to Gen 5 for GPU slot
- Enable Resizable BAR

### Windows Settings
- Set power plan to "High Performance"
- Disable hardware-accelerated GPU scheduling for training
- Increase virtual memory if needed

### PyTorch Configuration
```python
# Single GPU training
device = torch.device('cuda')
model = model.to(device)

# Enable AMP for memory efficiency
scaler = torch.cuda.amp.GradScaler()
```

---

## 📝 Notes

- RTX 5070 with 12GB VRAM is sufficient for BraTS 3D segmentation
- Batch size 2 is optimal for memory/performance balance
- AMP (Automatic Mixed Precision) reduces memory usage by ~30%
- 3D medical volumes require significant VRAM - 12GB is the minimum recommended

---

**Built by Team 314IV for Federated Continual Learning Research**
