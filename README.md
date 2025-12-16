# Federated Continual Learning for MRI Segmentation

**Team:** 314IV | **Topic:** #6 Federated Continual Learning for MRI Segmentation

A comprehensive implementation of federated continual learning for brain tumor segmentation using SegResNet with drift-aware adapters. This project demonstrates privacy-preserving collaborative learning across multiple virtual hospitals while mitigating catastrophic forgetting.

## 🎯 Project Overview

This project implements a federated continual learning framework for MRI brain tumor segmentation that addresses the dual challenges of:
- **Privacy preservation** through federated learning
- **Continual adaptability** through drift-aware adapters

The system simulates multiple hospitals collaborating on model training without sharing sensitive patient data, while continuously adapting to new data distributions.

## 📊 Key Features

- **SegResNet** with drift-aware adapters for continual learning
- **Multi-modal MRI processing** (T1, T1CE, T2, FLAIR)
- **Comprehensive evaluation metrics** (Dice, HD95, forgetting rate)
- **GPU-accelerated training** with mixed precision
- **Advanced visualization** and analysis tools
- **Production-ready code** with proper error handling

## 🖥️ Hardware Configuration

### Training System Specifications

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 5070 |
| **GPU Memory** | 12GB GDDR7 |
| **GPU Architecture** | Blackwell |
| **CPU** | AMD Ryzen 7 7800X3D (8 cores, 16 threads) |
| **RAM** | 32GB DDR5-5600 |
| **Storage** | 1TB + 2TB NVMe PCIe 4.0 SSDs |
| **Motherboard** | AMD B650 |
| **PSU** | 750W 80+ Gold |
| **Cooling** | Noctua NH-D15 Air Cooler |

### GPU Specifications (RTX 5070)

| Spec | Value |
|------|-------|
| Architecture | Blackwell |
| VRAM | 12GB GDDR7 |
| Memory Bandwidth | 672 GB/s |
| TDP | 250W |
| PCIe | 5.0 x16 |

## 🏗️ Architecture

```
├── SegResNet Backbone
│   ├── Encoder (blocks: 1, 2, 2, 4)
│   ├── Decoder (blocks: 1, 1, 1)
│   └── Drift-Aware Adapters (client-specific)
├── Federated Learning Layer
│   ├── Flower Framework
│   ├── FedAvg Strategy
│   └── Single GPU Training
└── Evaluation & Analysis
    ├── Segmentation Metrics
    ├── Forgetting Analysis
    └── Visualization Tools
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: RTX 5070 12GB or equivalent)
- Kaggle API credentials

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd federated-mri-segmentation
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup Kaggle credentials:**
```bash
# Place your kaggle.json in the project root
cp ~/Downloads/kaggle.json ./
```

### Data Preparation

```bash
# Download and preprocess BraTS2021 dataset
python src/data/download_dataset.py --download --extract --preprocess

# Verify dataset statistics
python src/data/download_dataset.py --stats
```

### Using the Trained Model

#### Inference on New Images
```bash
# Single patient inference
python src/inference/predict.py --input path/to/patient_folder --output results/predictions --visualize

# Batch inference on multiple patients
python src/inference/predict.py --input path/to/patients_directory --output results/predictions --batch --visualize

# With evaluation (if ground truth available)
python src/inference/predict.py --input path/to/patient_folder --output results/predictions --evaluate
```

### Training from Scratch

#### Federated Training
```bash
# Full experiment
python src/experiments/train_fcl.py --config configs/config.yaml

# Direct training (faster, no Ray dependency)
python src/experiments/train_direct.py --config configs/config.yaml
```

## 📈 Training Performance

### Training Time Breakdown (RTX 5070)

| Component | Time |
|-----------|------|
| Per round (4 clients × 3 epochs) | ~24 minutes |
| Training until convergence | **~74 hours** |
| Evaluation | ~3 hours |
| **Total training time** | **~80 hours (~3.3 days)** |

*Note: Early stopping triggered at round 185 after Dice score plateaued at 82.38%.*

### GPU Utilization

- **GPU memory usage:** 10.6 GB / 12 GB
- **GPU utilization:** 90-97%
- **Batch size:** 2 (limited by VRAM)

## 📊 Performance Metrics

### Segmentation Metrics (Test Set, n=60)

| Metric | TC | WT | ET | Mean |
|--------|-----|-----|-----|------|
| **Dice Score** | 82.84% | 87.12% | 77.18% | **82.38%** ✅ |
| **IoU (Jaccard)** | 70.76% | 77.16% | 62.85% | 70.12% |
| **HD95 (mm)** | 7.23 | 5.41 | 7.89 | 6.84 |
| **ASSD (mm)** | 1.94 | 1.32 | 2.21 | 1.82 |
| **Sensitivity** | 82.41% | 88.24% | 80.04% | 83.56% |
| **Precision** | 83.28% | 86.04% | 74.41% | 81.24% |
| **Specificity** | 99.82% | 99.68% | 99.84% | 99.78% |

*TC = Tumor Core, WT = Whole Tumor, ET = Enhancing Tumor*

### Additional Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **F1 Score** | 82.38% | Harmonic mean of precision and recall |
| **Volumetric Similarity** | 86.12% | Volume overlap coefficient |
| **Volume Error** | 12.8% | Average volume estimation error |
| **95% CI (Dice)** | [81.06%, 83.70%] | Confidence interval (n=60 test) |

### Inference Performance

| Metric | Value |
|--------|-------|
| Inference time per volume | 2.8 seconds |
| Total time per patient | 3.57 seconds |
| Throughput | 1,008 volumes/hour |
| GPU memory (inference) | 4.2 GB |

### Continual Learning Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Avg Forgetting Rate | 4.8% | Performance drop on previous tasks |
| Max Forgetting Rate | 8.2% | Worst-case forgetting |
| Backward Transfer | -2.3% | Impact on past task performance |
| Forward Transfer | +1.5% | Knowledge transfer to new tasks |
| Plasticity Score | 89.1% | Ability to learn new tasks |
| Stability Score | 95.2% | Resistance to forgetting |

### Comparison with Baselines

| Method | Dice | Δ Dice |
|--------|------|--------|
| **FCL + Adapters (Ours)** | **82.38%** | - |
| Centralized U-Net | 85.54% | -3.2% |
| FedAvg (no adapters) | 76.56% | +5.8% |
| Single Hospital | 74.98% | +7.4% |

## 🏥 Federated Setup

The system simulates 4 virtual hospitals with different data distributions:

- **Hospital A:** Focus on high-grade tumors
- **Hospital B:** Balanced tumor types
- **Hospital C:** Low-grade glioma emphasis
- **Hospital D:** Mixed pathology cases

### Federated Configuration

| Parameter | Value |
|-----------|-------|
| Number of Clients | 4 |
| Communication Rounds | 200 (185 until convergence) |
| Local Epochs per Round | 3 |
| Batch Size | 2 |
| Learning Rate | 0.0001 |
| Strategy | FedAvg |

## 🔧 Technical Details

### Model Architecture

```python
# SegResNet with adapters
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,  # T1ce, T1, T2, FLAIR
    out_channels=3,  # TC, WT, ET
    dropout_prob=0.2
)
```

### Single GPU Training

```python
# Single GPU training with mixed precision
device = torch.device('cuda')
model = model.to(device)
scaler = torch.cuda.amp.GradScaler()  # AMP for efficiency
```

### Federated Strategy

```python
# Continual learning strategy
strategy = ContinualLearningStrategy(
    config,
    save_path=model_dir,
    fraction_fit=1.0,
    min_available_clients=4
)
```

## 📁 Project Structure

```
federated-mri-segmentation/
├── configs/                 # Configuration files
│   ├── config.yaml         # Main configuration (RTX 5070)
│   └── config_final.yaml   # Detailed final config
├── src/
│   ├── data/               # Dataset handling
│   ├── models/             # SegResNet with adapters
│   ├── federated/          # FL client/server
│   ├── experiments/        # Training orchestrator
│   ├── inference/          # Prediction scripts
│   └── utils/              # Metrics & visualization
├── results/                # Experiment outputs
│   ├── final_experiment/   # Final model & report
│   └── predictions/        # Inference outputs
├── pretrained_models/      # Model weights
├── logs/                   # Training logs
├── data/                   # Dataset storage
└── requirements.txt        # Dependencies
```

## 🎯 Achieved Outcomes

✅ **Trained FCL Model:** SegResNet with drift-aware adapters achieving **82.38% Dice**
✅ **GPU Training:** Trained on RTX 5070 (12GB VRAM)
✅ **Evaluation Report:** Comprehensive metrics in `results/final_experiment/final_report.json`
✅ **Visualization:** Prediction visualizations in `results/predictions/`
✅ **Ready-to-Use Inference:** `src/inference/predict.py` for new image segmentation
✅ **Open-source Repository:** Well-documented, reproducible code

### Key Deliverables

| Deliverable | Location |
|-------------|----------|
| Trained Model | `pretrained_models/fcl_model/models/model.pt` |
| Main Config | `configs/config.yaml` |
| Inference Script | `src/inference/predict.py` |
| Final Report | `results/final_experiment/final_report.json` |
| Hardware Specs | `HARDWARE_SPECS.md` |
| Predictions | `results/predictions/` |

## 🔬 Research Contributions

1. **Drift-Aware Adapters:** Novel continual learning approach for medical imaging
2. **Federated Continual Learning:** Privacy-preserving adaptation to domain shifts
3. **Medical FL Benchmark:** Comprehensive evaluation framework for brain segmentation
4. **Multi-hospital Simulation:** Realistic federated learning evaluation setup

## 📝 Citation

```bibtex
@project{fcl_mri_segmentation_314iv,
  title={Federated Continual Learning for MRI Segmentation},
  authors={Ismoil Salohiddinov and Komiljon Qosimov and Abdurashid Djumabaev},
  team={314IV},
  year={2024},
  url={https://github.com/smailxpy/Federated-MRI-Segmentation}
}
```

## 🤝 Team

- **Ismoil Salohiddinov** (Coordinator) - Model architecture & FL implementation
- **Komiljon Qosimov** - Dataset preparation & evaluation metrics
- **Abdurrashid Djumabaev** - Adapters & continual learning

## 📧 Contact

- **Team Lead:** Ismoil Salohiddinov (220626@centralasian.uz)
- **GitHub:** [smailxpy/Federated-MRI-Segmentation](https://github.com/smailxpy/Federated-MRI-Segmentation)

## ⚠️ Requirements

### Minimum Requirements
- 1× GPU with 12GB+ VRAM
- 32GB RAM
- 100GB storage

### Recommended (Training Configuration)
- NVIDIA RTX 5070 (12GB) or equivalent
- 32GB DDR5 RAM
- AMD Ryzen 7 7800X3D or Intel i7-13700K
- 750W 80+ Gold PSU
- NVMe SSD storage

---

**Built with ❤️ by Team 314IV for advancing privacy-preserving medical AI**
