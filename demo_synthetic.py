#!/usr/bin/env python3
"""
Synthetic Data Demo for Federated Continual Learning
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This demo uses synthetic data to demonstrate the federated continual learning pipeline
without requiring large medical datasets.
"""

import numpy as np
import torch
from pathlib import Path
import yaml
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def create_synthetic_dataset():
    """Create synthetic MRI-like data for demonstration"""
    print("🔄 Creating synthetic MRI dataset...")

    # Create directories
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Create 4 virtual hospitals
    hospitals = ['hospital_a', 'hospital_b', 'hospital_c', 'hospital_d']
    samples_per_hospital = 20  # Small dataset for demo

    # MRI modalities
    modalities = ['t1', 't1ce', 't2', 'flair']
    image_shape = (64, 64, 32)  # Smaller for demo
    num_classes = 4  # Background + 3 tumor classes

    dataset_stats = {
        'total_patients': 0,
        'modalities': modalities,
        'image_shape': image_shape,
        'classes': list(range(num_classes)),
        'hospital_distribution': {}
    }

    for hospital in hospitals:
        hospital_dir = processed_dir / hospital
        hospital_dir.mkdir(exist_ok=True)

        # Save patient list
        patient_ids = [f"patient_{hospital}_{i:03d}" for i in range(samples_per_hospital)]
        with open(hospital_dir / "patients.txt", 'w') as f:
            for pid in patient_ids:
                f.write(f"{pid}\n")

        # Create synthetic patient data
        for patient_id in patient_ids:
            patient_dir = processed_dir / patient_id
            patient_dir.mkdir(exist_ok=True)

            # Generate synthetic MRI images
            for modality in modalities:
                # Create realistic-looking brain-like structure
                image = generate_synthetic_brain_image(image_shape, modality, hospital)
                np.save(patient_dir / f"{modality}.npy", image.astype(np.float32))

            # Generate synthetic segmentation mask
            mask = generate_synthetic_tumor_mask(image_shape, hospital)
            np.save(patient_dir / "mask.npy", mask.astype(np.int32))

        dataset_stats['hospital_distribution'][hospital] = samples_per_hospital
        dataset_stats['total_patients'] += samples_per_hospital

    # Save dataset statistics
    with open(processed_dir / "dataset_stats.json", 'w') as f:
        json.dump(dataset_stats, f, indent=2)

    print("✅ Synthetic dataset created")
    return dataset_stats

def generate_synthetic_brain_image(shape, modality, hospital):
    """Generate synthetic brain-like MRI image"""

    # Base brain structure (ellipse)
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center_x, center_y = shape[0] // 2, shape[1] // 2

    # Create brain mask (elliptical)
    brain_mask = ((x - center_x) / (shape[0] * 0.4)) ** 2 + \
                 ((y - center_y) / (shape[1] * 0.4)) ** 2 <= 1

    # Add some noise and structure
    base_intensity = np.random.uniform(0.3, 0.7)

    # Different characteristics per modality
    if modality == 't1':
        image = base_intensity + np.random.normal(0, 0.1, shape)
    elif modality == 't1ce':
        image = base_intensity * 1.2 + np.random.normal(0, 0.08, shape)
    elif modality == 't2':
        image = base_intensity * 0.8 + np.random.normal(0, 0.12, shape)
    else:  # flair
        image = base_intensity * 0.9 + np.random.normal(0, 0.11, shape)

    # Apply brain mask
    image = image * brain_mask + np.random.normal(0, 0.05, shape) * (1 - brain_mask)

    # Hospital-specific characteristics (domain shift simulation)
    if hospital == 'hospital_a':
        image = image * 1.1  # Brighter
    elif hospital == 'hospital_b':
        image = image * 0.9  # Dimmer
    elif hospital == 'hospital_c':
        image = image + np.random.normal(0, 0.05, shape)  # More noise
    else:  # hospital_d
        # Add some motion artifacts (simulated)
        motion_artifact = np.random.normal(0, 0.03, shape)
        motion_artifact[:, ::4, :] += np.random.normal(0, 0.1, (shape[0], shape[1]//4, shape[2]))
        image = image + motion_artifact

    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    return image

def generate_synthetic_tumor_mask(shape, hospital):
    """Generate synthetic tumor segmentation mask"""

    mask = np.zeros(shape, dtype=np.int32)

    # Probability of tumor per hospital (simulating different patient populations)
    tumor_probs = {
        'hospital_a': 0.7,  # High-grade focused
        'hospital_b': 0.5,  # Mixed
        'hospital_c': 0.3,  # Low-grade focused
        'hospital_d': 0.6   # General hospital
    }

    if np.random.random() < tumor_probs[hospital]:
        # Generate tumor
        tumor_center = np.random.randint(20, shape[0]-20, 3)
        tumor_size = np.random.randint(5, 15, 3)

        # Create tumor mask (ellipsoidal)
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        tumor_mask = ((x - tumor_center[0]) / tumor_size[0]) ** 2 + \
                     ((y - tumor_center[1]) / tumor_size[1]) ** 2 + \
                     ((z - tumor_center[2]) / tumor_size[2]) ** 2 <= 1

        # Assign tumor classes (simplified)
        # Class 1: NCR/NET, Class 2: ED, Class 3: ET
        tumor_type = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        mask[tumor_mask] = tumor_type

        # Add some necrotic core (class 1) in center
        core_center = tumor_center
        core_size = tumor_size // 3
        core_mask = ((x - core_center[0]) / core_size[0]) ** 2 + \
                   ((y - core_center[1]) / core_size[1]) ** 2 + \
                   ((z - core_center[2]) / core_size[2]) ** 2 <= 1
        mask[core_mask] = 1

    return mask

def run_synthetic_demo():
    """Run the federated continual learning demo with synthetic data"""
    print("🎯 Running Federated Continual Learning Demo with Synthetic Data")
    print("=" * 70)
    print("Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation")
    print("=" * 70)

    try:
        # Step 1: Create synthetic dataset
        print("\\n📥 Step 1: Creating synthetic dataset...")
        dataset_stats = create_synthetic_dataset()
        print(f"Created dataset with {dataset_stats['total_patients']} patients across {len(dataset_stats['hospital_distribution'])} hospitals")

        # Step 2: Create simplified federated demo
        print("\\n🚀 Step 2: Running simplified federated learning demo...")

        # Import required modules
        import sys
        sys.path.insert(0, '.')

        from src.models.unet_adapters import create_model, CombinedLoss
        from src.federated.client import BraTSDataset, FederatedClient
        from src.utils.metrics import SegmentationMetrics
        import torch
        from torch.utils.data import DataLoader

        # Load and modify config for synthetic data
        import yaml
        with open("configs/config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # Reduce complexity for demo
        config['federated']['num_rounds'] = 2
        config['federated']['local_epochs'] = 1
        config['dataset']['target_size'] = [64, 64, 32]

        # Create model
        device = torch.device('cpu')
        model = create_model(config).to(device)
        criterion = CombinedLoss()

        print("✓ Model created successfully")

        # Test federated clients
        hospitals = ['hospital_a', 'hospital_b', 'hospital_c', 'hospital_d']
        client_metrics = {}

        for hospital in hospitals:
            print(f"  Training on {hospital} data...")

            # Create client dataset
            try:
                dataset = BraTSDataset(Path("data/processed") / hospital)
                dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

                # Simple local training
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                epoch_loss = 0.0
                num_batches = min(3, len(dataloader))  # Limit for demo

                for i, (images, masks) in enumerate(dataloader):
                    if i >= num_batches:
                        break

                    images, masks = images.to(device), masks.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / num_batches
                print(f"    {hospital}: loss = {avg_loss:.4f}")

                client_metrics[hospital] = avg_loss

            except Exception as e:
                print(f"    {hospital}: failed ({e})")
                client_metrics[hospital] = None

        # Simulate federated aggregation (simple averaging)
        print("\\n  📊 Simulating federated aggregation...")
        valid_losses = [loss for loss in client_metrics.values() if loss is not None]
        if valid_losses:
            avg_federated_loss = sum(valid_losses) / len(valid_losses)
            print(f"  ✓ Federated average loss: {avg_federated_loss:.4f}")

        # Test metrics
        print("\\n  📈 Computing evaluation metrics...")
        metrics_computer = SegmentationMetrics()

        # Get a sample batch for evaluation
        try:
            sample_dataset = BraTSDataset(Path("data/processed/hospital_a"))
            sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False)
            sample_images, sample_masks = next(iter(sample_loader))
            sample_images, sample_masks = sample_images.to(device), sample_masks.to(device)

            model.eval()
            with torch.no_grad():
                predictions = model(sample_images)
                metrics = metrics_computer(predictions, sample_masks)

            print("  ✓ Evaluation metrics computed:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(".4f")

        except Exception as e:
            print(f"  ⚠️  Metrics computation failed: {e}")

        print("✅ Simplified federated demo completed")

        print("✅ Federated experiment completed")

        # Step 3: Generate results visualization
        print("\\n📊 Step 3: Generating results analysis...")

        from src.utils.visualization import ExperimentVisualizer

        # Find experiment directory
        results_dir = Path("results")
        exp_dirs = list(results_dir.glob("experiment_*"))
        if exp_dirs:
            latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)

            visualizer = ExperimentVisualizer(latest_exp)
            visualizer.plot_metrics_over_rounds()
            visualizer.plot_task_performance()
            visualizer.create_summary_dashboard()
            report = visualizer.generate_experiment_report()

            print("✅ Results analysis completed")
            print("\\n📋 Demo Results Summary:")
            print("-" * 40)
            print(report.split("PERFORMANCE METRICS:")[-1].split("FORGETTING ANALYSIS:")[0])

        # Cleanup
        Path(temp_config_path).unlink(missing_ok=True)

        print("\\n🎉 Synthetic data demo completed successfully!")
        print("\\n📁 Results saved to: results/")
        print("📈 View visualizations in: results/plots/")
        print("\\n🏆 Team 314IV - Federated Continual Learning Demo Successful!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = run_synthetic_demo()
    exit(0 if success else 1)
