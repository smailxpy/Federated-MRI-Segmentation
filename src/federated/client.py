#!/usr/bin/env python3
"""
Federated Learning Client for MRI Segmentation
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This module implements the federated learning client using Flower framework
with proper support for BraTS2021 4-class tumor segmentation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.common import NDArrays, Scalar
import yaml
import warnings
import sys
import os
warnings.filterwarnings("ignore")

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class BraTS2021Dataset(Dataset):
    """BraTS2021 Dataset for federated clients with 4-class segmentation"""

    def __init__(self, data_dir: Path, config: dict, transform=None):
        self.data_dir = Path(data_dir)
        self.config = config
        self.transform = transform
        self.modalities = config['dataset']['modalities']
        self.num_classes = config['dataset']['num_classes']
        
        # Find all patient directories
        self.patient_dirs = self._find_patient_dirs()
        
        if len(self.patient_dirs) == 0:
            raise ValueError(f"No patient data found in {data_dir}")

    def _find_patient_dirs(self) -> List[Path]:
        """Find all valid patient directories"""
        patient_dirs = []
        
        # Look for BraTS2021_* directories
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.startswith('BraTS2021_'):
                # Verify all required files exist
                if self._verify_patient_dir(item):
                    patient_dirs.append(item)
        
        return sorted(patient_dirs)

    def _verify_patient_dir(self, patient_dir: Path) -> bool:
        """Verify patient directory has all required files"""
        required_files = [f"{m}.npy" for m in self.modalities] + ["mask.npy"]
        return all((patient_dir / f).exists() for f in required_files)

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        
        # Load modalities
        images = []
        for modality in self.modalities:
            img = np.load(patient_dir / f"{modality}.npy")
            images.append(img)
        
        # Stack modalities: [C, D, H, W]
        x = np.stack(images, axis=0).astype(np.float32)
        
        # Load segmentation mask
        mask = np.load(patient_dir / "mask.npy").astype(np.int64)
        
        # Convert to tensors
        x = torch.from_numpy(x)
        y = torch.from_numpy(mask)
        
        if self.transform:
            x, y = self.transform(x, y)
        
        return x, y


class FederatedClient(fl.client.NumPyClient):
    """Federated Learning Client with Continual Learning Support for BraTS2021"""

    def __init__(self, client_id: str, data_dir: Path, config: dict):
        self.client_id = client_id
        self.data_dir = Path(data_dir)
        self.config = config
        
        # Detect device (use CPU for 3D as MPS has limited 3D support)
        device_config = config['hardware']['device']
        if device_config == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            # MPS has limited support for 3D operations, fall back to CPU
            self.device = torch.device('cpu')
        
        print(f"[CLIENT]  Client {client_id} using device: {self.device}")

        # Initialize model
        from src.models.unet_adapters import create_model
        self.model = create_model(config)
        self.model.to(self.device)
        
        # Loss function for multi-class segmentation
        self.criterion = MultiClassDiceCELoss(
            num_classes=config['dataset']['num_classes'],
            dice_weight=1.0,
            ce_weight=0.5
        )

        # Setup data
        self._setup_data()

        # Training state
        self.current_task = 0
        self.task_history = []
        self.best_dice = 0.0

        print(f"[OK] Client {client_id} initialized with {len(self.train_loader.dataset)} training samples")

    def _setup_data(self):
        """Setup data loaders for the client"""
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        
        # Create datasets
        train_dataset = BraTS2021Dataset(train_dir, self.config)
        val_dataset = BraTS2021Dataset(val_dir, self.config)

        # Create data loaders with reduced batch size for 3D data
        batch_size = self.config['federated']['batch_size']
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(self.config['hardware']['num_workers'], 2),
            pin_memory=self.config['hardware']['pin_memory'] and self.device.type == 'cuda'
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(self.config['hardware']['num_workers'], 2),
            pin_memory=self.config['hardware']['pin_memory'] and self.device.type == 'cuda'
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters for federated aggregation"""
        shared_params = self.model.get_shared_parameters()
        return [param.detach().cpu().numpy() for param in shared_params]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from federated aggregation"""
        shared_params = self.model.get_shared_parameters()
        for param, new_param in zip(shared_params, parameters):
            param.data = torch.from_numpy(new_param).to(self.device)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Local training round with comprehensive metrics and verbose output"""
        print(f"\n{'='*50}")
        print(f"[TRAIN] Client {self.client_id} - Starting Local Training")
        print(f"{'='*50}")
        print(f"  Dataset: {len(self.train_loader.dataset)} samples")
        print(f"  Batches: {len(self.train_loader)}")
        print(f"  Local Epochs: {self.config['federated']['local_epochs']}")
        print(f"  Device: {self.device}")
        
        try:
            # Set shared parameters from server
            self.set_parameters(parameters)

            # Setup optimizer with warmup
            lr = float(self.config['federated']['learning_rate'])
            weight_decay = float(self.config['training']['weight_decay'])
            
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

            # Learning rate scheduler with warmup
            num_epochs = self.config['federated']['local_epochs']
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr * 3,  # Peak at 3x base LR
                epochs=num_epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3  # 30% warmup
            )

            # Training loop with verbose metrics
            self.model.train()
            total_loss = 0.0
            total_dice = 0.0
            best_epoch_dice = 0.0
            num_batches = 0

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                epoch_dice = 0.0
                epoch_class_dice = {i: 0.0 for i in range(self.config['dataset']['num_classes'])}

                for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    
                    # Forward pass with mixed precision if available
                    if self.device.type == 'cuda' and self.config['hardware'].get('mixed_precision', False):
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(batch_x)
                            loss = self.criterion(outputs, batch_y)
                    else:
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Compute batch metrics for monitoring
                    with torch.no_grad():
                        pred = torch.argmax(outputs, dim=1)
                        batch_dice = compute_dice_score(pred, batch_y, self.config['dataset']['num_classes'])
                        epoch_dice += batch_dice

                        # Per-class dice tracking
                        for c in range(self.config['dataset']['num_classes']):
                            pred_c = (pred == c).float()
                            target_c = (batch_y == c).float()
                            intersection = (pred_c * target_c).sum()
                            union = pred_c.sum() + target_c.sum()
                            if union > 0:
                                epoch_class_dice[c] += (2.0 * intersection / (union + 1e-6)).item()
                    
                    # Verbose batch output
                    if self.config['logging'].get('print_every_batch', False) or batch_idx % 5 == 0:
                        print(f"     Batch {batch_idx+1}/{len(self.train_loader)} - "
                              f"Loss: {loss.item():.4f}, Dice: {batch_dice:.4f}, "
                              f"LR: {scheduler.get_last_lr()[0]:.6f}")

                # Epoch summary
                avg_loss = epoch_loss / len(self.train_loader)
                avg_dice = epoch_dice / len(self.train_loader)
                total_loss += avg_loss
                total_dice += avg_dice
                
                if avg_dice > best_epoch_dice:
                    best_epoch_dice = avg_dice
                
                # Per-class dice for epoch
                class_dice_str = ", ".join([f"C{i}:{epoch_class_dice[i]/len(self.train_loader):.3f}" 
                                           for i in range(1, self.config['dataset']['num_classes'])])
                
                print(f"\n   [Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}, "
                      f"Dice: {avg_dice:.4f} ({avg_dice*100:.1f}%)")
                print(f"   Class Dice: {class_dice_str}")

            # Update task history
            self.task_history.append(self.current_task)
            
            final_loss = total_loss / num_epochs
            final_dice = total_dice / num_epochs
            
            print(f"\n[OK] Client {self.client_id} training complete")
            print(f"     Final Loss: {final_loss:.4f}")
            print(f"     Mean Dice: {final_dice:.4f} ({final_dice*100:.1f}%)")
            print(f"     Best Epoch Dice: {best_epoch_dice:.4f} ({best_epoch_dice*100:.1f}%)")
            print(f"{'='*50}\n")
            
            # Track best dice
            if best_epoch_dice > self.best_dice:
                self.best_dice = best_epoch_dice

            return self.get_parameters(config), len(self.train_loader.dataset), {
                "loss": float(final_loss),
                "dice": float(final_dice),
                "best_dice": float(best_epoch_dice),
                "client_id": self.client_id
            }

        except Exception as e:
            print(f"[ERROR] Client {self.client_id} training failed: {e}")
            import traceback
            traceback.print_exc()
            return parameters, 0, {"loss": float('inf'), "error": str(e)}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Local evaluation with comprehensive multi-class metrics and verbose output"""
        print(f"\n{'='*50}")
        print(f"[EVAL] Client {self.client_id} - Evaluation")
        print(f"{'='*50}")
        print(f"  Validation samples: {len(self.val_loader.dataset)}")
        
        try:
            # Set parameters from server
            self.set_parameters(parameters)
            
            self.model.eval()
            
            total_loss = 0.0
            all_dice_scores = []
            all_hd95_scores = []
            class_dice_scores = {i: [] for i in range(self.config['dataset']['num_classes'])}
            
            # Additional metrics
            all_precision = []
            all_recall = []
            all_specificity = []

            with torch.no_grad():
                for batch_idx, (batch_x, batch_y) in enumerate(self.val_loader):
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    # Forward pass
                    if self.device.type == 'cuda' and self.config['hardware'].get('mixed_precision', False):
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(batch_x)
                    else:
                    outputs = self.model(batch_x)
                    
                    loss = self.criterion(outputs, batch_y)
                    total_loss += loss.item() * batch_x.size(0)

                    # Predictions
                    pred = torch.argmax(outputs, dim=1)
                    
                    # Compute metrics per sample
                    for i in range(batch_x.size(0)):
                        pred_i = pred[i].cpu().numpy()
                        target_i = batch_y[i].cpu().numpy()
                        
                        # Overall Dice (tumor classes only)
                        dice = compute_dice_per_sample(pred_i, target_i, self.config['dataset']['num_classes'])
                        all_dice_scores.append(dice)
                        
                        # Per-class Dice
                        for c in range(self.config['dataset']['num_classes']):
                            class_dice = compute_single_class_dice(pred_i, target_i, c)
                            class_dice_scores[c].append(class_dice)
                        
                        # HD95 (for tumor regions only)
                        hd95 = compute_hd95(pred_i, target_i)
                        all_hd95_scores.append(hd95)
                        
                        # Precision, Recall, Specificity for tumor regions
                        pred_tumor = pred_i > 0
                        target_tumor = target_i > 0
                        
                        tp = np.sum(pred_tumor & target_tumor)
                        fp = np.sum(pred_tumor & ~target_tumor)
                        fn = np.sum(~pred_tumor & target_tumor)
                        tn = np.sum(~pred_tumor & ~target_tumor)
                        
                        precision = tp / (tp + fp + 1e-6)
                        recall = tp / (tp + fn + 1e-6)
                        specificity = tn / (tn + fp + 1e-6)
                        
                        all_precision.append(precision)
                        all_recall.append(recall)
                        all_specificity.append(specificity)
                    
                    if batch_idx % 3 == 0:
                        batch_dice = np.mean([compute_dice_per_sample(
                            pred[j].cpu().numpy(), 
                            batch_y[j].cpu().numpy(), 
                            self.config['dataset']['num_classes']
                        ) for j in range(batch_x.size(0))])
                        print(f"     Batch {batch_idx+1}/{len(self.val_loader)} - "
                              f"Loss: {loss.item():.4f}, Dice: {batch_dice:.4f}")

            # Aggregate metrics
            num_samples = len(self.val_loader.dataset)
            avg_loss = total_loss / max(num_samples, 1)
            avg_dice = float(np.mean(all_dice_scores)) if all_dice_scores else 0.0
            avg_hd95 = float(np.mean([h for h in all_hd95_scores if h < 100])) if all_hd95_scores else 50.0
            avg_precision = float(np.mean(all_precision)) if all_precision else 0.0
            avg_recall = float(np.mean(all_recall)) if all_recall else 0.0
            avg_specificity = float(np.mean(all_specificity)) if all_specificity else 0.0
            
            # Per-class dice
            class_dice_means = {}
            for i, scores in class_dice_scores.items():
                class_dice_means[f"dice_class_{i}"] = float(np.mean(scores)) if scores else 0.0

            metrics = {
                "loss": float(avg_loss),
                "dice": avg_dice,
                "hd95": avg_hd95,
                "precision": avg_precision,
                "recall": avg_recall,
                "specificity": avg_specificity,
                "client_id": self.client_id,
                **class_dice_means
            }

            # Verbose output
            print(f"\n   [Evaluation Results]")
            print(f"     Loss: {avg_loss:.4f}")
            print(f"     Dice Score: {avg_dice:.4f} ({avg_dice*100:.1f}%)")
            print(f"     HD95: {avg_hd95:.2f}mm")
            print(f"     Precision: {avg_precision:.4f} ({avg_precision*100:.1f}%)")
            print(f"     Recall/Sensitivity: {avg_recall:.4f} ({avg_recall*100:.1f}%)")
            print(f"     Specificity: {avg_specificity:.4f} ({avg_specificity*100:.1f}%)")
            print(f"   [Per-Class Dice]")
            for c in range(1, self.config['dataset']['num_classes']):
                class_name = self.config['dataset']['class_names'][c]
                dice_val = class_dice_means.get(f"dice_class_{c}", 0)
                print(f"     {class_name}: {dice_val:.4f} ({dice_val*100:.1f}%)")
            
            # Check target achievement
            if avg_dice >= 0.70:
                print(f"\n   [SUCCESS] TARGET ACHIEVED! Dice >= 70%")
            else:
                gap = 0.70 - avg_dice
                print(f"\n   [INFO] Gap to target: {gap*100:.1f}%")
            
            print(f"{'='*50}\n")

            return float(avg_loss), num_samples, metrics

        except Exception as e:
            print(f"[ERROR] Client {self.client_id} evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf'), 0, {"loss": float('inf'), "error": str(e)}

    def adapt_to_task(self, task_id: int):
        """Adapt client to new continual learning task"""
        self.current_task = task_id
        
        if self.config['model']['use_adapters']:
            # Freeze shared parameters, train only adapters
            self.model.freeze_shared_parameters()
            self.model.unfreeze_adapters()
            print(f"[ADAPT] Client {self.client_id} adapting to task {task_id} (adapters only)")


class MultiClassDiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy Loss for multi-class segmentation"""

    def __init__(self, num_classes: int, dice_weight: float = 1.0, ce_weight: float = 0.5, 
                 smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Compute mean Dice score across all classes"""
    dice_scores = []
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice_c = (2.0 * intersection) / (union + 1e-6)
            dice_scores.append(dice_c.item())
    
    return np.mean(dice_scores) if dice_scores else 0.0


def compute_dice_per_sample(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    """Compute mean Dice for a single sample"""
    dice_scores = []
    
    for c in range(1, num_classes):  # Skip background (class 0)
        pred_c = (pred == c)
        target_c = (target == c)
        
        intersection = np.logical_and(pred_c, target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice_c = (2.0 * intersection) / (union + 1e-6)
            dice_scores.append(dice_c)
    
    return np.mean(dice_scores) if dice_scores else 0.0


def compute_single_class_dice(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """Compute Dice for a single class"""
    pred_c = (pred == class_id)
    target_c = (target == class_id)
    
    intersection = np.logical_and(pred_c, target_c).sum()
    union = pred_c.sum() + target_c.sum()
    
    if union > 0:
        return (2.0 * intersection) / (union + 1e-6)
    return 1.0 if target_c.sum() == 0 else 0.0


def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Hausdorff Distance 95th percentile for tumor regions"""
    try:
        from scipy.ndimage import distance_transform_edt
        
        # Combine all tumor classes (1, 2, 3)
        pred_tumor = pred > 0
        target_tumor = target > 0
        
        if not pred_tumor.any() or not target_tumor.any():
            return 0.0 if not target_tumor.any() else 100.0
        
        # Distance transforms
        pred_dist = distance_transform_edt(~pred_tumor)
        target_dist = distance_transform_edt(~target_tumor)
        
        # Surface distances
        pred_surface = pred_dist[target_tumor]
        target_surface = target_dist[pred_tumor]
        
        if len(pred_surface) == 0 or len(target_surface) == 0:
            return 100.0
        
        # 95th percentile
        hd95 = max(np.percentile(pred_surface, 95), np.percentile(target_surface, 95))
        
        return float(hd95)
        
    except Exception:
        return 50.0  # Default fallback


def create_client(client_id: str, data_dir: Path, config: dict = None) -> FederatedClient:
    """Factory function to create federated client
    
    Args:
        client_id: Unique identifier for the client
        data_dir: Path to client data directory
        config: Configuration dictionary. If None, will load from default paths.
    
    Returns:
        FederatedClient instance
    """
    if config is None:
        # Load configuration from default paths
        config_paths = ["configs/config.yaml", "../configs/config.yaml", "../../configs/config.yaml", 
                       "configs/config_15gb.yaml", "configs/config_32gb.yaml"]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                print(f"[CLIENT] Loaded config from {config_path}")
            break
    
    if config is None:
        raise FileNotFoundError("Could not find config.yaml")

    return FederatedClient(client_id, data_dir, config)


if __name__ == "__main__":
    # Test client creation
    import argparse

    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--client_id", type=str, required=True, help="Client identifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to client data directory")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Server address")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    client = create_client(args.client_id, data_dir)

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
        grpc_max_message_length=1024*1024*1024  # 1GB
    )
