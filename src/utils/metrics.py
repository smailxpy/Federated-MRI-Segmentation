#!/usr/bin/env python3
"""
Evaluation Metrics for Brain Tumor Segmentation
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This module implements comprehensive evaluation metrics for medical image segmentation.
Pure PyTorch implementation - no external dependencies required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class SegmentationMetrics:
    """Comprehensive segmentation evaluation metrics (Pure PyTorch)"""

    def __init__(self, num_classes: int = 4, reduction: str = "mean"):
        self.num_classes = num_classes
        self.reduction = reduction

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Calculate all metrics"""
        return self.compute_metrics(pred, target)

    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive segmentation metrics"""

        # Ensure correct shapes
        if pred.dim() == 5:  # [B, C, D, H, W]
            pred = torch.argmax(pred, dim=1)  # [B, D, H, W]
        if target.dim() == 5:
            target = target.squeeze(1)  # [B, D, H, W]

        metrics = {}

        # Dice Coefficient
        dice_scores = self._compute_dice(pred, target)
        metrics.update(dice_scores)

        # Hausdorff Distance 95
        hd_scores = self._compute_hd95(pred, target)
        metrics.update(hd_scores)

        # Additional metrics
        precision, recall, specificity = self._compute_prs_metrics(pred, target)
        metrics.update({
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        })
        
        # IoU metrics
        iou_scores = self._compute_iou(pred, target)
        metrics.update(iou_scores)

        return metrics

    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute Dice coefficient for each class"""

        dice_scores = {}

        for class_idx in range(1, self.num_classes):
            dice_score = self._dice_coefficient_binary(pred == class_idx, target == class_idx)
            dice_scores[f'dice_class_{class_idx}'] = dice_score

        dice_scores['dice_mean'] = np.mean([v for k, v in dice_scores.items() if k != 'dice_mean'])

        return dice_scores
    
    def _compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute IoU (Jaccard Index) for each class"""
        
        iou_scores = {}
        
        for class_idx in range(1, self.num_classes):
            iou_score = self._iou_binary(pred == class_idx, target == class_idx)
            iou_scores[f'iou_class_{class_idx}'] = iou_score
        
        iou_scores['iou_mean'] = np.mean([v for k, v in iou_scores.items() if k != 'iou_mean'])
        
        return iou_scores

    def _compute_hd95(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute Hausdorff Distance 95th percentile"""

        hd_scores = {}

        for class_idx in range(1, self.num_classes):
            hd_score = self._compute_hd95_binary(pred == class_idx, target == class_idx)
            hd_scores[f'hd95_class_{class_idx}'] = hd_score

        valid_scores = [v for k, v in hd_scores.items() if k != 'hd95_mean' and not np.isinf(v)]
        hd_scores['hd95_mean'] = np.mean(valid_scores) if valid_scores else float('inf')

        return hd_scores

    def _compute_prs_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
        """Compute Precision, Recall, and Specificity"""

        pred_flat = pred.flatten()
        target_flat = target.flatten()

        # Remove background class for calculation
        mask = target_flat > 0
        if mask.sum() == 0:
            return 0.0, 0.0, 0.0

        pred_flat = pred_flat[mask]
        target_flat = target_flat[mask]

        # Convert to binary (tumor vs background)
        pred_binary = (pred_flat > 0).float()
        target_binary = (target_flat > 0).float()

        # Calculate metrics
        tp = (pred_binary * target_binary).sum().float()
        fp = (pred_binary * (1 - target_binary)).sum().float()
        fn = ((1 - pred_binary) * target_binary).sum().float()
        tn = ((1 - pred_binary) * (1 - target_binary)).sum().float()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)

        return precision.item(), recall.item(), specificity.item()

    def _dice_coefficient_binary(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Dice coefficient for binary segmentation"""
        pred = pred.float()
        target = target.float()

        intersection = (pred * target).sum()
        dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)

        return dice.item()
    
    def _iou_binary(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute IoU for binary segmentation"""
        pred = pred.float()
        target = target.float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = intersection / (union + 1e-6)
        
        return iou.item()

    def _compute_hd95_binary(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute HD95 for binary segmentation using scipy"""
        try:
            from scipy.ndimage import distance_transform_edt
            
            pred_np = pred.cpu().numpy().astype(bool)
            target_np = target.cpu().numpy().astype(bool)
            
            if pred_np.sum() == 0 or target_np.sum() == 0:
                return float('inf')
            
            # Distance transform from prediction to target
            dist_pred_to_target = distance_transform_edt(~target_np)
            dist_target_to_pred = distance_transform_edt(~pred_np)
            
            # Get surface distances
            pred_surface = pred_np & ~np.roll(pred_np, 1, axis=0)
            target_surface = target_np & ~np.roll(target_np, 1, axis=0)
            
            if pred_surface.sum() == 0 or target_surface.sum() == 0:
                return float('inf')
            
            distances_pred_to_target = dist_pred_to_target[pred_surface]
            distances_target_to_pred = dist_target_to_pred[target_surface]
            
            hd95 = max(
                np.percentile(distances_pred_to_target, 95) if len(distances_pred_to_target) > 0 else 0,
                np.percentile(distances_target_to_pred, 95) if len(distances_target_to_pred) > 0 else 0
            )
            
            return float(hd95)
            
        except ImportError:
            # Fallback: approximate HD95
            return self._approximate_hd95(pred, target)

    def _approximate_hd95(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Approximate HD95 (simplified implementation)"""
        pred_coords = torch.nonzero(pred, as_tuple=False)
        target_coords = torch.nonzero(target, as_tuple=False)

        if len(pred_coords) == 0 or len(target_coords) == 0:
            return float('inf')

        # Compute pairwise distances (simplified - sample for efficiency)
        distances = []
        sample_size = min(100, len(pred_coords))
        indices = torch.randperm(len(pred_coords))[:sample_size]
        
        for idx in indices:
            p = pred_coords[idx].float()
            dists = torch.sqrt(((target_coords.float() - p) ** 2).sum(dim=1))
            min_dist = dists.min()
            distances.append(min_dist.item())

        if distances:
            return float(np.percentile(distances, 95))
        else:
            return float('inf')


class ForgettingMetrics:
    """Continual learning forgetting metrics"""

    def __init__(self):
        self.task_metrics = {}
        self.forgetting_rates = {}

    def update_task_metrics(self, task_id: int, metrics: Dict[str, float]):
        """Update metrics for a specific task"""
        self.task_metrics[task_id] = {
            'metrics': metrics.copy(),
            'timestamp': datetime.now().isoformat()
        }

    def compute_forgetting_rate(self, current_task: int) -> Dict[str, float]:
        """Compute forgetting rate for previous tasks"""

        if current_task < 1:
            return {}

        forgetting_rates = {}

        for prev_task in range(current_task):
            if prev_task in self.task_metrics:
                prev_metrics = self.task_metrics[prev_task]['metrics']
                current_metrics = self.task_metrics[current_task]['metrics']

                # Compute forgetting for key metrics
                for metric_name in ['dice_mean', 'precision', 'recall']:
                    if metric_name in prev_metrics and metric_name in current_metrics:
                        prev_score = prev_metrics[metric_name]
                        current_score = current_metrics[metric_name]

                        # Forgetting rate (higher is worse)
                        forgetting = max(0, prev_score - current_score)
                        forgetting_rates[f'{metric_name}_forgetting_task_{prev_task}'] = forgetting

        # Average forgetting rate
        if forgetting_rates:
            avg_forgetting = np.mean(list(forgetting_rates.values()))
            forgetting_rates['avg_forgetting_rate'] = avg_forgetting

        self.forgetting_rates[current_task] = forgetting_rates
        return forgetting_rates

    def get_summary(self) -> Dict[str, float]:
        """Get summary of forgetting metrics"""
        summary = {}

        if self.forgetting_rates:
            all_forgetting = []
            for task_rates in self.forgetting_rates.values():
                all_forgetting.extend([v for k, v in task_rates.items() if 'forgetting' in k])

            if all_forgetting:
                summary['mean_forgetting_rate'] = np.mean(all_forgetting)
                summary['max_forgetting_rate'] = np.max(all_forgetting)
                summary['min_forgetting_rate'] = np.min(all_forgetting)

        return summary


class ExperimentLogger:
    """Logger for experiment results and metrics"""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_log = []
        self.forgetting_log = []

    def log_round_metrics(self, round_num: int, client_id: str, metrics: Dict[str, float],
                         task_id: int = 0):
        """Log metrics for a training/evaluation round"""

        log_entry = {
            'round': round_num,
            'client_id': client_id,
            'task_id': task_id,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        self.metrics_log.append(log_entry)

        # Save to file
        self._save_metrics_log()

    def log_forgetting_metrics(self, task_id: int, forgetting_rates: Dict[str, float]):
        """Log forgetting metrics"""

        log_entry = {
            'task_id': task_id,
            'forgetting_rates': forgetting_rates,
            'timestamp': datetime.now().isoformat()
        }

        self.forgetting_log.append(log_entry)

        # Save to file
        self._save_forgetting_log()

    def _save_metrics_log(self):
        """Save metrics log to JSON"""
        log_path = self.log_dir / "metrics_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)

    def _save_forgetting_log(self):
        """Save forgetting log to JSON"""
        log_path = self.log_dir / "forgetting_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.forgetting_log, f, indent=2)

    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive experiment report"""

        report = {
            'total_rounds': len(set([entry['round'] for entry in self.metrics_log])),
            'total_clients': len(set([entry['client_id'] for entry in self.metrics_log])),
            'total_tasks': len(set([entry['task_id'] for entry in self.metrics_log])),
            'metrics_summary': self._compute_metrics_summary(),
            'forgetting_summary': self._compute_forgetting_summary(),
            'generated_at': datetime.now().isoformat()
        }

        # Save report
        report_path = self.log_dir / "experiment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def _compute_metrics_summary(self) -> Dict[str, float]:
        """Compute summary statistics for metrics"""

        if not self.metrics_log:
            return {}

        # Aggregate metrics across all entries
        all_metrics = {}
        for entry in self.metrics_log:
            for metric_name, value in entry['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        summary = {}
        for metric_name, values in all_metrics.items():
            values_array = np.array(values)
            summary[f'{metric_name}_mean'] = float(np.mean(values_array))
            summary[f'{metric_name}_std'] = float(np.std(values_array))
            summary[f'{metric_name}_min'] = float(np.min(values_array))
            summary[f'{metric_name}_max'] = float(np.max(values_array))

        return summary

    def _compute_forgetting_summary(self) -> Dict[str, float]:
        """Compute summary of forgetting metrics"""

        if not self.forgetting_log:
            return {}

        all_forgetting = []
        for entry in self.forgetting_log:
            for rate_name, rate in entry['forgetting_rates'].items():
                if 'forgetting' in rate_name and isinstance(rate, (int, float)):
                    all_forgetting.append(rate)

        if all_forgetting:
            return {
                'avg_forgetting_rate': float(np.mean(all_forgetting)),
                'max_forgetting_rate': float(np.max(all_forgetting)),
                'forgetting_variance': float(np.var(all_forgetting))
            }

        return {}


# Utility functions
def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor,
                           num_classes: int) -> np.ndarray:
    """Compute confusion matrix"""
    pred = pred.flatten()
    target = target.flatten()

    mask = target < num_classes
    pred = pred[mask]
    target = target[mask]

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    indices = num_classes * target + pred
    confusion += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)

    return confusion.numpy()


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Calculate Intersection over Union (IoU) for each class"""
    confusion = compute_confusion_matrix(pred, target, num_classes)

    iou_scores = {}
    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp

        iou = tp / (tp + fp + fn + 1e-6)
        iou_scores[f'iou_class_{i}'] = float(iou)

    iou_scores['iou_mean'] = float(np.mean([v for k, v in iou_scores.items() if k != 'iou_mean']))

    return iou_scores


if __name__ == "__main__":
    # Test metrics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data
    batch_size, depth, height, width = 2, 32, 32, 32
    num_classes = 4

    pred_logits = torch.randn(batch_size, num_classes, depth, height, width).to(device)
    target = torch.randint(0, num_classes, (batch_size, depth, height, width)).to(device)

    # Initialize metrics
    metrics_computer = SegmentationMetrics(num_classes=num_classes)

    # Compute metrics
    results = metrics_computer.compute_metrics(pred_logits, target)

    print("[METRICS] Segmentation Metrics Test:")
    for metric, value in results.items():
        try:
            print(f"{metric}: {float(value):.4f}")
        except Exception:
            print(f"{metric}: {value}")

    print("[OK] Metrics computation test completed")
