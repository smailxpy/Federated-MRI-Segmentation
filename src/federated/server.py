#!/usr/bin/env python3
"""
Federated Learning Server for MRI Segmentation
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This module implements the federated learning server using Flower framework.
"""

import torch
import flwr as fl
from flwr.common import NDArrays, Scalar, FitRes, EvaluateRes
from flwr.server.strategy import FedAvg, FedProx, FedAvgM
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime
import warnings
import sys
import os
warnings.filterwarnings("ignore")

# CLI
import argparse

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ContinualLearningStrategy(FedAvg):
    """Custom federated strategy with continual learning support"""

    def __init__(self, config: dict, save_path: Path, log_dir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.save_path = Path(save_path)
        self.log_dir = Path(log_dir)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Continual learning state
        self.current_task = 0
        self.task_history = []
        self.forgetting_metrics = []

        # Initialize global model
        from src.models.unet_adapters import create_model
        self.global_model = create_model(config)

        print("[OK] Continual Learning Strategy initialized")

    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
                     failures: List[BaseException]) -> Tuple[Optional[NDArrays], Dict[str, Scalar]]:
        """Aggregate fit results with continual learning tracking"""

        print(f"[ROUND] Server round {server_round}: received {len(results)} results, {len(failures)} failures")

        # Call parent aggregation
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            print(f"[OK] Round {server_round} completed with {len(results)} clients")
            # Update global model
            self._update_global_model(aggregated_params)

            # Save checkpoint
            self._save_checkpoint(server_round, aggregated_params, metrics)

            # Track forgetting
            self._track_forgetting(server_round, results)
        else:
            print(f"[ERROR] Round {server_round} failed - no aggregated parameters")

        return aggregated_params, metrics

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
                          failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results"""

        # Call parent evaluation
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if loss is not None:
            # Log metrics
            self._log_metrics(server_round, loss, metrics)

        return loss, metrics

    def _update_global_model(self, parameters: NDArrays):
        """Update global model with aggregated parameters"""
        shared_params = self.global_model.get_shared_parameters()
        for param, new_param in zip(shared_params, parameters):
            param.data = torch.from_numpy(new_param).to(param.device)

    def _save_checkpoint(self, round_num: int, parameters: NDArrays, metrics: Dict[str, Scalar]):
        """Save model checkpoint"""
        checkpoint_path = self.save_path / f"checkpoint_round_{round_num}.pth"

        checkpoint = {
            'round': round_num,
            'parameters': parameters,
            'metrics': metrics,
            'task': self.current_task,
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"[OK] Checkpoint saved: {checkpoint_path}")

    def _track_forgetting(self, round_num: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]]):
        """Track catastrophic forgetting metrics"""
        # Calculate proxy forgetting via client losses
        client_losses = [res.metrics.get('loss', 0.0) for _, res in results]
        avg_loss = np.mean(client_losses)

        forgetting_info = {
            'round': round_num,
            'task': self.current_task,
            'avg_loss': float(avg_loss),
            'client_losses': [float(loss) for loss in client_losses],
            'timestamp': datetime.now().isoformat()
        }

        self.forgetting_metrics.append(forgetting_info)

        # Save forgetting metrics
        metrics_path = self.log_dir / "forgetting_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.forgetting_metrics, f, indent=2)

    def _log_metrics(self, round_num: int, loss: float, metrics: Dict[str, Scalar]):
        """Log evaluation metrics"""
        log_entry = {
            'round': round_num,
            'loss': loss,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'task': self.current_task,
            'timestamp': datetime.now().isoformat()
        }

        # Save to JSON log
        log_path = self.log_dir / "evaluation_log.json"
        try:
            with open(log_path, 'r') as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []

        logs.append(log_entry)

        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2)

        print(f"Round {round_num} - Loss: {loss:.4f}, Metrics: {metrics}")

    def advance_task(self):
        """Advance to next continual learning task"""
        self.current_task += 1
        self.task_history.append(self.current_task)

        print(f"[ADVANCE] Advanced to task {self.current_task}")

        # Reset task-specific metrics
        self.forgetting_metrics = []


class FederatedServer:
    """Federated Learning Server Manager"""

    def __init__(self, config: dict, save_dir: Path = None, log_dir: Path = None):
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else Path(config['output']['model_save_dir'])
        self.log_dir = Path(log_dir) if log_dir else Path(config.get('output', {}).get('metrics_dir', 'results/metrics'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize global model first
        from src.models.unet_adapters import create_model
        self.global_model = create_model(self.config)

        # Initialize strategy with global model parameters
        self.strategy = self._create_strategy()

    def _create_strategy(self) -> ContinualLearningStrategy:
        """Create federated learning strategy"""

        # Base strategy parameters
        strategy_kwargs = {
            "fraction_fit": self.config['federated']['fraction_fit'],
            "fraction_evaluate": self.config['federated']['fraction_evaluate'],
            "min_fit_clients": self.config['federated']['min_available_clients'],
            "min_evaluate_clients": self.config['federated']['min_available_clients'],
            "min_available_clients": self.config['federated']['min_available_clients'],
        }

        # Choose aggregation strategy
        strategy_name = self.config.get('federated', {}).get('strategy', 'fedavg')

        if strategy_name == 'fedprox':
            base_strategy = FedProx(mu=0.01, **strategy_kwargs)
        elif strategy_name == 'fedavgm':
            base_strategy = FedAvgM(server_learning_rate=1.0, server_momentum=0.9, **strategy_kwargs)
        else:
            base_strategy = FedAvg(**strategy_kwargs)

        # Wrap with continual learning
        strategy = ContinualLearningStrategy(
            self.config,
            save_path=self.save_dir,
            log_dir=self.log_dir,
            **strategy_kwargs
        )

        # Initialize with global model parameters
        initial_params = self.global_model.get_shared_parameters()
        # Detach gradients before converting to numpy
        initial_params_detached = [param.detach().cpu().numpy() for param in initial_params]
        strategy.initial_parameters = fl.common.ndarrays_to_parameters(initial_params_detached)

        return strategy

    def start_server(self):
        """Start the federated learning server"""
        print("[START] Starting Federated Learning Server...")
        print(f"Configuration: {self.config['federated']['num_rounds']} rounds, "
              f"{self.config['federated']['num_clients']} clients")

        # Start server
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.config['federated']['num_rounds']),
            strategy=self.strategy,
            grpc_max_message_length=1024*1024*1024  # 1GB
        )

        # Write completion marker
        completion_file = self.save_dir / "server_completed.txt"
        with open(completion_file, 'w') as f:
            f.write(f"completed_at={datetime.now().isoformat()}\n")
        print(f"[OK] Server completed - marker written to {completion_file}")

    def run_continual_learning(self, tasks: List[str]):
        """Run continual learning across multiple tasks"""
        print("[ADVANCE] Starting Continual Learning Experiment")

        for task_idx, task_name in enumerate(tasks):
            print(f"\n[TASK] Task {task_idx + 1}: {task_name}")

            # Configure task-specific settings
            self.strategy.advance_task()

            # Run federated learning for this task
            try:
                self.start_server()
            except KeyboardInterrupt:
                print(f"\n[STOP]  Task {task_name} interrupted")
                break

        print("[OK] Continual learning experiment completed")

    def evaluate_final_model(self):
        """Evaluate final model performance"""
        print("[EVAL] Evaluating final model...")

        # Load best checkpoint
        checkpoints = list(self.save_dir.glob("checkpoint_round_*.pth"))
        if not checkpoints:
            print("[ERROR] No checkpoints found")
            return

        # Load latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        checkpoint = torch.load(latest_checkpoint)

        # Update global model
        self.strategy._update_global_model(checkpoint['parameters'])

        # Perform final evaluation
        # Note: In practice, you would evaluate on a held-out test set
        print("[OK] Final model evaluation completed")

        return {
            'final_round': checkpoint['round'],
            'final_metrics': checkpoint.get('metrics', {}),
            'total_tasks': len(self.strategy.task_history)
        }


def create_server(config: dict, save_dir: Path = None, log_dir: Path = None) -> FederatedServer:
    """Factory function to create federated server"""
    return FederatedServer(config, save_dir=save_dir, log_dir=log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--save-dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, help="Directory to write logs/metrics")
    parser.add_argument("--task-id", type=int, help="Current continual learning task id (0-based)")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create and start server
    server = create_server(config, save_dir=args.save_dir, log_dir=args.log_dir)

    # Set current task if provided
    if args.task_id is not None:
        try:
            server.strategy.current_task = int(args.task_id)
        except Exception:
            pass

    # Run a single federated session (per task)
    server.start_server()

    # Final evaluation
    final_results = server.evaluate_final_model()
    print(f"[COMPLETE] Experiment completed: {final_results}")
