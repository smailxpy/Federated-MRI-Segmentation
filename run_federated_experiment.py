#!/usr/bin/env python3
"""
Run Federated Continual Learning Experiment for BraTS2021 MRI Segmentation
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This script runs a complete federated learning experiment using the BraTS2021 dataset
with drift-aware adapters for continual learning across 4 virtual hospitals.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar
from flwr.server.strategy import FedAvg

from src.models.unet_adapters import create_model
from src.federated.client import FederatedClient, BraTS2021Dataset


class FederatedExperiment:
    """Main experiment orchestrator for federated learning on BraTS2021"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.experiment_dir = self._setup_experiment_dir()
        self.results = {
            'rounds': [],
            'metrics': [],
            'forgetting': []
        }
        
        print(f"🧪 Federated Learning Experiment")
        print(f"   Dataset: BraTS2021")
        print(f"   Hospitals: {self.config['federated']['num_clients']}")
        print(f"   Rounds: {self.config['federated']['num_rounds']}")
        print(f"   Local Epochs: {self.config['federated']['local_epochs']}")
        print(f"   Results: {self.experiment_dir}")

    def _setup_experiment_dir(self) -> Path:
        """Setup experiment output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(self.config['output']['results_dir']) / f"experiment_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(exp_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        # Create subdirectories
        (exp_dir / "models").mkdir(exist_ok=True)
        (exp_dir / "metrics").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        
        return exp_dir

    def run(self):
        """Run the federated learning experiment"""
        print("\n🚀 Starting Federated Learning Experiment...")
        
        # Initialize global model
        print("\n📐 Initializing global model...")
        global_model = create_model(self.config)
        initial_parameters = [p.detach().cpu().numpy() for p in global_model.get_shared_parameters()]
        
        # Setup client data directories
        hospitals = self.config['continual']['tasks']
        processed_dir = Path(self.config['dataset']['processed_dir'])
        
        # Verify data exists
        for hospital in hospitals:
            hospital_dir = processed_dir / hospital
            if not (hospital_dir / "train").exists():
                raise FileNotFoundError(f"Training data not found for {hospital}")
            print(f"   ✓ {hospital}: {len(list((hospital_dir / 'train').iterdir()))} train patients")

        # Create client factory function
        def client_fn(cid: str) -> fl.client.NumPyClient:
            hospital_name = hospitals[int(cid)]
            data_dir = processed_dir / hospital_name
            return FederatedClient(hospital_name, data_dir, self.config)

        # Define evaluation aggregation
        def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
            """Aggregate evaluation metrics from clients"""
            if not metrics:
                return {}
            
            total_samples = sum(num_samples for num_samples, _ in metrics)
            if total_samples == 0:
                return {}
            
            aggregated = {}
            metric_names = ['dice', 'hd95', 'loss']
            
            for name in metric_names:
                values = [m.get(name, 0) * n for n, m in metrics if name in m]
                if values:
                    aggregated[name] = sum(values) / total_samples
            
            return aggregated

        # Setup Flower strategy
        strategy = FedAvg(
            fraction_fit=self.config['federated']['fraction_fit'],
            fraction_evaluate=self.config['federated']['fraction_evaluate'],
            min_fit_clients=self.config['federated']['min_available_clients'],
            min_evaluate_clients=self.config['federated']['min_available_clients'],
            min_available_clients=self.config['federated']['min_available_clients'],
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters)
        )

        # Run simulation
        print(f"\n🔄 Running {self.config['federated']['num_rounds']} federated rounds...")
        
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=self.config['federated']['num_clients'],
            config=fl.server.ServerConfig(num_rounds=self.config['federated']['num_rounds']),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0}
        )

        # Process and save results
        self._process_results(history)
        
        print(f"\n✅ Experiment completed!")
        print(f"   Results saved to: {self.experiment_dir}")
        
        return history

    def _process_results(self, history):
        """Process and save experiment results"""
        print("\n📊 Processing results...")
        
        results = {
            'experiment_timestamp': datetime.now().isoformat(),
            'config': self.config,
            'training_history': {
                'losses_distributed': [(r, float(l)) for r, l in history.losses_distributed] if history.losses_distributed else [],
                'metrics_distributed': dict(history.metrics_distributed) if history.metrics_distributed else {}
            },
            'metrics_summary': {}
        }
        
        # Calculate summary metrics
        if history.losses_distributed:
            losses = [l for _, l in history.losses_distributed]
            results['metrics_summary']['final_loss'] = float(losses[-1]) if losses else None
            results['metrics_summary']['best_loss'] = float(min(losses)) if losses else None
            results['metrics_summary']['avg_loss'] = float(np.mean(losses)) if losses else None
        
        # Extract dice scores if available
        if history.metrics_distributed:
            for key, values in history.metrics_distributed.items():
                if 'dice' in key.lower():
                    dice_scores = [v for _, v in values]
                    results['metrics_summary']['final_dice'] = float(dice_scores[-1]) if dice_scores else None
                    results['metrics_summary']['best_dice'] = float(max(dice_scores)) if dice_scores else None
        
        # Save results
        results_path = self.experiment_dir / "final_report.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   Results saved to {results_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Dataset: BraTS2021 ({self.config['dataset'].get('max_patients', 'all')} patients)")
        print(f"Hospitals: {self.config['federated']['num_clients']}")
        print(f"Rounds: {self.config['federated']['num_rounds']}")
        print(f"Local Epochs: {self.config['federated']['local_epochs']}")
        
        if results['metrics_summary']:
            print(f"\nMetrics:")
            for key, value in results['metrics_summary'].items():
                if value is not None:
                    print(f"  {key}: {value:.4f}")
        
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with 2 rounds")
    
    args = parser.parse_args()
    
    # Override for quick test
    if args.quick:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['federated']['num_rounds'] = 2
        config['federated']['local_epochs'] = 1
        
        quick_config = "configs/config_quick_run.yaml"
        with open(quick_config, 'w') as f:
            yaml.dump(config, f)
        args.config = quick_config
        print("🏃 Running in quick mode (2 rounds, 1 epoch)")
    
    # Run experiment
    experiment = FederatedExperiment(args.config)
    experiment.run()


if __name__ == "__main__":
    main()



