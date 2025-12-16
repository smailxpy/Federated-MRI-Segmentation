#!/usr/bin/env python3
"""
Federated Continual Learning Training Script
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This script orchestrates the complete federated continual learning experiment.
"""

import torch
import torch.multiprocessing as mp
from pathlib import Path
import yaml
import argparse
import subprocess
import signal
import sys
import os
from typing import Dict, List, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.federated.server import FederatedServer
class FederatedContinualLearningExperiment:
    """Main experiment orchestrator"""

    def __init__(self, config_path: str):
        self.config_path = config_path

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup directories
        self._setup_directories()

        # Initialize components
        self.logger = None
        self.server_process = None
        self.client_processes = []

        print("[OK] Federated Continual Learning Experiment initialized")

    def _setup_directories(self):
        """Setup experiment directories"""
        self.exp_dir = Path(self.config['output']['results_dir']) / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.exp_dir / "models").mkdir(exist_ok=True)
        (self.exp_dir / "logs").mkdir(exist_ok=True)
        (self.exp_dir / "plots").mkdir(exist_ok=True)
        (self.exp_dir / "metrics").mkdir(exist_ok=True)

        # Save configuration
        with open(self.exp_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f, indent=2)

        print(f"[OK] Experiment directory created: {self.exp_dir}")

    def prepare_data(self):
        """Prepare dataset for federated learning using BraTS2021"""
        print("[DOWNLOAD] Preparing dataset...")

        processed_dir = Path(self.config['dataset']['processed_dir'])

        # Check if processed BraTS2021 data already exists
        stats_file = processed_dir / "dataset_stats.json"
        hospital_a = processed_dir / "hospital_a"
        
        if hospital_a.exists() and stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            # Check if it's BraTS2021 data
            if stats.get('dataset_name') == 'BraTS2021':
                print("[OK] BraTS2021 processed dataset already exists")
                print(f"  Total patients: {stats.get('total_patients', 'N/A')}")
                return stats
            else:
                print("[WARNING]  Found non-BraTS2021 data, will reprocess...")

        # Check for BraTS2021 raw data
        brats_raw_dir = Path(self.config['dataset']['data_dir']) / "brats2021"
        if brats_raw_dir.exists():
            patient_dirs = list(brats_raw_dir.glob("BraTS2021_*"))
            if len(patient_dirs) > 0:
                print(f"[OK] Found BraTS2021 dataset with {len(patient_dirs)} patients")
                
                # Clear old processed data if exists
                if processed_dir.exists():
                    import shutil
                    for item in processed_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    print("  Cleared old processed data")
                
                # Process BraTS2021 dataset
                from src.data.process_brats2021 import BraTS2021Processor
                processor = BraTS2021Processor(self.config)
                
                # Process a subset for faster iteration (adjust as needed)
                max_patients = self.config.get('dataset', {}).get('max_patients', 1000)
                stats = processor.process_dataset(max_patients=max_patients)
                
                print("[OK] BraTS2021 dataset preparation completed")
                return stats
        
        # Fallback: Check for LGG dataset
        lgg_raw_dir = Path(self.config['dataset']['data_dir']) / "lgg-mri-segmentation" / "kaggle_3m"
        if lgg_raw_dir.exists():
            print("[WARNING]  BraTS2021 not found, using LGG dataset as fallback...")
            from src.data.process_lgg_dataset import LGGDatasetProcessor
            processor = LGGDatasetProcessor(self.config)
            stats = processor.process_dataset()
            return stats

        raise RuntimeError(
            "No dataset found! Please ensure BraTS2021 data is in data/raw/brats2021/\n"
            "Expected structure: data/raw/brats2021/BraTS2021_*/BraTS2021_*_{t1,t1ce,t2,flair,seg}.nii.gz"
        )

    def setup_federated_environment(self):
        """Setup federated learning environment"""
        print("[SETUP] Setting up federated environment...")

        # Create client data directories
        processed_dir = Path(self.config['dataset']['processed_dir'])
        num_clients = self.config['federated']['num_clients']

        client_dirs = []
        for i in range(num_clients):
            # Map client to hospital (simulating different hospitals)
            hospital_id = chr(97 + (i % 4))  # a, b, c, d cycle
            hospital_dir = processed_dir / f"hospital_{hospital_id}"

            # Check if hospital directory has proper train/val structure
            if hospital_dir.exists():
                train_dir = hospital_dir / "train"
                val_dir = hospital_dir / "val"
                
                if train_dir.exists() and val_dir.exists():
                    # Use hospital directory directly as client data (already has train/val)
                    print(f"  Client {i} -> hospital_{hospital_id} (using existing train/val)")
                    client_dirs.append(hospital_dir)
                else:
                    # Old structure - need to create train/val splits
                    client_dir = processed_dir / f"client_{i}"
                    client_dir.mkdir(exist_ok=True)
                self._create_client_splits(hospital_dir, client_dir)
                    client_dirs.append(client_dir)
            else:
                print(f"[WARNING] Hospital directory not found: {hospital_dir}")
                # Fallback: create client directory and copy from hospital
                client_dir = processed_dir / f"client_{i}"
                client_dir.mkdir(exist_ok=True)
            client_dirs.append(client_dir)

        print(f"[OK] Configured {len(client_dirs)} client data directories")
        for i, cd in enumerate(client_dirs):
            train_count = len(list((cd / "train").glob("BraTS*"))) if (cd / "train").exists() else 0
            val_count = len(list((cd / "val").glob("BraTS*"))) if (cd / "val").exists() else 0
            print(f"     Client {i}: {train_count} train, {val_count} val samples")
        
        return client_dirs

    def _create_client_splits(self, hospital_dir: Path, client_dir: Path):
        """Create train/validation splits for a client"""
        import shutil

        print(f"[DEBUG] Creating client splits from {hospital_dir} to {client_dir}")

        # Get all patient directories
        patient_dirs = [d for d in hospital_dir.iterdir() if d.is_dir() and d.name.startswith('BraTS2021_')]
        print(f"[DEBUG] Found {len(patient_dirs)} patient directories")
        if not patient_dirs:
            print("[WARNING] No patient directories found!")
            return

        # Simple split: 80% train, 20% validation
        train_split = int(0.8 * len(patient_dirs))
        train_patients = patient_dirs[:train_split]
        val_patients = patient_dirs[train_split:]

        print(f"[DEBUG] Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")

        # Create subdirectories
        train_dir = client_dir / "train"
        val_dir = client_dir / "val"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)

        print(f"[DEBUG] Created train dir: {train_dir}, val dir: {val_dir}")

        # Copy patient directories
        copied_train = 0
        for patient_dir in train_patients:
            dest_dir = train_dir / patient_dir.name
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(patient_dir, dest_dir)
            copied_train += 1

        copied_val = 0
        for patient_dir in val_patients:
            dest_dir = val_dir / patient_dir.name
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(patient_dir, dest_dir)
            copied_val += 1

        print(f"[DEBUG] Copied {copied_train} train and {copied_val} val patients")

    def run_experiment(self):
        """Run the complete federated continual learning experiment"""
        print("[START] Starting Federated Continual Learning Experiment")

        try:
            # Prepare data
            dataset_stats = self.prepare_data()

            # Setup federated environment
            client_dirs = self.setup_federated_environment()

            # Initialize logger
            from src.utils.metrics import ExperimentLogger
            self.logger = ExperimentLogger(self.exp_dir / "logs")

            # Run continual learning tasks
            tasks = self.config['continual']['tasks']
            for task_idx, task_name in enumerate(tasks):
                print(f"\n[TASK] Task {task_idx + 1}/{len(tasks)}: {task_name}")

                # Run federated learning for this task
                self._run_task(task_idx, task_name, client_dirs)

                # Evaluate forgetting
                self._evaluate_task_performance(task_idx)

            # Generate final report
            self._generate_final_report()

            print("[SUCCESS] Experiment completed successfully!")

        except Exception as e:
            print(f"[ERROR] Experiment failed: {e}")
            self._cleanup()
            raise

    def _run_task(self, task_idx: int, task_name: str, client_dirs: List[Path]):
        """Run federated learning for a single task"""

        print(f"[RUNNING] Running federated learning for task: {task_name}")

        # Use Flower simulation for more reliable federated learning
        self._run_simulation(task_idx, task_name, client_dirs)

        print(f"[OK] Task {task_name} completed")

    def _run_simulation(self, task_idx: int, task_name: str, client_dirs: List[Path]):
        """Run federated learning using Flower simulation with verbose output"""
        import flwr as fl
        import torch

        # Create server
        server = FederatedServer(self.config)
        strategy = server.strategy

        # Determine GPU availability
        use_gpu = torch.cuda.is_available()
        gpu_count = 1 if use_gpu else 0
        device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
        
        print(f"\n{'='*60}")
        print(f"[SIMULATION] Starting Federated Continual Learning Simulation")
        print(f"{'='*60}")
        print(f"  Task: {task_name} (ID: {task_idx})")
        print(f"  Clients: {len(client_dirs)}")
        print(f"  Rounds: {self.config['federated']['num_rounds']}")
        print(f"  Local Epochs: {self.config['federated']['local_epochs']}")
        print(f"  Batch Size: {self.config['federated']['batch_size']}")
        print(f"  Learning Rate: {self.config['federated']['learning_rate']}")
        print(f"  Device: {device_name}")
        print(f"{'='*60}\n")

        try:
            # Test client creation first
            print("[INIT] Testing client creation...")
            test_client = None
            try:
                from src.federated.client import create_client
                test_client = create_client("test_0", client_dirs[0], self.config)
                print(f"[OK] Client creation successful - Model has {sum(p.numel() for p in test_client.model.parameters()):,} parameters")
                del test_client  # Free memory
                torch.cuda.empty_cache() if use_gpu else None
            except Exception as ce:
                print(f"[ERROR] Client creation failed: {ce}")
                import traceback
                traceback.print_exc()
                raise ce

            # Create client function for simulation
            def client_fn(cid: str):
                try:
                    from src.federated.client import create_client
                    client_dir = client_dirs[int(cid)]
                    client = create_client(cid, client_dir, self.config)
                    return client
                except Exception as ce:
                    print(f"[ERROR] Failed to create client {cid}: {ce}")
                    raise ce

            # Use Flower simulation API
            print(f"\n[TRAIN] Starting training for {self.config['federated']['num_rounds']} rounds...")
            print("-" * 60)
            
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(client_dirs),
                config=fl.server.ServerConfig(num_rounds=self.config['federated']['num_rounds']),
                strategy=strategy,
                client_resources={"num_cpus": 2, "num_gpus": gpu_count * 0.25},  # Share GPU
            )

            print("-" * 60)
            print(f"\n[OK] Simulation completed!")
            
            # Extract and display results
            self._display_training_summary(history)

            # Save simulation results with checkpointing
            self._save_simulation_results(task_idx, history)
            
            # Save best model checkpoint
            self._save_best_checkpoint(task_idx, strategy)

        except Exception as e:
            print(f"\n[ERROR] Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            print("\n[FALLBACK] Attempting subprocess fallback method...")
            self._run_subprocess_fallback(task_idx, task_name, client_dirs)
    
    def _display_training_summary(self, history):
        """Display a summary of training results"""
        print(f"\n{'='*60}")
        print("[SUMMARY] Training Results Summary")
        print(f"{'='*60}")
        
        # Extract metrics from history
        if hasattr(history, 'losses_distributed') and history.losses_distributed:
            losses = [loss for _, loss in history.losses_distributed]
            print(f"  Final Loss: {losses[-1]:.4f}")
            print(f"  Best Loss: {min(losses):.4f}")
            print(f"  Loss Improvement: {losses[0] - losses[-1]:.4f}")
        
        if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
            # Get final round metrics
            final_metrics = history.metrics_distributed.get('dice', [])
            if final_metrics:
                final_dice = final_metrics[-1][1] if final_metrics else 0
                print(f"  Final Dice Score: {final_dice:.4f}")
                if final_dice >= 0.70:
                    print(f"  [SUCCESS] Target Dice > 70% ACHIEVED!")
                else:
                    print(f"  [INFO] Current Dice: {final_dice*100:.1f}% (target: 70%)")
        
        if hasattr(history, 'metrics_centralized') and history.metrics_centralized:
            print(f"  Centralized Metrics Rounds: {len(history.metrics_centralized)}")
            for round_num, metrics in history.metrics_centralized[-3:]:  # Last 3 rounds
                print(f"    Round {round_num}: {metrics}")
        
        print(f"{'='*60}\n")
    
    def _save_best_checkpoint(self, task_idx: int, strategy):
        """Save the best model checkpoint"""
        try:
            checkpoint_dir = self.exp_dir / "models"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Get model state from strategy
            if hasattr(strategy, 'global_model'):
                import torch
                model_path = checkpoint_dir / f"best_model_task_{task_idx}.pth"
                torch.save({
                    'task_idx': task_idx,
                    'model_state_dict': strategy.global_model.state_dict(),
                    'config': self.config,
                    'timestamp': datetime.now().isoformat()
                }, model_path)
                print(f"[CHECKPOINT] Best model saved: {model_path}")
            
            # Also save a completion marker
            marker_path = checkpoint_dir / f"task_{task_idx}_completed.txt"
            with open(marker_path, 'w') as f:
                f.write(f"task={task_idx}\n")
                f.write(f"completed_at={datetime.now().isoformat()}\n")
                f.write(f"status=success\n")
            
        except Exception as e:
            print(f"[WARNING] Could not save checkpoint: {e}")

    def _run_subprocess_fallback(self, task_idx: int, task_name: str, client_dirs: List[Path]):
        """Fallback to subprocess method if simulation fails"""
        # Start clients first (they will wait for server)
        self._start_clients(client_dirs)

        # Give clients a moment to start
        import time
        time.sleep(2)

        # Start server
        self._start_server(task_idx)

        # Wait for completion
        self._wait_for_completion(task_idx)

        # Cleanup processes
        self._cleanup_processes()

    def _save_simulation_results(self, task_idx: int, history):
        """Save simulation results to logs"""
        # Convert history to our log format
        for round_num, metrics in enumerate(history.metrics_centralized):
            log_entry = {
                'round': round_num,
                'client_id': 'simulation',
                'task_id': task_idx,
                'metrics': {k: float(v) for k, v in metrics.items()},
                'timestamp': datetime.now().isoformat()
            }

            # Log using our logger
            if self.logger:
                self.logger.log_round_metrics(round_num, 'simulation', log_entry['metrics'], task_idx)

    def _start_server(self, task_idx: int):
        """Start federated learning server"""
        print("[SERVER]  Starting federated server...")

        cmd = [
            sys.executable, "src/federated/server.py",
            "--config", str(self.config_path),
            "--save-dir", str(self.exp_dir / "models"),
            "--log-dir", str(self.exp_dir / "logs"),
            "--task-id", str(task_idx)
        ]

        # Start server process
        self.server_process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("[OK] Server started")

    def _start_clients(self, client_dirs: List[Path]):
        """Start federated clients"""
        print(f"[CLIENTS] Starting {len(client_dirs)} clients...")

        for i, client_dir in enumerate(client_dirs):
            cmd = [
                sys.executable, "src/federated/client.py",
                "--client_id", f"client_{i}",
                "--data_dir", str(client_dir)
            ]

            # Start client process
            client_process = subprocess.Popen(
                cmd,
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            self.client_processes.append(client_process)

        print(f"[OK] {len(client_dirs)} clients started")

    def _wait_for_completion(self, task_idx: int):
        """Wait for federated learning round to complete"""
        import time

        completion_file = self.exp_dir / "models" / f"task_{task_idx}_completed.txt"
        server_completion_file = self.exp_dir / "models" / "server_completed.txt"
        timeout_seconds = 600  # 10 minutes max
        start_time = time.time()
        print(f"[WAITING] Waiting for task {task_idx} completion (timeout ~{timeout_seconds}s)...")

        while True:
            # Check if server wrote completion marker
            if server_completion_file.exists():
                print("[OK] Server completed federated learning")
                # Write task completion marker
                with open(completion_file, 'w') as f:
                    f.write(f"task_{task_idx}_completed_at={datetime.now().isoformat()}\n")
                return True

            # Fallback: check if processes are done (but this might not work reliably)
            server_done = (self.server_process.poll() is not None) if self.server_process else True
            clients_done = all(p.poll() is not None for p in self.client_processes)

            if server_done and clients_done:
                print("[OK] All processes finished")
                # Write task completion marker
                with open(completion_file, 'w') as f:
                    f.write(f"task_{task_idx}_completed_at={datetime.now().isoformat()}\n")
                return True

            if time.time() - start_time > timeout_seconds:
                print("[WARNING]  Timeout reached, proceeding to cleanup")
                return False

            time.sleep(5)

    def _cleanup_processes(self):
        """Cleanup server and client processes"""
        # Terminate server
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            print("[OK] Server terminated")

        # Terminate clients
        for client_process in self.client_processes:
            client_process.terminate()
            client_process.wait()
        print("[OK] Clients terminated")

        self.server_process = None
        self.client_processes = []

    def _evaluate_task_performance(self, task_idx: int):
        """Evaluate performance and forgetting for current task"""
        print(f"[METRICS] Evaluating task {task_idx} performance...")

        # Load metrics from logs
        eval_log_path = self.exp_dir / "logs" / "evaluation_log.json"
        if eval_log_path.exists():
            try:
                with open(eval_log_path, 'r') as f:
                    eval_logs = json.load(f)
            except json.JSONDecodeError:
                eval_logs = []

            # Filter entries for this task if present
            task_entries = [e for e in eval_logs if int(e.get('task', -1)) == task_idx]
            if not task_entries:
                # Fallback: if task field isn't present, use all entries
                task_entries = eval_logs

            if task_entries:
                # Aggregate metrics
                metric_lists = {}
                for entry in task_entries:
                    for k, v in entry.get('metrics', {}).items():
                        metric_lists.setdefault(k, []).append(float(v))

                final_metrics = {k: float(sum(v) / max(len(v), 1)) for k, v in metric_lists.items()}

                # Track forgetting across tasks using ExperimentLogger + ForgettingMetrics
                if self.logger and final_metrics:
                    from src.utils.metrics import ForgettingMetrics
                    forgetting_tracker = ForgettingMetrics()
                    forgetting_tracker.update_task_metrics(task_idx, final_metrics)
                    forgetting_rates = forgetting_tracker.compute_forgetting_rate(task_idx)
                    self.logger.log_forgetting_metrics(task_idx, forgetting_rates)

                print(f"[OK] Task {task_idx} evaluation completed")

    def _generate_final_report(self):
        """Generate comprehensive experiment report"""
        print("[TASK] Generating final experiment report...")

        if self.logger:
            # Build metrics summary from evaluation_log if available
            eval_log_path = self.exp_dir / "logs" / "evaluation_log.json"
            metrics_summary = {}
            forgetting_summary = {}

            if eval_log_path.exists():
                try:
                    with open(eval_log_path, 'r') as f:
                        eval_logs = json.load(f)
                except json.JSONDecodeError:
                    eval_logs = []

                # Aggregate all metrics across rounds/clients
                all_metrics = {}
                for entry in eval_logs:
                    for metric_name, value in entry.get('metrics', {}).items():
                        all_metrics.setdefault(metric_name, []).append(float(value))

                for metric_name, values in all_metrics.items():
                    arr = values
                    mean_val = float(sum(arr) / max(len(arr), 1))
                    metrics_summary[f'{metric_name}_mean'] = mean_val
                    # Backward-compatibility aliases (used in some readers)
                    if metric_name == 'dice':
                        metrics_summary['dice_mean_mean'] = mean_val
                    if metric_name == 'hd95':
                        metrics_summary['hd95_mean_mean'] = mean_val
                    # std/min/max optional
                    try:
                        import numpy as _np
                        std_val = float(_np.std(arr))
                        metrics_summary[f'{metric_name}_std'] = std_val
                        metrics_summary[f'{metric_name}_min'] = float(min(arr))
                        metrics_summary[f'{metric_name}_max'] = float(max(arr))
                        if metric_name == 'dice':
                            metrics_summary['dice_std_mean'] = std_val
                        if metric_name == 'hd95':
                            metrics_summary['hd95_std_mean'] = std_val
                    except Exception:
                        pass

            # Use ExperimentLogger to serialize report skeleton
            report = self.logger.generate_report()
            # Inject computed summaries if available
            if metrics_summary:
                report['metrics_summary'] = metrics_summary
            if forgetting_summary:
                report['forgetting_summary'] = forgetting_summary

            # Add experiment metadata
            report.update({
                'experiment_type': 'federated_continual_learning',
                'team': '314IV',
                'topic': '#6 Federated Continual Learning for MRI Segmentation',
                'dataset': 'BraTS2021',
                'model': 'U-Net with Drift-Aware Adapters',
                'federated_framework': 'Flower',
                'total_clients': self.config['federated']['num_clients'],
                'total_rounds': self.config['federated']['num_rounds'],
                'total_tasks': len(self.config['continual']['tasks'])
            })

            # Save enhanced report
            with open(self.exp_dir / "final_report.json", 'w') as f:
                json.dump(report, f, indent=2)

            print("[OK] Final report generated")
            return report

    def _cleanup(self):
        """Cleanup experiment resources"""
        self._cleanup_processes()

        if hasattr(self, 'exp_dir'):
            print(f"[SAVE] Experiment results saved to: {self.exp_dir}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()


def run_ablation_study(config_path: str):
    """Run ablation studies comparing different configurations"""
    print("[ABLATION] Running ablation studies...")

    base_config_path = Path(config_path)
    ablation_configs = [
        {"name": "baseline", "use_adapters": False},
        {"name": "adapters", "use_adapters": True},
        {"name": "fedprox", "strategy": "fedprox"},
        {"name": "fedavgm", "strategy": "fedavgm"}
    ]

    results = {}

    for ablation_config in ablation_configs:
        print(f"\n[ABLATION] Running ablation: {ablation_config['name']}")

        # Load and modify config
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Apply ablation changes
        if 'use_adapters' in ablation_config:
            config['model']['use_adapters'] = ablation_config['use_adapters']
        if 'strategy' in ablation_config:
            config['federated']['strategy'] = ablation_config['strategy']

        # Save modified config
        ablation_dir = Path(config['output']['results_dir']) / f"ablation_{ablation_config['name']}"
        ablation_dir.mkdir(parents=True, exist_ok=True)

        ablation_config_path = ablation_dir / "config.yaml"
        with open(ablation_config_path, 'w') as f:
            yaml.dump(config, f, indent=2)

        # Run experiment
        try:
            with FederatedContinualLearningExperiment(str(ablation_config_path)) as exp:
                exp.run_experiment()
                results[ablation_config['name']] = "success"
        except Exception as e:
            print(f"[ERROR] Ablation {ablation_config['name']} failed: {e}")
            results[ablation_config['name']] = f"failed: {e}"

    # Save ablation results
    ablation_results_path = Path(config['output']['results_dir']) / "ablation_results.json"
    with open(ablation_results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("[OK] Ablation studies completed")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Federated Continual Learning for MRI Segmentation")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--ablation", action="store_true",
                       help="Run ablation studies")
    parser.add_argument("--prepare-data", action="store_true",
                       help="Only prepare dataset")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick test (reduced rounds)")

    args = parser.parse_args()

    # Modify config for quick test
    if args.quick_test:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        config['federated']['num_rounds'] = 2
        config['federated']['local_epochs'] = 1

        # Save modified config
        quick_config_path = Path(args.config).parent / "config_quick.yaml"
        with open(quick_config_path, 'w') as f:
            yaml.dump(config, f, indent=2)

        args.config = str(quick_config_path)

    if args.ablation:
        run_ablation_study(args.config)
    elif args.prepare_data:
        exp = FederatedContinualLearningExperiment(args.config)
        exp.prepare_data()
    else:
        # Run main experiment
        with FederatedContinualLearningExperiment(args.config) as exp:
            exp.run_experiment()


if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Already set

    main()
