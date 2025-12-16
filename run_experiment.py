#!/usr/bin/env python3
"""
Complete Experiment Runner for Federated Continual Learning
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This script provides a complete pipeline from data preparation to final results.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json
from datetime import datetime


def print_header():
    """Print experiment header"""
    print("=" * 80)
    print("🎯 FEDERATED CONTINUAL LEARNING FOR MRI SEGMENTATION")
    print("=" * 80)
    print("Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation")
    print("Members: Ismoil Salohiddinov, Komiljon Qosimov, Abdurashid Djumabaev")
    print("=" * 80)


def run_command(cmd: list, description: str, cwd: str = None) -> bool:
    """Run command with proper output handling"""
    print(f"\n🔧 {description}")
    print("-" * 50)

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or ".",
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")

    checks = []

    # Check Python
    python_ok = sys.version_info >= (3, 8)
    print(f"🐍 Python 3.8+: {'✅' if python_ok else '❌'}")
    checks.append(python_ok)

    # Check required files
    files_to_check = [
        "requirements.txt",
        "configs/config.yaml",
        "kaggle.json"
    ]

    for file_path in files_to_check:
        exists = Path(file_path).exists()
        print(f"📄 {file_path}: {'✅' if exists else '❌'}")
        checks.append(exists)

    # Check installed packages (basic)
    try:
        import torch
        torch_ok = True
        print("📦 PyTorch: ✅")
    except ImportError:
        torch_ok = False
        print("📦 PyTorch: ❌")
    checks.append(torch_ok)

    success = all(checks)
    if success:
        print("✅ All prerequisites met")
    else:
        print("❌ Some prerequisites not met")
        print("Please run: python setup.py")

    return success


def prepare_dataset():
    """Prepare the BraTS dataset"""
    print("\n📥 PHASE 1: DATASET PREPARATION")
    print("=" * 50)

    cmd = [
        sys.executable, "src/data/download_dataset.py",
        "--download", "--extract", "--preprocess"
    ]

    success = run_command(cmd, "Downloading and preprocessing BraTS2021 dataset")
    if success:
        # Show dataset statistics
        run_command([
            sys.executable, "src/data/download_dataset.py", "--stats"
        ], "Computing dataset statistics")

    return success


def run_main_experiment():
    """Run the main federated continual learning experiment"""
    print("\n🚀 PHASE 2: FEDERATED CONTINUAL LEARNING EXPERIMENT")
    print("=" * 50)

    success = run_command([
        sys.executable, "src/experiments/train_fcl.py"
    ], "Running federated continual learning experiment")

    return success


def run_ablation_studies():
    """Run ablation studies to compare different configurations"""
    print("\n🔬 PHASE 3: ABLATION STUDIES")
    print("=" * 50)

    success = run_command([
        sys.executable, "src/experiments/train_fcl.py", "--ablation"
    ], "Running ablation studies comparing different FL strategies")

    return success


def generate_results():
    """Generate comprehensive results and visualizations"""
    print("\n📊 PHASE 4: RESULTS ANALYSIS & VISUALIZATION")
    print("=" * 50)

    # Find latest experiment directory
    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ No results directory found")
        return False

    exp_dirs = list(results_dir.glob("experiment_*"))
    if not exp_dirs:
        print("❌ No experiment directories found")
        return False

    latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 Analyzing results from: {latest_exp}")

    # Generate all visualizations
    success = run_command([
        sys.executable, "src/utils/visualization.py",
        "--results_dir", str(latest_exp),
        "--plot", "all"
    ], "Generating comprehensive visualizations")

    if success:
        # Generate final report
        run_command([
            sys.executable, "src/utils/visualization.py",
            "--results_dir", str(latest_exp)
        ], "Generating final experiment report")

    return success


def show_summary():
    """Show experiment summary"""
    print("\n🎉 EXPERIMENT COMPLETED!")
    print("=" * 50)

    # Find latest results
    results_dir = Path("results")
    if results_dir.exists():
        exp_dirs = list(results_dir.glob("experiment_*"))
        if exp_dirs:
            latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)

            # Load experiment report
            report_file = latest_exp / "final_report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report = json.load(f)

                print("📊 FINAL RESULTS SUMMARY:")
                print("-" * 30)

                metrics = report.get('metrics_summary', {})
                dice = metrics.get('dice_mean_mean')
                hd95 = metrics.get('hd95_mean_mean')
                precision = metrics.get('precision_mean')
                recall = metrics.get('recall_mean')
                if dice is not None:
                    print(f"Dice (mean): {dice:.4f}")
                if hd95 is not None:
                    print(f"HD95 (mean): {hd95:.4f} mm")
                if precision is not None:
                    print(f"Precision (mean): {precision:.4f}")
                if recall is not None:
                    print(f"Recall (mean): {recall:.4f}")

                forgetting = report.get('forgetting_summary', {})
                forget_rate = forgetting.get('avg_forgetting_rate')
                if forget_rate is not None:
                    status = "✅ GOOD" if forget_rate < 0.1 else "⚠️  HIGH"
                    print(f"Forgetting rate (avg): {forget_rate:.4f} {status}")
                print(f"\n📁 Complete results saved to: {latest_exp}")
                print("📈 View visualizations in: results/plots/")
    print("\n🏆 Team 314IV - Federated Continual Learning for MRI Segmentation")
    print("✅ All phases completed successfully!")


def cleanup_temp_files():
    """Clean up temporary files"""
    print("\n🧹 Cleaning up temporary files...")

    # Remove any temporary files if needed
    temp_patterns = [
        "**/*.pyc",
        "**/__pycache__",
        "**/.DS_Store"
    ]

    for pattern in temp_patterns:
        for temp_file in Path(".").glob(pattern):
            if temp_file.is_file():
                temp_file.unlink()
            elif temp_file.is_dir():
                import shutil
                shutil.rmtree(temp_file)

    print("✅ Cleanup completed")


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description="Complete Federated Continual Learning Experiment")
    parser.add_argument("--phases", nargs='+', choices=['data', 'experiment', 'ablation', 'results', 'all'],
                       default=['all'], help="Phases to run")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip prerequisite checks")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up temporary files after completion")

    args = parser.parse_args()

    print_header()

    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            print("❌ Prerequisites not met. Please run setup first.")
            return False

    success = True
    phases_run = []

    # Phase 1: Data Preparation
    if 'all' in args.phases or 'data' in args.phases:
        if not prepare_dataset():
            success = False
        else:
            phases_run.append('data')

    # Phase 2: Main Experiment
    if 'all' in args.phases or 'experiment' in args.phases:
        if not run_main_experiment():
            success = False
        else:
            phases_run.append('experiment')

    # Phase 3: Ablation Studies
    if 'all' in args.phases or 'ablation' in args.phases:
        if not run_ablation_studies():
            success = False
        else:
            phases_run.append('ablation')

    # Phase 4: Results Analysis
    if 'all' in args.phases or 'results' in args.phases:
        if not generate_results():
            success = False
        else:
            phases_run.append('results')

    # Show summary if all phases completed
    if success and phases_run:
        show_summary()

    # Cleanup
    if args.cleanup:
        cleanup_temp_files()

    if success:
        print(f"\n🎯 Experiment completed successfully! Phases run: {', '.join(phases_run)}")
    else:
        print(f"\n❌ Experiment failed. Completed phases: {', '.join(phases_run)}")

    return success


if __name__ == "__main__":
    start_time = datetime.now()
    success = main()
    end_time = datetime.now()

    duration = end_time - start_time
    print(f"\n⏱️  Total runtime: {duration}")

    sys.exit(0 if success else 1)
