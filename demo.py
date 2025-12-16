#!/usr/bin/env python3
"""
Demo Script for Federated Continual Learning
Run this to quickly test the system with minimal data
"""

import subprocess
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def main():
    print("🎯 Running Federated Continual Learning Demo")
    print("=" * 50)

    # Step 1: Prepare small dataset sample
    print("\n📥 Step 1: Preparing demo dataset...")
    result = subprocess.run([
        sys.executable, "src/data/download_dataset.py",
        "--download", "--extract", "--preprocess"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Dataset preparation failed")
        print(result.stderr)
        return

    print("✅ Demo dataset prepared")

    # Step 2: Run quick experiment
    print("\n🚀 Step 2: Running quick experiment...")
    result = subprocess.run([
        sys.executable, "src/experiments/train_fcl.py", "--quick-test"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Experiment failed")
        print(result.stderr)
        return

    print("✅ Quick experiment completed")

    # Step 3: Generate visualizations
    print("\n📊 Step 3: Generating results...")
    result = subprocess.run([
        sys.executable, "src/utils/visualization.py",
        "--results_dir", "results", "--plot", "dashboard"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Visualization failed")
        print(result.stderr)
        return

    print("✅ Results generated")
    print("\n🎉 Demo completed successfully!")
    print("📁 Check the 'results/' directory for outputs")

if __name__ == "__main__":
    main()
