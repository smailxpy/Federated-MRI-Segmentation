#!/usr/bin/env python3
"""
Setup Script for Federated Continual Learning Project
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This script automates the project setup and environment configuration.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse
import json

# Add current directory to Python path
sys.path.insert(0, os.getcwd())


def run_command(cmd: str, description: str = "", check: bool = True) -> bool:
    """Run a shell command with proper error handling"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False


def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_cuda_availability():
    """Check CUDA availability"""
    print("🖥️  Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"✅ CUDA available: {device_count} device(s), using {device_name}")
            return True
        else:
            print("⚠️  CUDA not available, using CPU (training will be slow)")
            return True
    except ImportError:
        print("⚠️  PyTorch not installed yet, CUDA check deferred")
        return True


def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    print("🔑 Setting up Kaggle credentials...")

    kaggle_json = Path("kaggle.json")
    kaggle_dir = Path.home() / ".kaggle"

    if not kaggle_json.exists():
        print("⚠️  kaggle.json not found in project root")
        print("Please download your Kaggle API token from https://www.kaggle.com/account")
        print("and place it in the project root directory")
        return False

    try:
        kaggle_dir.mkdir(exist_ok=True)
        shutil.copy(kaggle_json, kaggle_dir / "kaggle.json")
        (kaggle_dir / "kaggle.json").chmod(0o600)
        print("✅ Kaggle credentials configured")
        return True
    except Exception as e:
        print(f"❌ Failed to setup Kaggle credentials: {e}")
        return False


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")

    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False

    # Upgrade pip first
    run_command("pip install --upgrade pip", "Upgrading pip")

    # Install requirements
    success = run_command("pip install -r requirements.txt", "Installing requirements")
    if success:
        print("✅ Dependencies installed successfully")
    return success


def verify_installation():
    """Verify that all components are properly installed"""
    print("🔍 Verifying installation...")

    checks = []

    # Check core libraries
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        checks.append(True)
    except ImportError:
        print("❌ PyTorch not installed")
        checks.append(False)

    try:
        import flwr
        print(f"✅ Flower {flwr.__version__} installed")
        checks.append(True)
    except ImportError:
        print("❌ Flower not installed")
        checks.append(False)

    try:
        import nibabel
        print(f"✅ NiBabel {nibabel.__version__} installed")
        checks.append(True)
    except ImportError:
        print("❌ NiBabel not installed")
        checks.append(False)

    try:
        import kaggle
        print(f"✅ Kaggle API {kaggle.__version__} installed")
        checks.append(True)
    except ImportError:
        print("❌ Kaggle API not installed")
        checks.append(False)

    # Check CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available")
        checks.append(True)
    except:
        checks.append(False)

    # Check Kaggle credentials
    kaggle_configured = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if kaggle_configured:
        print("✅ Kaggle credentials configured")
    else:
        print("❌ Kaggle credentials not configured")
    checks.append(kaggle_configured)

    success = all(checks)
    if success:
        print("✅ Installation verification completed successfully")
    else:
        print("❌ Some components failed verification")

    return success


def create_project_structure():
    """Create necessary directories"""
    print("📁 Creating project structure...")

    directories = [
        "data/raw",
        "data/processed",
        "results/models",
        "results/plots",
        "results/metrics",
        "logs",
        "configs"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("✅ Project structure created")
    return True


def test_basic_functionality():
    """Test basic functionality of the system"""
    print("🧪 Testing basic functionality...")

    try:
        # Test configuration loading
        import yaml
        with open("configs/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loading works")

        # Test model creation
        from src.models.unet_adapters import create_model
        model = create_model(config)
        print("✅ Model creation works")

        # Test metrics computation
        from src.utils.metrics import SegmentationMetrics
        metrics = SegmentationMetrics()
        print("✅ Metrics computation works")

        print("✅ Basic functionality test passed")
        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def create_demo_script():
    """Create a demo script for quick testing"""
    demo_script = '''#!/usr/bin/env python3
"""
Demo Script for Federated Continual Learning
Run this to quickly test the system with minimal data
"""

import subprocess
import sys

def main():
    print("🎯 Running Federated Continual Learning Demo")
    print("=" * 50)

    # Step 1: Prepare small dataset sample
    print("\\n📥 Step 1: Preparing demo dataset...")
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
    print("\\n🚀 Step 2: Running quick experiment...")
    result = subprocess.run([
        sys.executable, "src/experiments/train_fcl.py", "--quick-test"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Experiment failed")
        print(result.stderr)
        return

    print("✅ Quick experiment completed")

    # Step 3: Generate visualizations
    print("\\n📊 Step 3: Generating results...")
    result = subprocess.run([
        sys.executable, "src/utils/visualization.py",
        "--results_dir", "results", "--plot", "dashboard"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Visualization failed")
        print(result.stderr)
        return

    print("✅ Results generated")
    print("\\n🎉 Demo completed successfully!")
    print("📁 Check the 'results/' directory for outputs")

if __name__ == "__main__":
    main()
'''

    with open("demo.py", 'w') as f:
        f.write(demo_script)

    # Make executable
    os.chmod("demo.py", 0o755)
    print("✅ Demo script created (run with: python demo.py)")
    return True


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Federated Continual Learning Project")
    parser.add_argument("--skip-kaggle", action="store_true",
                       help="Skip Kaggle setup (you'll need to do it manually)")
    parser.add_argument("--demo", action="store_true",
                       help="Create demo script after setup")

    args = parser.parse_args()

    print("🚀 Setting up Federated Continual Learning for MRI Segmentation")
    print("=" * 70)
    print("Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation")
    print("=" * 70)

    success = True

    # Check Python version
    if not check_python_version():
        return False

    # Check CUDA
    check_cuda_availability()

    # Create project structure
    if not create_project_structure():
        success = False

    # Setup Kaggle (unless skipped)
    if not args.skip_kaggle:
        if not setup_kaggle_credentials():
            success = False

    # Install dependencies
    if not install_dependencies():
        success = False

    # Verify installation
    if not verify_installation():
        success = False

    # Test basic functionality
    if not test_basic_functionality():
        success = False

    # Create demo script
    if args.demo:
        create_demo_script()

    print("\\n" + "=" * 70)
    if success:
        print("🎉 Setup completed successfully!")
        print("\\n📋 Next steps:")
        print("1. Run demo: python demo.py")
        print("2. Full experiment: python src/experiments/train_fcl.py")
        print("3. View results: python src/utils/visualization.py --results_dir results")
        print("\\n📚 Documentation: See README.md for detailed instructions")
    else:
        print("❌ Setup completed with errors")
        print("Please check the error messages above and fix any issues")
        return False

    print("=" * 70)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
