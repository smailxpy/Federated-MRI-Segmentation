#!/usr/bin/env python3
"""
Direct Training Script for Federated Continual Learning
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This script performs federated-style training directly without Ray/subprocess overhead.
Designed for 32GB RAM systems to achieve >70% Dice score.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import yaml
import argparse
import json
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.unet_adapters import create_model, save_model_checkpoint
from src.federated.client import BraTS2021Dataset, MultiClassDiceCELoss, compute_dice_score


def get_device(config):
    """Get the best available device"""
    if config['hardware']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[DEVICE] Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print(f"[DEVICE] Using CPU")
    return device


def load_client_data(client_dirs, config):
    """Load datasets for all clients"""
    client_loaders = []
    
    for i, client_dir in enumerate(client_dirs):
        train_dir = client_dir / "train"
        val_dir = client_dir / "val"
        
        if not train_dir.exists() or not val_dir.exists():
            print(f"[WARNING] Client {i} data not found at {client_dir}")
            continue
        
        train_dataset = BraTS2021Dataset(train_dir, config)
        val_dataset = BraTS2021Dataset(val_dir, config)
        
        batch_size = config['federated']['batch_size']
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory'] and torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory'] and torch.cuda.is_available()
        )
        
        client_loaders.append({
            'client_id': i,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset)
        })
        
        print(f"  Client {i}: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return client_loaders


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, config, epoch, client_id):
    """Train for one epoch with verbose output"""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if CUDA
        if device.type == 'cuda' and config['hardware'].get('mixed_precision', False):
            with torch.amp.autocast('cuda'):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
        else:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Compute dice
        with torch.no_grad():
            pred = torch.argmax(outputs, dim=1)
            batch_dice = compute_dice_score(pred, batch_y, config['dataset']['num_classes'])
        
        total_loss += loss.item()
        total_dice += batch_dice
        
        # Verbose output
        if batch_idx % 2 == 0 or batch_idx == num_batches - 1:
            print(f"     [Client {client_id}] Batch {batch_idx+1}/{num_batches} - "
                  f"Loss: {loss.item():.4f}, Dice: {batch_dice:.4f}")
    
    return total_loss / num_batches, total_dice / num_batches


def evaluate(model, val_loader, criterion, device, config, client_id):
    """Evaluate model with comprehensive metrics"""
    model.eval()
    total_loss = 0.0
    all_dice = []
    class_dice = {i: [] for i in range(config['dataset']['num_classes'])}
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            
            pred = torch.argmax(outputs, dim=1)
            
            # Per-sample metrics
            for i in range(batch_x.size(0)):
                pred_i = pred[i]
                target_i = batch_y[i]
                
                # Overall dice (tumor classes)
                dice_scores = []
                for c in range(1, config['dataset']['num_classes']):
                    pred_c = (pred_i == c).float()
                    target_c = (target_i == c).float()
                    intersection = (pred_c * target_c).sum()
                    union = pred_c.sum() + target_c.sum()
                    if union > 0:
                        dice_c = (2.0 * intersection / (union + 1e-6)).item()
                        dice_scores.append(dice_c)
                        class_dice[c].append(dice_c)
                
                if dice_scores:
                    all_dice.append(np.mean(dice_scores))
    
    avg_loss = total_loss / len(val_loader.dataset)
    avg_dice = np.mean(all_dice) if all_dice else 0.0
    
    # Per-class dice means
    class_dice_means = {}
    for c in range(1, config['dataset']['num_classes']):
        class_dice_means[f'dice_class_{c}'] = np.mean(class_dice[c]) if class_dice[c] else 0.0
    
    return avg_loss, avg_dice, class_dice_means


def federated_averaging(global_params, client_updates, client_weights):
    """Perform FedAvg aggregation"""
    total_weight = sum(client_weights)
    
    # Weighted average of parameters
    avg_params = []
    for param_idx in range(len(global_params)):
        weighted_sum = None
        for client_idx, update in enumerate(client_updates):
            weight = client_weights[client_idx] / total_weight
            if weighted_sum is None:
                weighted_sum = update[param_idx] * weight
            else:
                weighted_sum += update[param_idx] * weight
        avg_params.append(weighted_sum)
    
    return avg_params


def run_federated_training(config, exp_dir):
    """Run federated training with verbose output"""
    print("\n" + "="*70)
    print("FEDERATED CONTINUAL LEARNING - DIRECT TRAINING")
    print("="*70)
    print(f"Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation")
    print(f"Experiment: {exp_dir}")
    print("="*70 + "\n")
    
    device = get_device(config)
    
    # Setup directories
    processed_dir = Path(config['dataset']['processed_dir'])
    models_dir = exp_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Get client directories (hospitals)
    client_dirs = []
    for hospital_id in ['a', 'b', 'c', 'd']:
        hospital_dir = processed_dir / f"hospital_{hospital_id}"
        if hospital_dir.exists() and (hospital_dir / "train").exists():
            client_dirs.append(hospital_dir)
    
    print(f"[DATA] Loading data from {len(client_dirs)} clients...")
    client_loaders = load_client_data(client_dirs, config)
    
    if not client_loaders:
        raise ValueError("No client data found!")
    
    # Create global model
    print(f"\n[MODEL] Creating U-Net with Drift-Aware Adapters...")
    global_model = create_model(config)
    global_model.to(device)
    
    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = MultiClassDiceCELoss(
        num_classes=config['dataset']['num_classes'],
        dice_weight=1.0,
        ce_weight=0.5
    )
    
    # Training configuration
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    lr = config['federated']['learning_rate']
    
    print(f"\n[CONFIG] Training Configuration:")
    print(f"  Rounds: {num_rounds}")
    print(f"  Local Epochs: {local_epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Batch Size: {config['federated']['batch_size']}")
    
    # Metrics tracking
    best_dice = 0.0
    best_round = 0
    history = {'rounds': [], 'train_loss': [], 'val_loss': [], 'dice': [], 'class_dice': []}
    
    print("\n" + "="*70)
    print("STARTING FEDERATED TRAINING")
    print("="*70 + "\n")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"{'='*60}")
        
        round_start = datetime.now()
        client_updates = []
        client_weights = []
        round_train_losses = []
        round_val_losses = []
        round_dices = []
        
        # Get global parameters
        global_params = [p.data.clone() for p in global_model.get_shared_parameters()]
        
        # Train each client
        for client_data in client_loaders:
            client_id = client_data['client_id']
            train_loader = client_data['train_loader']
            val_loader = client_data['val_loader']
            train_size = client_data['train_size']
            
            print(f"\n  [CLIENT {client_id}] Training...")
            
            # Reset model to global parameters
            for param, global_param in zip(global_model.get_shared_parameters(), global_params):
                param.data.copy_(global_param)
            
            # Setup optimizer for this client
            optimizer = optim.AdamW(
                global_model.parameters(),
                lr=lr,
                weight_decay=config['training']['weight_decay']
            )
            
            # Scheduler
            total_steps = local_epochs * len(train_loader)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr * 3,
                total_steps=total_steps,
                pct_start=0.3
            )
            
            # Local training
            client_train_loss = 0.0
            client_train_dice = 0.0
            
            for epoch in range(local_epochs):
                epoch_loss, epoch_dice = train_one_epoch(
                    global_model, train_loader, criterion, optimizer, scheduler,
                    device, config, epoch, client_id
                )
                client_train_loss += epoch_loss
                client_train_dice += epoch_dice
                
                print(f"     Epoch {epoch+1}/{local_epochs} - Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}")
            
            avg_train_loss = client_train_loss / local_epochs
            avg_train_dice = client_train_dice / local_epochs
            
            # Evaluate
            val_loss, val_dice, class_dices = evaluate(
                global_model, val_loader, criterion, device, config, client_id
            )
            
            print(f"  [CLIENT {client_id}] Results:")
            print(f"     Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
            print(f"     Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f} ({val_dice*100:.1f}%)")
            
            # Store update
            client_updates.append([p.data.clone() for p in global_model.get_shared_parameters()])
            client_weights.append(train_size)
            round_train_losses.append(avg_train_loss)
            round_val_losses.append(val_loss)
            round_dices.append(val_dice)
        
        # Federated Averaging
        print(f"\n  [AGGREGATION] Performing FedAvg...")
        avg_params = federated_averaging(global_params, client_updates, client_weights)
        
        # Update global model
        for param, avg_param in zip(global_model.get_shared_parameters(), avg_params):
            param.data.copy_(avg_param)
        
        # Round summary
        round_train_loss = np.mean(round_train_losses)
        round_val_loss = np.mean(round_val_losses)
        round_dice = np.mean(round_dices)
        round_time = (datetime.now() - round_start).total_seconds()
        
        print(f"\n  [ROUND {round_num} SUMMARY]")
        print(f"     Avg Train Loss: {round_train_loss:.4f}")
        print(f"     Avg Val Loss: {round_val_loss:.4f}")
        print(f"     Avg Dice: {round_dice:.4f} ({round_dice*100:.1f}%)")
        print(f"     Time: {round_time:.1f}s")
        
        # Track history
        history['rounds'].append(round_num)
        history['train_loss'].append(round_train_loss)
        history['val_loss'].append(round_val_loss)
        history['dice'].append(round_dice)
        
        # Check if best
        if round_dice > best_dice:
            best_dice = round_dice
            best_round = round_num
            
            # Save best model
            best_path = models_dir / "best_model.pth"
            torch.save({
                'round': round_num,
                'model_state_dict': global_model.state_dict(),
                'dice': round_dice,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }, best_path)
            print(f"     [CHECKPOINT] New best model saved! Dice: {best_dice:.4f}")
        
        # Save checkpoint every N rounds
        if round_num % config['training'].get('save_interval', 5) == 0:
            ckpt_path = models_dir / f"checkpoint_round_{round_num}.pth"
            torch.save({
                'round': round_num,
                'model_state_dict': global_model.state_dict(),
                'dice': round_dice,
                'history': history
            }, ckpt_path)
            print(f"     [CHECKPOINT] Round {round_num} checkpoint saved")
        
        # Check target achievement
        if round_dice >= 0.70:
            print(f"\n  [SUCCESS] TARGET ACHIEVED! Dice >= 70%")
        else:
            gap = 0.70 - round_dice
            print(f"     Gap to 70% target: {gap*100:.1f}%")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  Best Round: {best_round}")
    print(f"  Best Dice: {best_dice:.4f} ({best_dice*100:.1f}%)")
    print(f"  Final Dice: {history['dice'][-1]:.4f} ({history['dice'][-1]*100:.1f}%)")
    
    if best_dice >= 0.70:
        print(f"\n  [SUCCESS] TARGET ACHIEVED! Best Dice >= 70%")
    else:
        print(f"\n  [INFO] Target not reached. Consider:")
        print(f"     - More training rounds")
        print(f"     - Higher learning rate")
        print(f"     - More training data")
    
    # Save final report
    report = {
        'best_round': best_round,
        'best_dice': float(best_dice),
        'final_dice': float(history['dice'][-1]),
        'total_rounds': num_rounds,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(exp_dir / "final_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Report saved to: {exp_dir / 'final_report.json'}")
    print(f"  Best model saved to: {models_dir / 'best_model.pth'}")
    print("="*70 + "\n")
    
    return best_dice, history


def main():
    parser = argparse.ArgumentParser(description="Direct Federated Training")
    parser.add_argument("--config", type=str, default="configs/config_32gb.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment directory
    exp_dir = Path(config['output']['results_dir']) / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, indent=2)
    
    # Run training
    best_dice, history = run_federated_training(config, exp_dir)
    
    print(f"Final Best Dice: {best_dice:.4f} ({best_dice*100:.1f}%)")
    
    return best_dice


if __name__ == "__main__":
    main()


