#!/usr/bin/env python3
"""
U-Net with Drift-Aware Adapters for Federated Continual Learning
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This module implements a U-Net architecture with drift-aware adapters for
federated continual learning in brain tumor segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class DriftAwareAdapter(nn.Module):
    """Drift-aware adapter module for continual learning"""

    def __init__(self, channels: int, adapter_channels: int = 16, dropout: float = 0.1):
        super().__init__()
        self.adapter_channels = adapter_channels

        # Adapter layers
        self.adapter = nn.Sequential(
            nn.Linear(channels, adapter_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_channels, channels),
            nn.Sigmoid()
        )

        # Domain-specific parameters (learnable)
        self.domain_scale = nn.Parameter(torch.ones(1))
        self.domain_shift = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with drift adaptation"""
        # Apply adapter modulation - use global average pooling
        pooled = x.mean(dim=[-3, -2, -1])  # [batch, channels]
        adapter_output = self.adapter(pooled)  # [batch, channels]
        # Expand back to spatial dimensions
        adapter_output = adapter_output.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [batch, channels, 1, 1, 1]
        adapter_output = adapter_output.expand_as(x)

        # Apply domain-specific transformation
        adapted = x * (1 + self.domain_scale) + self.domain_shift
        adapted = adapted * adapter_output

        return adapted


class EncoderBlock(nn.Module):
    """U-Net Encoder Block with optional adapter"""

    def __init__(self, in_channels: int, out_channels: int, use_adapter: bool = True,
                 adapter_channels: int = 16, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.adapter = DriftAwareAdapter(out_channels, adapter_channels, dropout) if use_adapter else None
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))

        if self.adapter is not None:
            x = self.adapter(x)

        return x


class DecoderBlock(nn.Module):
    """U-Net Decoder Block"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)

        # Handle size mismatch from upsampling
        diff_z = skip.size(2) - x.size(2)
        diff_y = skip.size(3) - x.size(3)
        diff_x = skip.size(4) - x.size(4)

        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2,
                      diff_z // 2, diff_z - diff_z // 2])

        x = torch.cat([skip, x], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))

        return x


class UNetWithAdapters(nn.Module):
    """U-Net with Drift-Aware Adapters for Federated Continual Learning
    
    Architecture for BraTS2021 4-class segmentation:
    - 4 modalities input (T1, T1ce, T2, FLAIR)
    - 4 class output (Background, NCR/NET, ED, ET)
    - Drift-aware adapters in encoder blocks for continual learning
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.num_classes = config['dataset']['num_classes']
        self.encoder_channels = config['model']['encoder_channels']
        self.use_adapters = config['model']['use_adapters']
        self.adapter_channels = config['model']['adapter_channels']
        self.dropout = config['model']['dropout']
        self.in_channels = len(config['dataset']['modalities'])

        # Input layer
        self.input_conv = nn.Sequential(
            nn.Conv3d(self.in_channels, self.encoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(self.encoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i in range(len(self.encoder_channels) - 1):
            self.encoders.append(
                EncoderBlock(
                    self.encoder_channels[i],
                    self.encoder_channels[i + 1],
                    use_adapter=self.use_adapters,
                    adapter_channels=self.adapter_channels,
                    dropout=self.dropout
                )
            )
            self.pools.append(nn.MaxPool3d(2))

        # Bottleneck
        bottleneck_channels = self.encoder_channels[-1] * 2
        self.bottleneck = EncoderBlock(
            self.encoder_channels[-1],
            bottleneck_channels,
            use_adapter=self.use_adapters,
            adapter_channels=self.adapter_channels,
            dropout=self.dropout
        )

        # Decoder path
        # Each decoder block upsamples and concatenates with skip
        # DecoderBlock expects: in_channels = upsampled + skip
        decoder_in_channels = [bottleneck_channels] + list(self.encoder_channels[::-1][:-1])
        decoder_out_channels = list(self.encoder_channels[::-1])
        
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(len(decoder_out_channels)):
            # Upconv halves channels, then concat with skip doubles it
            self.upconvs.append(
                nn.ConvTranspose3d(decoder_in_channels[i], decoder_out_channels[i], kernel_size=2, stride=2)
            )
            # After concat: decoder_out_channels[i] (from upconv) + decoder_out_channels[i] (from skip)
            self.decoders.append(
                nn.Sequential(
                    nn.Conv3d(decoder_out_channels[i] * 2, decoder_out_channels[i], kernel_size=3, padding=1),
                    nn.BatchNorm3d(decoder_out_channels[i]),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(self.dropout),
                    nn.Conv3d(decoder_out_channels[i], decoder_out_channels[i], kernel_size=3, padding=1),
                    nn.BatchNorm3d(decoder_out_channels[i]),
                    nn.ReLU(inplace=True)
                )
            )

        # Output layer
        self.output = nn.Conv3d(decoder_out_channels[-1], self.num_classes, kernel_size=1)

        # Print architecture summary
        print(f"[MODEL] U-Net Architecture:")
        print(f"   Input: {self.in_channels} modalities")
        print(f"   Encoder: {self.encoder_channels}")
        print(f"   Bottleneck: {bottleneck_channels}")
        print(f"   Decoder: {decoder_out_channels}")
        print(f"   Output: {self.num_classes} classes")
        print(f"   Adapters: {self.use_adapters}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net"""
        # Input conv
        x = self.input_conv(x)

        # Encoder path with skip connections
        skips = [x]
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        skips = skips[::-1]  # Reverse for decoder (most recent first)
        
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            
            # Handle size mismatch from upsampling
            skip = skips[i]
            if x.shape != skip.shape:
                diff_z = skip.size(2) - x.size(2)
                diff_y = skip.size(3) - x.size(3)
                diff_x = skip.size(4) - x.size(4)
                x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                              diff_y // 2, diff_y - diff_y // 2,
                              diff_z // 2, diff_z - diff_z // 2])
            
            # Concatenate with skip connection
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # Output
        x = self.output(x)
        return x

    def get_shared_parameters(self) -> List[nn.Parameter]:
        """Get parameters that should be shared in federated learning (all except adapters)"""
        shared_params = []
        adapter_param_ids = set(id(p) for p in self.get_adapter_parameters())

        for param in self.parameters():
            if id(param) not in adapter_param_ids:
                shared_params.append(param)

        return shared_params

    def get_adapter_parameters(self) -> List[nn.Parameter]:
        """Get adapter-specific parameters (not shared in federated learning)"""
        adapter_params = []

        for module in self.modules():
            if isinstance(module, DriftAwareAdapter):
                adapter_params.extend(module.parameters())

        return adapter_params

    def freeze_shared_parameters(self):
        """Freeze shared parameters during local adaptation"""
        for param in self.get_shared_parameters():
            param.requires_grad = False

    def unfreeze_shared_parameters(self):
        """Unfreeze shared parameters"""
        for param in self.get_shared_parameters():
            param.requires_grad = True

    def freeze_adapters(self):
        """Freeze adapter parameters"""
        for param in self.get_adapter_parameters():
            param.requires_grad = False

    def unfreeze_adapters(self):
        """Unfreeze adapter parameters"""
        for param in self.get_adapter_parameters():
            param.requires_grad = True


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)

        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Dice + Cross-Entropy Loss"""

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)

        return self.dice_weight * dice + self.ce_weight * ce


def create_model(config: dict) -> UNetWithAdapters:
    """Factory function to create U-Net model"""
    return UNetWithAdapters(config)


def load_model_weights(model: UNetWithAdapters, checkpoint_path: str,
                      load_adapters: bool = True) -> UNetWithAdapters:
    """Load model weights from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if load_adapters:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Load only shared parameters
        model_state_dict = model.state_dict()
        shared_params = {k: v for k, v in checkpoint['model_state_dict'].items()
                        if not any(adapter_name in k for adapter_name in ['adapter'])}

        model_state_dict.update(shared_params)
        model.load_state_dict(model_state_dict)

    return model


def save_model_checkpoint(model: UNetWithAdapters, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, save_path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config
    }

    torch.save(checkpoint, save_path)
    print(f"[OK] Model checkpoint saved to {save_path}")


if __name__ == "__main__":
    # Test the model
    import yaml

    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model = create_model(config)

    # Test input
    batch_size, channels, depth, height, width = 2, 4, 32, 32, 32
    x = torch.randn(batch_size, channels, depth, height, width)

    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected classes: {config['dataset']['num_classes']}")

        # Test parameter separation
        shared_params = model.get_shared_parameters()
        adapter_params = model.get_adapter_parameters()

        print(f"Shared parameters: {len(shared_params)}")
        print(f"Adapter parameters: {len(adapter_params)}")
        print("[OK] Model test completed successfully")
