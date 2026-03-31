"""
model.py — Segmentation Model for Free Space Detection

Supports:
    - DeepLabV3+  (default, best accuracy/speed tradeoff)
    - SegFormer    (transformer-based, better on complex urban scenes)
    - UNet         (lightweight, good for constrained edge hardware)

Usage:
    from model import build_model
    model = build_model(arch="deeplabv3plus", encoder="resnet50")
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    arch: str = "deeplabv3plus",
    encoder: str = "resnet50",
    pretrained: bool = True,
    num_classes: int = 1,       # 1 = binary output (free vs not-free)
):
    """
    Build a segmentation model with ImageNet-pretrained encoder.

    Args:
        arch:        "deeplabv3plus" | "segformer" | "unet" | "fpn"
        encoder:     backbone name — see encoder options below
        pretrained:  use ImageNet weights (strongly recommended)
        num_classes: 1 for binary segmentation

    Recommended encoder options:
        "resnet50"       — balanced speed + accuracy (default)
        "resnet101"      — higher accuracy, more memory
        "efficientnet-b4" — efficient for edge deployment
        "mit_b2"         — SegFormer backbone (best with arch="segformer")
        "mit_b4"         — larger SegFormer backbone
    """
    encoder_weights = "imagenet" if pretrained else None
    activation = None   # We apply sigmoid/softmax manually in loss

    arch = arch.lower()

    if arch == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=activation,
        )
    elif arch == "segformer":
        # SegFormer works best with mit_b2 or mit_b4 encoder
        if not encoder.startswith("mit"):
            print(f"[Warning] SegFormer works best with mit_b2/b4 encoder, got: {encoder}")
        model = smp.Segformer(
            encoder_name=encoder if encoder.startswith("mit") else "mit_b2",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=activation,
        )
    elif arch == "unet":
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=activation,
        )
    elif arch == "fpn":
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=activation,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: deeplabv3plus, segformer, unet, fpn")

    return model


# ---------------------------------------------------------------------------
# Combined loss function
# ---------------------------------------------------------------------------

class FreeSpaceLoss(nn.Module):
    """
    Combines Dice Loss + Binary Cross Entropy.
    
    Dice handles class imbalance (road pixels can be sparse in some scenes).
    BCE provides stable gradients and pixel-level supervision.
    
    alpha controls the mix: total_loss = alpha * dice + (1 - alpha) * bce
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce  = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if logits.dim() == 4:
            logits = logits.squeeze(1)   # [B, 1, H, W] → [B, H, W]
        dice_loss = self.dice(logits.unsqueeze(1), targets.unsqueeze(1))
        bce_loss  = self.bce(logits, targets)
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class SegmentationMetrics:
    """Computes IoU, precision, recall, and F1 for binary segmentation."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = (torch.sigmoid(logits) > self.threshold).float()
        if preds.dim() == 4:
            preds = preds.squeeze(1)
        targets = targets.float()

        self.tp += ((preds == 1) & (targets == 1)).sum().item()
        self.fp += ((preds == 1) & (targets == 0)).sum().item()
        self.fn += ((preds == 0) & (targets == 1)).sum().item()
        self.tn += ((preds == 0) & (targets == 0)).sum().item()

    def compute(self) -> dict:
        eps = 1e-7
        iou       = self.tp / (self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall    = self.tp / (self.tp + self.fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        accuracy  = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }


# ---------------------------------------------------------------------------
# Quick model summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_model(arch="deeplabv3plus", encoder="resnet50")
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # [2, 1, 512, 512]

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.1f}M")

    loss_fn = FreeSpaceLoss()
    dummy_mask = torch.randint(0, 2, (2, 512, 512))
    loss = loss_fn(out.squeeze(1), dummy_mask.float())
    print(f"Loss (random): {loss.item():.4f}")
