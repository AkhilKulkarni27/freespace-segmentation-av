"""
dataset.py — Free Space Segmentation Dataset Loader

Directory structure expected:
    data/
        images/
            train/  *.jpg / *.png
            val/
            test/
        masks/
            train/  *.png  (binary: 255 = free space, 0 = not free)
            val/
            test/
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(height: int = 512, width: int = 512):
    return A.Compose([
        A.Resize(height, width),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=5, p=0.2),          # simulates camera motion
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),   # weather robustness
        A.RandomRain(slant_lower=-5, slant_upper=5, drop_length=10, p=0.1),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(height: int = 512, width: int = 512):
    return A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class FreeSpaceDataset(Dataset):
    """
    Loads image + binary segmentation mask pairs.

    Masks must be single-channel PNGs:
        255  →  free space (driveable)
          0  →  not free  (curb, barrier, sidewalk, etc.)
    """

    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # Collect all image files and match to masks
        self.image_paths = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.jpeg")) +
            list(self.images_dir.glob("*.png"))
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Mask shares the same stem, always saved as PNG
        mask_path = self.masks_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Load image (BGR → RGB)
        image = cv2.imread(str(img_path))
        if image is None:
            raise IOError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask as grayscale, binarize to {0, 1}
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to read mask: {mask_path}")
        mask = (mask > 127).astype(np.uint8)   # 1 = free space, 0 = not free

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]           # Tensor [3, H, W]
            mask = augmented["mask"].long()      # Tensor [H, W]

        return image, mask


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root: str = "data",
    img_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
):
    """
    Returns (train_loader, val_loader, test_loader).

    Args:
        data_root:   root folder containing images/ and masks/ subdirs
        img_size:    square resize target
        batch_size:  per-GPU batch size (reduce to 4 if OOM)
        num_workers: dataloader workers (set 0 on Windows)
    """
    train_ds = FreeSpaceDataset(
        images_dir=os.path.join(data_root, "images", "train"),
        masks_dir=os.path.join(data_root, "masks", "train"),
        transform=get_train_transforms(img_size, img_size),
    )
    val_ds = FreeSpaceDataset(
        images_dir=os.path.join(data_root, "images", "val"),
        masks_dir=os.path.join(data_root, "masks", "val"),
        transform=get_val_transforms(img_size, img_size),
    )
    test_ds = FreeSpaceDataset(
        images_dir=os.path.join(data_root, "images", "test"),
        masks_dir=os.path.join(data_root, "masks", "test"),
        transform=get_val_transforms(img_size, img_size),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_loader, val_loader, _ = build_dataloaders(
        data_root="data", img_size=512, batch_size=4
    )

    images, masks = next(iter(train_loader))
    print("Image batch:", images.shape)   # [B, 3, 512, 512]
    print("Mask batch: ", masks.shape)    # [B, 512, 512]
    print("Mask unique values:", masks.unique())

    # Visualise first sample
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_show = (images[0] * std + mean).permute(1, 2, 0).numpy().clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_show);     axes[0].set_title("Image");      axes[0].axis("off")
    axes[1].imshow(masks[0], cmap="gray"); axes[1].set_title("Free Space Mask"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("sample_check.png", dpi=120)
    print("Saved sample_check.png")
