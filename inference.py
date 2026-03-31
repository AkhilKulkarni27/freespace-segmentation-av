"""
inference.py — Run inference and export model for deployment

Usage:
    # Run on a single image
    python inference.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth

    # Run on a video file
    python inference.py --video path/to/video.mp4 --checkpoint checkpoints/best_model.pth

    # Export to ONNX
    python inference.py --export_onnx --checkpoint checkpoints/best_model.pth
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import build_model


MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def get_inference_transform(img_size: int = 512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class FreeSpaceInference:
    def __init__(
        self,
        checkpoint_path: str,
        arch: str = "deeplabv3plus",
        encoder: str = "resnet50",
        img_size: int = 512,
        threshold: float = 0.5,
        device: str = None,
    ):
        self.device    = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.img_size  = img_size
        self.threshold = threshold
        self.transform = get_inference_transform(img_size)

        # Load model
        self.model = build_model(arch=arch, encoder=encoder, pretrained=False)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model", ckpt)
        # Strip DataParallel prefix if present
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded model from {checkpoint_path} on {self.device}")

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            image_bgr: BGR image as numpy array (H, W, 3)
        Returns:
            mask: binary numpy array (H, W), uint8 {0, 255}
        """
        orig_h, orig_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        inp = self.transform(image=image_rgb)["image"].unsqueeze(0).to(self.device)
        with autocast():
            logits = self.model(inp)                  # [1, 1, H, W]
        prob = torch.sigmoid(logits[0, 0]).cpu().numpy()

        # Resize back to original resolution
        prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask = (prob > self.threshold).astype(np.uint8) * 255
        return mask

    def overlay(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        color: tuple = (0, 255, 100),
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Blend free-space mask onto image for visualization."""
        overlay = image_bgr.copy()
        overlay[mask > 0] = (
            np.array(image_bgr[mask > 0], dtype=np.float32) * (1 - alpha) +
            np.array(color, dtype=np.float32) * alpha
        ).astype(np.uint8)
        return overlay


# ---------------------------------------------------------------------------
# Video inference
# ---------------------------------------------------------------------------

def run_video(engine: FreeSpaceInference, video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_count = 0
    total_time  = 0.0
    print(f"Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0   = time.perf_counter()
        mask = engine.predict(frame)
        dt   = time.perf_counter() - t0

        total_time  += dt
        frame_count += 1

        vis = engine.overlay(frame, mask)

        # FPS counter
        cv2.putText(vis, f"{1/dt:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(vis)

        if frame_count % 50 == 0:
            print(f"  Frame {frame_count} | {dt*1000:.1f} ms | avg {total_time/frame_count*1000:.1f} ms")

    cap.release()
    writer.release()
    avg_fps = frame_count / total_time
    print(f"Done. {frame_count} frames | avg {avg_fps:.1f} FPS → {output_path}")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    checkpoint_path: str,
    arch: str = "deeplabv3plus",
    encoder: str = "resnet50",
    img_size: int = 512,
    output_path: str = "freespace_model.onnx",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(arch=arch, encoder=encoder, pretrained=False)
    ckpt   = torch.load(checkpoint_path, map_location=device)
    state  = {k.replace("module.", ""): v for k, v in ckpt.get("model", ckpt).items()}
    model.load_state_dict(state)
    model.to(device).eval()

    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    torch.onnx.export(
        model, dummy, output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image":  {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {output_path}")
    print("Next step for TensorRT:")
    print(f"  trtexec --onnx={output_path} --saveEngine=freespace.trt --fp16")


# ---------------------------------------------------------------------------
# Evaluate on test set
# ---------------------------------------------------------------------------

def evaluate_test_set(engine: FreeSpaceInference, test_dir: str):
    from dataset import FreeSpaceDataset, get_val_transforms
    from model import SegmentationMetrics

    ds = FreeSpaceDataset(
        images_dir=f"{test_dir}/images/test",
        masks_dir=f"{test_dir}/masks/test",
        transform=get_val_transforms(engine.img_size),
    )
    metrics = SegmentationMetrics(threshold=engine.threshold)

    print(f"Evaluating on {len(ds)} test samples...")
    for i, (image, mask) in enumerate(ds):
        with torch.no_grad():
            logits = engine.model(image.unsqueeze(0).to(engine.device))
        metrics.update(logits.cpu(), mask.unsqueeze(0))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ds)} ...")

    results = metrics.compute()
    print("\n=== Test Set Results ===")
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True,             help="Path to .pth checkpoint")
    p.add_argument("--arch",         default="deeplabv3plus")
    p.add_argument("--encoder",      default="resnet50")
    p.add_argument("--img_size",     default=512,   type=int)
    p.add_argument("--threshold",    default=0.5,   type=float)
    p.add_argument("--image",        default=None,              help="Single image inference")
    p.add_argument("--video",        default=None,              help="Video inference")
    p.add_argument("--output",       default="output.mp4",      help="Output video path")
    p.add_argument("--data_root",    default="data",            help="Test set root for evaluation")
    p.add_argument("--export_onnx",  action="store_true",       help="Export to ONNX")
    p.add_argument("--onnx_path",    default="freespace.onnx",  help="ONNX output path")
    args = p.parse_args()

    if args.export_onnx:
        export_onnx(args.checkpoint, args.arch, args.encoder, args.img_size, args.onnx_path)

    elif args.image:
        engine = FreeSpaceInference(
            args.checkpoint, args.arch, args.encoder, args.img_size, args.threshold
        )
        frame = cv2.imread(args.image)
        mask  = engine.predict(frame)
        vis   = engine.overlay(frame, mask)
        out   = Path(args.image).stem + "_freespace.jpg"
        cv2.imwrite(out, vis)
        print(f"Saved → {out}")

    elif args.video:
        engine = FreeSpaceInference(
            args.checkpoint, args.arch, args.encoder, args.img_size, args.threshold
        )
        run_video(engine, args.video, args.output)

    else:
        engine = FreeSpaceInference(
            args.checkpoint, args.arch, args.encoder, args.img_size, args.threshold
        )
        evaluate_test_set(engine, args.data_root)
