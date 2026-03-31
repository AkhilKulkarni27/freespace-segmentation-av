import torch
import time
import segmentation_models_pytorch as smp

device = torch.device('cuda')
model  = smp.DeepLabV3Plus(
    encoder_name='efficientnet-b4',
    encoder_weights=None,
    in_channels=3, classes=1
).to(device).eval()

ckpt  = torch.load('checkpoints_v3/best_model.pth', map_location=device)
state = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
model.load_state_dict(state)

x = torch.randn(1, 3, 512, 512).to(device)

# Warmup
for _ in range(10):
    model(x)

# Benchmark
t0  = time.perf_counter()
for _ in range(100):
    model(x)
fps = 100 / (time.perf_counter() - t0)

print(f'FPS: {fps:.1f}')
print(f'Inference time: {1000/fps:.1f} ms per frame')
