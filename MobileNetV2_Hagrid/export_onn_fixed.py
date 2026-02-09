"""
Re-export ONNX Model with Embedded Weights (TensorRT Compatible)
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.mobilenetv2_heatmap import MobileNetV2Heatmap

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = "outputs/best_freihand_model.pth"
IMG_SIZE = 224
ONNX_PATH = "outputs/freihand_model_trt.onnx"  # New name

device = "cpu"  # Use CPU for export
print(f"Using device: {device}\n")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("Loading trained model...")
model = MobileNetV2Heatmap(num_keypoints=21, in_channels=1)

checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
else:
    model.load_state_dict(checkpoint)
    print("✓ Loaded model weights")

model.to(device)
model.eval()

# --------------------------------------------------
# EXPORT TO ONNX (TensorRT Compatible)
# --------------------------------------------------
print("\n📤 Exporting to ONNX (TensorRT compatible)...")

dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)

try:
    # Simplified export with embedded weights
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,  # Lower version for better TRT compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['heatmaps'],
        verbose=False
    )
    print(f"✅ ONNX export successful: {ONNX_PATH}")
    
    # Verify
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    print(f"✓ Model size: {Path(ONNX_PATH).stat().st_size / (1024*1024):.2f} MB")
    print("✓ ONNX model verified")
    
except Exception as e:
    print(f"❌ ONNX export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ONNX export complete!")
print("=" * 60)
print(f"File: {ONNX_PATH}")
print("\nNext step: Convert to TensorRT")
print("Run: python convert_to_trt_simple.py")
print("=" * 60)