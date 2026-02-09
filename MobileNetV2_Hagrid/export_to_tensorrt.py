"""
Export Trained FreiHAND Model to ONNX and TensorRT
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from models.mobilenetv2_heatmap import MobileNetV2Heatmap

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = "outputs/best_freihand_model.pth"
IMG_SIZE = 224
ONNX_PATH = "outputs/freihand_model.onnx"
TRT_PATH = "outputs/freihand_model.trt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("Loading trained model...")
model = MobileNetV2Heatmap(num_keypoints=21, in_channels=1)

# Load checkpoint
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
# EXPORT TO ONNX
# --------------------------------------------------
print("\n📤 Exporting to ONNX...")

dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)

try:
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['heatmaps'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'heatmaps': {0: 'batch_size'}
        },
        verbose=False
    )
    print(f"✅ ONNX export successful: {ONNX_PATH}")
    
    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")
    
except Exception as e:
    print(f"❌ ONNX export failed: {e}")
    sys.exit(1)

# --------------------------------------------------
# CONVERT TO TENSORRT
# --------------------------------------------------
print("\n⚡ Converting to TensorRT...")

try:
    import tensorrt_rtx as trt
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    def build_engine(onnx_file_path, engine_file_path):
        """Build TensorRT engine from ONNX file"""
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        print("  Parsing ONNX model...")
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('  ERROR: Failed to parse ONNX file')
                for error in range(parser.num_errors):
                    print(f"    {parser.get_error(error)}")
                return False
        
        print("  ✓ ONNX parsed successfully")
        
        # Build engine
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Enable FP16 if supported
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  ✓ FP16 mode enabled")
        
        print("  Building TensorRT engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("  ERROR: Failed to build engine")
            return False
        
        # Save engine
        with open(engine_file_path, 'wb') as f:
            f.write(serialized_engine)
        
        return True
    
    if build_engine(ONNX_PATH, TRT_PATH):
        print(f"✅ TensorRT engine saved: {TRT_PATH}")
        print("✓ TensorRT conversion successful!")
    else:
        print("❌ TensorRT conversion failed")
    
except ImportError:
    print("⚠️  TensorRT (tensorrt_rtx) not available")
    print("   You can still use the ONNX model for inference")
except Exception as e:
    print(f"❌ TensorRT conversion failed: {e}")

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print("\n" + "=" * 60)
print("📊 EXPORT SUMMARY")
print("=" * 60)
print(f"Model loaded from: {MODEL_PATH}")
print(f"✓ ONNX model: {ONNX_PATH}")
if Path(TRT_PATH).exists():
    print(f"✓ TensorRT engine: {TRT_PATH}")
    print("\n🚀 Your model is ready for high-speed inference!")
else:
    print("⚠️  TensorRT engine not created (use ONNX instead)")
print("=" * 60)