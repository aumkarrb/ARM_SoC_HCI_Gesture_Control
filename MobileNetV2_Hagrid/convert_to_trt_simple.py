"""
Convert ONNX to TensorRT (Simplified)
"""

import tensorrt_rtx as trt
from pathlib import Path
import sys

ONNX_PATH = "outputs/freihand_model_trt.onnx"
TRT_PATH = "outputs/freihand_model.trt"

if not Path(ONNX_PATH).exists():
    print(f"❌ ONNX file not found: {ONNX_PATH}")
    print("Run: python export_onnx_fixed.py first")
    sys.exit(1)

print("Converting ONNX to TensorRT...\n")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse ONNX
print("Parsing ONNX...")
with open(ONNX_PATH, 'rb') as f:
    if not parser.parse(f.read()):
        print("Failed to parse ONNX:")
        for error in range(parser.num_errors):
            print(f"  {parser.get_error(error)}")
        sys.exit(1)

print("✓ ONNX parsed")

# Build engine
print("Building TensorRT engine (may take 2-5 minutes)...")
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("  FP16 enabled")

engine = builder.build_serialized_network(network, config)

if engine:
    with open(TRT_PATH, 'wb') as f:
        f.write(engine)
    
    size_mb = Path(TRT_PATH).stat().st_size / (1024 * 1024)
    print(f"\n✅ SUCCESS!")
    print(f"TensorRT engine: {TRT_PATH}")
    print(f"Size: {size_mb:.2f} MB")
else:
    print("\n❌ Failed to build engine")