#!/usr/bin/env python3
"""
Convert ONNX Model to TensorRT Engine
For optimized inference on Jetson Nano
RUN THIS ON JETSON NANO, NOT ON LAPTOP
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pathlib import Path
import argparse
import json


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_file_path, engine_file_path, fp16_mode=True, max_batch_size=1):
    """
    Build TensorRT engine from ONNX model
    
    Args:
        onnx_file_path: Path to ONNX model
        engine_file_path: Path to save TensorRT engine
        fp16_mode: Enable FP16 precision (faster on Jetson)
        max_batch_size: Maximum batch size
        
    Returns:
        TensorRT engine
    """
    print("\n" + "="*80)
    print("BUILDING TENSORRT ENGINE")
    print("="*80)
    
    print(f"\nONNX Model: {onnx_file_path}")
    print(f"Output Engine: {engine_file_path}")
    print(f"FP16 Mode: {fp16_mode}")
    print(f"Max Batch Size: {max_batch_size}")
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print("\n📦 Parsing ONNX model...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('❌ ERROR: Failed to parse ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print("✓ ONNX model parsed successfully")
    
    # Build engine
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 28  # 256MB
    
    # Enable FP16 if supported
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled")
    else:
        print("⚠ FP16 mode not available or disabled")
    
    # Set optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    input_shape = network.get_input(0).shape
    
    # Get sequence length and features from input shape
    seq_len = input_shape[1]
    features = input_shape[2]
    
    # Set dynamic batch size range
    min_shape = (1, seq_len, features)
    opt_shape = (1, seq_len, features)
    max_shape = (max_batch_size, seq_len, features)
    
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    print(f"\n🔧 Building engine (this may take several minutes)...")
    print(f"   Input shape: {input_shape}")
    print(f"   Dynamic batch size: 1 to {max_batch_size}")
    
    # Build engine
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("❌ ERROR: Failed to build engine")
        return None
    
    print("✓ Engine built successfully")
    
    # Serialize engine
    print(f"\n💾 Serializing engine to: {engine_file_path}")
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    
    # Get file size
    file_size = Path(engine_file_path).stat().st_size
    print(f"✓ Engine saved ({file_size / 1024:.2f} KB)")
    
    return engine


def test_engine(engine_file_path, metadata_path):
    """
    Test TensorRT engine with dummy input
    
    Args:
        engine_file_path: Path to TensorRT engine
        metadata_path: Path to metadata JSON
    """
    print("\n" + "="*80)
    print("TESTING TENSORRT ENGINE")
    print("="*80)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    input_shape = tuple(metadata['input_shape'])
    
    # Load engine
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Allocate buffers
    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(metadata['output_shape'], dtype=np.float32)
    
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, h_input, stream)
    
    # Run inference
    context.execute_async_v2(
        bindings=[int(d_input), int(d_output)],
        stream_handle=stream.handle
    )
    
    # Transfer predictions back
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    
    # Print results
    predicted_class = np.argmax(h_output[0])
    confidence = np.max(h_output[0])
    
    print(f"\n✓ Test inference successful")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {confidence:.4f}")
    
    # Benchmark
    print(f"\n⏱️ Running benchmark (100 iterations)...")
    import time
    
    times = []
    for _ in range(100):
        start = time.time()
        
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(
            bindings=[int(d_input), int(d_output)],
            stream_handle=stream.handle
        )
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    fps = 1000 / avg_time
    
    print(f"  Average inference time: {avg_time:.2f} ms")
    print(f"  Throughput: {fps:.1f} FPS")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument('onnx_file', type=str,
                       help='Path to ONNX model file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output TensorRT engine file (default: model.trt)')
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Enable FP16 precision (default: True)')
    parser.add_argument('--max_batch_size', type=int, default=1,
                       help='Maximum batch size (default: 1)')
    parser.add_argument('--test', action='store_true',
                       help='Test engine after building')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        args.output = Path(args.onnx_file).with_suffix('.trt')
    
    # Build engine
    engine = build_engine(
        args.onnx_file,
        args.output,
        fp16_mode=args.fp16,
        max_batch_size=args.max_batch_size
    )
    
    if engine is None:
        print("\n❌ Failed to build engine")
        return
    
    # Test engine
    if args.test:
        metadata_path = Path(args.onnx_file).with_suffix('.json')
        if metadata_path.exists():
            test_engine(args.output, metadata_path)
        else:
            print(f"\n⚠ Metadata file not found: {metadata_path}")
            print("   Skipping test")
    
    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"\nTensorRT Engine: {args.output}")
    print(f"\nNext step:")
    print(f"  python jetson_inference.py --engine {args.output}")


if __name__ == "__main__":
    main()