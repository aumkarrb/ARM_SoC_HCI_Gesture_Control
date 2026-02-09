"""
FreiHAND Hand Keypoint Training
- GPU/CUDA support
- Proper checkpointing
- TensorRT export
- Progress tracking
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.mobilenetv2_heatmap import MobileNetV2Heatmap
from datasets.freihand_dataset import FreiHANDDataset


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
ROOT = r"C:\Users\Soham\Downloads\FreiHAND_pub_v2"
BATCH_SIZE = 16           # Adjust based on GPU memory
EPOCHS = 30
LR = 1e-3
NUM_WORKERS = 0           # Set to 0 for Windows if you get errors
IMG_SIZE = 224
HEATMAP_SIZE = 7

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 60)
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
print("=" * 60)


# --------------------------------------------------
# DATASET
# --------------------------------------------------
print("\n📁 Loading FreiHAND dataset...")
try:
    dataset = FreiHANDDataset(
        root_dir=ROOT,
        img_size=IMG_SIZE,
        heatmap_size=HEATMAP_SIZE
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    sys.exit(1)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True if device == "cuda" else False
)

print(f"✓ DataLoader ready: {len(loader)} batches per epoch")


# --------------------------------------------------
# MODEL
# --------------------------------------------------
print("\n🤖 Initializing MobileNetV2 model...")
try:
    model = MobileNetV2Heatmap(num_keypoints=21, in_channels=1)
    model.to(device)
    print(f"✓ Model loaded on {device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------
print("\n🚀 Starting training...\n")
model.train()

best_loss = float('inf')
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    
    for batch_idx, (img, target) in enumerate(loader):
        # Move to device
        img = img.to(device)
        target = target.to(device)
        
        # Forward pass
        pred = model(img)
        loss = F.mse_loss(pred, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Print progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            avg_batch_loss = epoch_loss / (batch_idx + 1)
            progress = (batch_idx + 1) / len(loader) * 100
            print(f"Epoch [{epoch+1}/{EPOCHS}] [{progress:>5.1f}%] "
                  f"Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.6f} (Avg: {avg_batch_loss:.6f})")
    
    # Epoch statistics
    avg_loss = epoch_loss / len(loader)
    
    print("=" * 60)
    print(f"EPOCH {epoch + 1}/{EPOCHS} COMPLETE")
    print(f"Average Loss: {avg_loss:.6f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print("=" * 60 + "\n")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, output_dir / "best_freihand_model.pth")
        print(f"✅ Best model saved! Loss: {avg_loss:.6f}\n")
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}\n")
    
    # Update learning rate
    scheduler.step()

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)


# --------------------------------------------------
# SAVE FINAL MODEL
# --------------------------------------------------
final_path = output_dir / "freihand_final.pth"
torch.save(model.state_dict(), final_path)
print(f"\n✓ Final model saved: {final_path}")


# --------------------------------------------------
# EXPORT TO ONNX
# --------------------------------------------------
print("\n📤 Exporting to ONNX for TensorRT...")
model.eval()

dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)
onnx_path = output_dir / "freihand_model.onnx"

try:
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['heatmaps'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'heatmaps': {0: 'batch_size'}
        }
    )
    print(f"✓ ONNX model saved: {onnx_path}")
except Exception as e:
    print(f"⚠ ONNX export failed: {e}")


# --------------------------------------------------
# CONVERT TO TENSORRT (OPTIONAL)
# --------------------------------------------------
print("\n⚡ Converting to TensorRT...")

try:
    import tensorrt_rtx as trt
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    def build_engine(onnx_file_path):
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Build engine
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Enable FP16 if supported
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ FP16 mode enabled")
        
        serialized_engine = builder.build_serialized_network(network, config)
        return serialized_engine
    
    engine = build_engine(str(onnx_path))
    
    if engine:
        trt_path = output_dir / "freihand_model.trt"
        with open(trt_path, 'wb') as f:
            f.write(engine)
        print(f"✓ TensorRT engine saved: {trt_path}")
        print("✓ TensorRT conversion successful!")
    
except ImportError:
    print("⚠ TensorRT not available (tensorrt_rtx not imported)")
    print("  You can convert ONNX to TensorRT manually using trtexec:")
    print(f"  trtexec --onnx={onnx_path} --saveEngine=freihand_model.trt")
except Exception as e:
    print(f"⚠ TensorRT conversion failed: {e}")
    print("  ONNX model is still available for inference")


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
print("\n" + "=" * 60)
print("📊 TRAINING SUMMARY")
print("=" * 60)
print(f"Best Loss: {best_loss:.6f}")
print(f"\nOutput files in '{output_dir}':")
print(f"  • best_freihand_model.pth  (best checkpoint)")
print(f"  • freihand_final.pth       (final weights)")
print(f"  • freihand_model.onnx      (ONNX format)")
if (output_dir / "freihand_model.trt").exists():
    print(f"  • freihand_model.trt       (TensorRT engine)")
print("=" * 60)