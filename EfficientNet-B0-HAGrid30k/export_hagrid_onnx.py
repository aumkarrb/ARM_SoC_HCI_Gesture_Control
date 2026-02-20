"""
export_hagrid_onnx.py — Export trained EfficientNet-B0 to ONNX for Jetson
==========================================================================
Usage:
  python3 export_hagrid_onnx.py --ckpt checkpoints_hagrid/best_model.pth
"""

import argparse, json
import torch
import torchvision.models as tvm
import torch.nn as nn
from pathlib import Path


def build_model(num_classes, dropout=0.4):
    model = tvm.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def main(args):
    ckpt     = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    classes  = ckpt['classes']
    model    = build_model(len(classes))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    dummy   = torch.randn(1, 3, 224, 224)
    out_path = Path(args.ckpt).parent / 'gesture_model.onnx'

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=11,
    )

    # Save class list alongside model
    json.dump(classes, open(Path(args.ckpt).parent / 'classes.json', 'w'))

    print(f"✅ Exported to {out_path}")
    print(f"   Classes: {classes}")
    print(f"   Input  : (1, 3, 224, 224)")
    print(f"\n   Copy to Jetson:")
    print(f"   scp {out_path} user@JETSON_IP:~/gesture/")
    print(f"   scp {Path(args.ckpt).parent}/classes.json user@JETSON_IP:~/gesture/")

    # Verify
    try:
        import onnx
        m = onnx.load(str(out_path))
        onnx.checker.check_model(m)
        print(f"\n✅ ONNX model is valid")
    except ImportError:
        print("\n  (Install onnx to verify: pip install onnx --break-system-packages)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='checkpoints_hagrid/best_model.pth')
    main(p.parse_args())