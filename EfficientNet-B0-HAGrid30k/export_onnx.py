"""
Export trained CNN+LSTM model to ONNX format for Jetson Nano deployment.
"""

import torch
import argparse
import numpy as np
from pathlib import Path
from models import GestureCNNLSTM


def export_onnx(model_path, output_path, seq_len=30, input_size=42):
    # Load checkpoint
    ckpt = torch.load(model_path, map_location='cpu')
    num_classes = ckpt['num_classes']
    classes     = ckpt['classes']
    seq_len     = ckpt.get('seq_len', seq_len)

    print(f"Model classes: {classes}")
    print(f"Num classes:   {num_classes}")
    print(f"Seq length:    {seq_len}")

    # Build and load model
    model = GestureCNNLSTM(input_size=input_size, seq_len=seq_len,
                           num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Dummy input
    dummy = torch.randn(1, seq_len, input_size)

    # Export
    torch.onnx.export(
        model, dummy, output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['keypoints'],
        output_names=['gesture_logits'],
        dynamic_axes={
            'keypoints':       {0: 'batch_size'},
            'gesture_logits':  {0: 'batch_size'}
        }
    )

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"\n✅ ONNX exported → {output_path}  ({size_mb:.1f} MB)")
    print(f"   Classes: {classes}")

    # Save class mapping
    import json
    mapping_path = Path(output_path).with_suffix('.json')
    with open(mapping_path, 'w') as f:
        json.dump({'classes': classes, 'seq_len': seq_len,
                   'input_size': input_size}, f, indent=2)
    print(f"   Mapping → {mapping_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',  default='checkpoints/best_model.pth')
    p.add_argument('--output_path', default='gesture_model.onnx')
    p.add_argument('--seq_len',     type=int, default=30)
    args = p.parse_args()
    export_onnx(args.model_path, args.output_path, args.seq_len)