"""
Gesture Recognition Models - CNN+LSTM Architecture
Based on research papers:
- MediaPipe + LSTM for 98.99% accuracy (NIT Raipur, 2023)
- CNN+LSTM on Jetson Nano for gesture commands (2025)

Architecture:
  MediaPipe extracts 21 hand landmarks (42 features/frame)
  CNN extracts spatial features from landmark positions per frame
  BiLSTM captures temporal motion patterns across frames
  Attention focuses on most discriminative frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCNN(nn.Module):
    """
    CNN block that processes spatial arrangement of 21 landmarks per frame.
    Input:  (batch*seq_len, 1, 42)
    Output: (batch*seq_len, feature_dim)
    """
    def __init__(self, input_size=42, feature_dim=256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),   # 42 → 21

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),                 # → (B*T, feature_dim, 1)
        )

    def forward(self, x):
        out = self.conv_block(x)
        return out.squeeze(-1)                        # (B*T, feature_dim)


class AttentionModule(nn.Module):
    """
    Temporal attention: lets model focus on most discriminative frames.
    """
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, lstm_out):
        # lstm_out: (B, T, hidden_dim)
        weights = F.softmax(self.attn(lstm_out), dim=1)   # (B, T, 1)
        attended = (lstm_out * weights).sum(dim=1)         # (B, hidden_dim)
        return attended, weights


class GestureCNNLSTM(nn.Module):
    """
    Full CNN+LSTM model for gesture recognition.

    Pipeline:
      1. (B, T, 42) → reshape → (B*T, 1, 42)
      2. SpatialCNN → (B*T, 256)
      3. reshape → (B, T, 256)
      4. BiLSTM → all outputs (B, T, 512) + final hidden (B, 512)
      5. Attention → weighted context (B, 512)
      6. Concat [final_hidden + attention] → (B, 1024)
      7. FC classifier → (B, num_classes)

    GPU Usage:
      RTX 2050: ~90-100% during training with batch_size=32
    """
    def __init__(self,
                 input_size=42,
                 seq_len=30,
                 cnn_feature_dim=256,
                 lstm_hidden_dim=256,
                 lstm_layers=2,
                 num_classes=6,
                 dropout=0.4):
        super().__init__()
        self.seq_len = seq_len
        self.cnn_feature_dim = cnn_feature_dim

        # 1. Spatial CNN
        self.spatial_cnn = SpatialCNN(input_size, cnn_feature_dim)

        # 2. Bidirectional LSTM (cuDNN-optimized on GPU)
        self.bilstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim * 2)

        # 3. Attention
        self.attention = AttentionModule(hidden_dim=lstm_hidden_dim * 2)

        # 4. Classifier
        fc_in = lstm_hidden_dim * 2 * 2  # attention + final hidden
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),

            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if   'weight_ih' in name: nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
                    elif 'bias'      in name: nn.init.zeros_(param.data)

    def forward(self, x):
        B = x.size(0)

        # --- CNN spatial features ---
        x_cnn = x.contiguous().view(B * self.seq_len, 1, -1)   # (B*T, 1, 42)
        feats  = self.spatial_cnn(x_cnn)                         # (B*T, 256)
        feats  = feats.view(B, self.seq_len, self.cnn_feature_dim)  # (B, T, 256)

        # --- BiLSTM temporal dynamics ---
        lstm_out, (hidden, _) = self.bilstm(feats)               # (B, T, 512), (4, B, 256)
        lstm_out = F.dropout(lstm_out, p=0.3, training=self.training)

        # Final hidden: concat last fwd + last bwd
        final_h = torch.cat([hidden[-2], hidden[-1]], dim=1)     # (B, 512)
        final_h = self.layer_norm(final_h)

        # --- Attention ---
        attended, _ = self.attention(lstm_out)                   # (B, 512)

        # --- Combine & classify ---
        combined = torch.cat([final_h, attended], dim=1)         # (B, 1024)
        return self.classifier(combined)

    def count_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class LightweightCNNLSTM(nn.Module):
    """
    Lighter version for Jetson Nano if full model is too slow.
    ~2x faster inference, ~2% lower accuracy.
    """
    def __init__(self, input_size=42, seq_len=30, num_classes=6, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True,
                            bidirectional=False, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B * self.seq_len, 1, -1)
        x = self.cnn(x).squeeze(-1).view(B, self.seq_len, -1)
        _, (h, _) = self.lstm(x)
        return self.classifier(h[-1])


def build_model(model_type='cnn_lstm', num_classes=6, seq_len=30):
    if model_type == 'cnn_lstm':
        model = GestureCNNLSTM(num_classes=num_classes, seq_len=seq_len)
    elif model_type == 'lightweight':
        model = LightweightCNNLSTM(num_classes=num_classes, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if hasattr(model, 'count_params'):
        total, trainable = model.count_params()
        print(f"\n{'='*55}")
        print(f"  Model      : {model_type.upper()} (CNN + BiLSTM + Attention)")
        print(f"  Parameters : {total:,} total / {trainable:,} trainable")
        print(f"  Input      : (batch, {seq_len}, 42)  →  Output: (batch, {num_classes})")
        print(f"  Expected   : 95-99% accuracy (papers: 98.99%)")
        print(f"{'='*55}\n")
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = build_model('cnn_lstm', num_classes=6, seq_len=30).to(device)
    x = torch.randn(32, 30, 42).to(device)
    y = model(x)
    print(f"Input: {x.shape}  →  Output: {y.shape}  ✅")