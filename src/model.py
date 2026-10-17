import torch
import torch.nn as nn

from src.config import CLASSES


class ChronoQRSNet(nn.Module):
    """
    Dual-Branch Spatio-Temporal 1D-CNN for Inter-Patient Arrhythmia Classification.
    - Branch 1 (Morphology): Multi-scale 1D convolutions over the 252-sample ECG beat.
    - Branch 2 (Chrono): Dense projection of local and raw RR-interval dynamics (4 features).
    """

    def __init__(self, num_classes: int = len(CLASSES), rr_dim: int = 4, dropout: float = 0.3):
        super().__init__()

        # Branch 1: 1D-CNN over 252-sample heartbeat waveform
        self.wave_branch = nn.Sequential(
            # Block 1: Wide kernel (15 samples = ~42 ms) to capture overall QRS width
            nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 252 -> 126

            # Block 2: Medium kernel (7 samples = ~19 ms) for morphological inflection points
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout1d(0.15),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 126 -> 63

            # Block 3: Narrow kernel (3 samples) for high-frequency QRS notches
            nn.Conv1d(64, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
        )

        # Dual pooling preserves both mean baseline shift and sharp R/S peak amplitudes
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Branch 2: Temporal RR-interval feature projection
        self.rr_branch = nn.Sequential(
            nn.Linear(rr_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        # Fusion Classifier: (96 avg + 96 max) + 32 RR = 224 features
        fusion_dim = (96 * 2) + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_wave: torch.Tensor, x_rr: torch.Tensor) -> torch.Tensor:
        # 1. Extract morphological features
        feat_w = self.wave_branch(x_wave)  # (B, 96, 63)
        w_avg = self.avg_pool(feat_w).squeeze(-1)  # (B, 96)
        w_max = self.max_pool(feat_w).squeeze(-1)  # (B, 96)
        wave_emb = torch.cat([w_avg, w_max], dim=1)  # (B, 192)

        # 2. Extract temporal RR features
        rr_emb = self.rr_branch(x_rr)  # (B, 32)

        # 3. Concatenate and classify
        fused = torch.cat([wave_emb, rr_emb], dim=1)  # (B, 224)
        logits = self.classifier(fused)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = ChronoQRSNet()
    dummy_wave = torch.randn(8, 1, 252)
    dummy_rr = torch.randn(8, 4)
    out = net(dummy_wave, dummy_rr)
    print(f"ChronoQRSNet Output Shape: {out.shape} | Trainable Params: {count_parameters(net):,}")