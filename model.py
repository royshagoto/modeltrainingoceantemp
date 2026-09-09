"""
model.py
========
Satellite-Embedding Deep Learning architecture for subsurface ocean
temperature reconstruction.

Design: Residual / Attention U-Net
-----------------------------------
Why this architecture for this problem:
  * The task is dense pixel-wise regression (H x W -> N_DEPTHS x H x W),
    so we need an architecture that outputs at the SAME spatial resolution
    as the input. A plain CNN encoder (e.g. ResNet backbone -> GAP -> FC)
    would collapse spatial structure; a U-Net-style encoder-decoder with
    skip connections preserves fine-scale frontal/eddy structure while
    still building a compact global "satellite embedding" at the bottleneck.
  * Residual blocks let us go moderately deep (needed to capture nonlinear
    thermocline-tilt / eddy-pumping relationships) without vanishing
    gradients, and train stably on the comparatively small daily-ocean
    datasets available (years, not millions of images).
  * Attention gates on the skip connections let the decoder down-weight
    land pixels / noisy coastal satellite retrievals and focus on
    dynamically active regions (western boundary currents, eddies,
    upwelling zones) that carry most of the subsurface signal.
  * The bottleneck feature map IS the "compact satellite embedding"
    requested in the problem statement; it is exposed via
    `AttentionResUNet.encode()` for downstream use / diagnostics.

The same class can be reused as a plain CNN (attention gates + depth
configurable) or extended with a ConvLSTM/temporal-attention head if you
want to model temporal evolution beyond simple channel-stacking of the
time_window.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Two 3x3 convs with GroupNorm + SiLU, residual skip, optional dropout."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class DownBlock(nn.Module):
    """Residual block followed by strided-conv downsampling."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, dropout)
        self.pool = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        skip = self.res(x)
        down = self.pool(skip)
        return down, skip


class AttentionGate(nn.Module):
    """
    Additive attention gate (Oktay et al., 2018 style).
    Gates encoder skip-connection features `x` using decoder features `g`,
    learning to suppress irrelevant (e.g. land / noisy coastal) regions.
    """

    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=True),
            nn.GroupNorm(min(8, inter_ch), inter_ch),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=True),
            nn.GroupNorm(min(8, inter_ch), inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        attn = self.psi(self.act(self.w_g(g) + self.w_x(x)))
        return x * attn


class UpBlock(nn.Module):
    """Upsample, (optionally attention-gate the skip), concat, residual block."""

    def __init__(self, in_ch, skip_ch, out_ch, use_attention=True, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionGate(gate_ch=in_ch, skip_ch=skip_ch,
                                       inter_ch=max(skip_ch // 2, 8))
        self.res = ResidualBlock(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_attention:
            skip = self.attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------
class AttentionResUNet(nn.Module):
    """
    Residual / Attention U-Net that maps stacked surface satellite fields
    to depth-wise subsurface temperature.

    Input : (B, time_window * n_input_channels, H, W)
    Output: (B, n_depths, H, W)
    """

    def __init__(self,
                 in_channels=C.TIME_WINDOW * C.N_INPUT_CHANNELS,
                 out_channels=C.N_DEPTHS,
                 base_ch=C.BASE_CHANNELS,
                 embedding_dim=C.EMBEDDING_DIM,
                 use_attention=C.USE_ATTENTION_GATES,
                 dropout=C.DROPOUT):
        super().__init__()

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.stem = ResidualBlock(in_channels, c1, dropout)

        # Encoder (4 downsampling stages)
        self.down1 = DownBlock(c1, c1, dropout)
        self.down2 = DownBlock(c1, c2, dropout)
        self.down3 = DownBlock(c2, c3, dropout)
        self.down4 = DownBlock(c3, c4, dropout)

        # Bottleneck == compact "satellite embedding"
        self.bottleneck = nn.Sequential(
            ResidualBlock(c4, embedding_dim, dropout),
            ResidualBlock(embedding_dim, embedding_dim, dropout),
        )

        # Decoder (mirrors encoder, with attention-gated skip connections)
        self.up4 = UpBlock(embedding_dim, c4, c3, use_attention, dropout)
        self.up3 = UpBlock(c3, c3, c2, use_attention, dropout)
        self.up2 = UpBlock(c2, c2, c1, use_attention, dropout)
        self.up1 = UpBlock(c1, c1, c1, use_attention, dropout)

        # Depth-wise regression head. A 1x1 conv projects to n_depths;
        # a small extra conv smooths spurious high-frequency noise.
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, out_channels, 1),
        )

        # Learnable per-depth bias/scale lets the network start close to a
        # sensible climatological temperature profile (warm at surface,
        # cold at depth) before fine-tuning on anomalies -- speeds up
        # convergence substantially versus a naive zero init.
        self.depth_scale = nn.Parameter(torch.ones(out_channels, 1, 1) * 5.0)
        self.depth_bias = nn.Parameter(
            torch.linspace(28.0, 4.0, out_channels).view(-1, 1, 1)
        )

    def encode(self, x):
        """Return only the bottleneck satellite-embedding feature map."""
        x = self.stem(x)
        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        x, s3 = self.down3(x)
        x, s4 = self.down4(x)
        emb = self.bottleneck(x)
        return emb

    def forward(self, x, return_embedding=False):
        x0 = self.stem(x)
        x1, s1 = self.down1(x0)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4, s4 = self.down4(x3)

        emb = self.bottleneck(x4)          # (B, embedding_dim, H/16, W/16)

        d4 = self.up4(emb, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        raw = self.head(d1)                                  # (B, n_depths, H, W)
        out = raw * torch.tanh(self.depth_scale) + self.depth_bias

        if return_embedding:
            # global-average-pooled embedding vector, useful for downstream
            # diagnostics / clustering of "ocean states"
            pooled = emb.mean(dim=[2, 3])
            return out, pooled
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
class MaskedDepthWeightedLoss(nn.Module):
    """
    Masked Huber (Smooth-L1) loss, averaged only over valid (ocean/observed)
    pixels, with optional per-depth weighting (e.g. up-weight the upper
    thermocline where most subsurface variability / marine-heatwave signal
    lives).
    """

    def __init__(self, depth_levels=C.DEPTH_LEVELS, upweight_thermocline=True,
                 huber_beta=0.5):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=huber_beta, reduction="none")
        weights = torch.ones(len(depth_levels))
        if upweight_thermocline:
            for i, d in enumerate(depth_levels):
                if d <= 150:          # main thermocline zone in N. Indian Ocean
                    weights[i] = 1.5
        self.register_buffer("depth_weights", weights.view(1, -1, 1, 1))

    def forward(self, pred, target, mask):
        loss = self.huber(pred, target) * self.depth_weights
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom


if __name__ == "__main__":
    # quick shape / parameter-count sanity check
    model = AttentionResUNet()
    dummy = torch.randn(2, C.TIME_WINDOW * C.N_INPUT_CHANNELS, C.H, C.W)
    out = model(dummy)
    print("Input :", dummy.shape)
    print("Output:", out.shape)
    print(f"Trainable parameters: {model.num_parameters():,}")
