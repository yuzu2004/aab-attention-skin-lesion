# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 2.1  SE, CBAM, ECA, Triplet Attention (references, as in the paper)
# =============================================================================
class SE(nn.Module):
    """Hu et al., CVPR 2018."""

    def __init__(self, c, r=16):
        super().__init__()
        h = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.ReLU(inplace=True), nn.Linear(h, c))

    def forward(self, x):
        return x * torch.sigmoid(self.fc(x.mean((2, 3))))[:, :, None, None]


class CBAM(nn.Module):
    """Woo et al., ECCV 2018: channel (avg+max, shared MLP) then spatial (7x7)."""

    def __init__(self, c, r=16, k=7):
        super().__init__()
        h = max(c // r, 4)
        self.mlp = nn.Sequential(nn.Linear(c, h), nn.ReLU(inplace=True), nn.Linear(h, c))
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3))))[:, :, None, None]
        x = x * ca
        sa = torch.sigmoid(self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


class ECA(nn.Module):
    """Wang et al., CVPR 2020: 1-D conv over channels, adaptive kernel size."""

    def __init__(self, c, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(c) + b) / gamma)); k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = self.conv(x.mean((2, 3)).unsqueeze(1)).squeeze(1)
        return x * torch.sigmoid(y)[:, :, None, None]


class _ZGate(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False); self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        z = torch.cat([x.amax(1, keepdim=True), x.mean(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.bn(self.conv(z)))


class TripletAttention(nn.Module):
    """Misra et al., WACV 2021: three branches (C-W, C-H, H-W), averaged."""

    def __init__(self, c=None):
        super().__init__()
        self.cw, self.hc, self.hw = _ZGate(), _ZGate(), _ZGate()

    def forward(self, x):
        a = self.cw(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        b = self.hc(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return (a + b + self.hw(x)) / 3.0
