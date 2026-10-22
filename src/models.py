import torch, torch.nn as nn

class SE(nn.Module):
    """Squeeze-and-excitation: global pool -> bottleneck MLP -> sigmoid channel weights."""
    def __init__(self, c, r=4):
        super().__init__()
        h = max(c // r, 8)
        self.fc1, self.fc2 = nn.Linear(c, h), nn.Linear(h, c)
    def forward(self, x):
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(x.mean((2, 3))))))
        return x * s[:, :, None, None]

class DSBlock(nn.Module):
    """Depthwise 3x3 -> pointwise 1x1 -> (SE) -> (+ shortcut) -> ReLU."""
    def __init__(self, cin, cout, stride, residual, se):
        super().__init__()
        self.dw = nn.Sequential(nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
                                nn.BatchNorm2d(cin), nn.ReLU(inplace=True))
        self.pw = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.se = SE(cout) if se else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        if residual:
            same = (stride == 1 and cin == cout)
            self.short = nn.Identity() if same else nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        else:
            self.short = None
    def forward(self, x):
        y = self.se(self.pw(self.dw(x)))
        if self.short is not None: y = y + self.short(x)
        return self.act(y)

CFG = [(32, 2, 2), (64, 2, 2), (128, 3, 2), (256, 2, 2)]    # (channels, blocks, first-block stride)

class ArogyaNet(nn.Module):
    def __init__(self, residual=True, se=True, meta=False, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 24, 3, 2, 1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True))
        layers, c = [], 24
        for co, n, s in CFG:
            for i in range(n):
                layers.append(DSBlock(c, co, s if i == 0 else 1, residual, se)); c = co
        self.features = nn.Sequential(*layers)
        self.meta = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()) if meta else None
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c + (16 if meta else 0), 1)
    def forward(self, x, m=None):                            # x: B,1,H,W in [0,1]; m: B,3 standardised
        f = self.features(self.stem((x - 0.5) / 0.25)).mean((2, 3))     # global average pooling
        if self.meta is not None: f = torch.cat([f, self.meta(m)], 1)
        return self.fc(self.drop(f))
