from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- Building blocks ------------------------------
def autopad(k: int, p: int | None = None) -> int:
    return k // 2 if p is None else p


class Conv(nn.Module):
    def __init__(self, ci: int, co: int, k: int = 3, s: int = 1, p: int | None = None, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(co)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, ci: int, co: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(co * e)
        self.cv1 = Conv(ci, c_, 1)
        self.cv2 = Conv(c_, co, 3)
        self.add = shortcut and ci == co

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):

    def __init__(self, ci: int, co: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(co * e)
        self.cv1 = Conv(ci, c_, 1)
        self.cv2 = Conv(ci, c_, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)])
        self.cv3 = Conv(2 * c_, co, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class SPPF(nn.Module):

    def __init__(self, ci: int, co: int, k: int = 5):
        super().__init__()
        c_ = ci // 2
        self.cv1 = Conv(ci, c_, 1)
        self.cv2 = Conv(c_ * 4, co, 1)
        self.pool = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# --------------------------- Backbone -------------------------------------
class CSPDarknetLite(nn.Module):

    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.stem = Conv(in_ch, c1, k=3, s=2)     # /2  -> 128
        self.stage2 = nn.Sequential(Conv(c1, c2, 3, 2), C3(c2, c2, n=1))    # /4  -> 64
        self.stage3 = nn.Sequential(Conv(c2, c3, 3, 2), C3(c3, c3, n=2))    # /8  -> 32
        self.stage4 = nn.Sequential(Conv(c3, c4, 3, 2), C3(c4, c4, n=2))    # /16 -> 16
        self.stage5 = nn.Sequential(Conv(c4, c5, 3, 2), C3(c5, c5, n=1), SPPF(c5, c5))  # /32 -> 8
        self.out_channels = (c3, c4, c5)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5


# --------------------------- PAN-FPN Neck --------------------------------
class PANFPN(nn.Module):
    def __init__(self, chs: Tuple[int, int, int]):
        super().__init__()
        c3, c4, c5 = chs
        # top-down
        self.lat_p5 = Conv(c5, c4, 1)
        self.td_p4 = C3(c4 * 2, c4, n=1, shortcut=False)
        self.lat_p4 = Conv(c4, c3, 1)
        self.td_p3 = C3(c3 * 2, c3, n=1, shortcut=False)
        # bottom-up
        self.dn_p3 = Conv(c3, c3, 3, 2)
        self.bu_p4 = C3(c3 + c4, c4, n=1, shortcut=False)
        self.dn_p4 = Conv(c4, c4, 3, 2)
        self.bu_p5 = C3(c4 + c4, c5, n=1, shortcut=False)
        self.out_channels = (c3, c4, c5)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        p5_lat = self.lat_p5(p5)
        p4_td = self.td_p4(torch.cat([F.interpolate(p5_lat, scale_factor=2, mode="nearest"), p4], dim=1))
        p4_lat = self.lat_p4(p4_td)
        p3_out = self.td_p3(torch.cat([F.interpolate(p4_lat, scale_factor=2, mode="nearest"), p3], dim=1))
        p4_out = self.bu_p4(torch.cat([self.dn_p3(p3_out), p4_td], dim=1))
        p5_out = self.bu_p5(torch.cat([self.dn_p4(p4_out), p5_lat], dim=1))
        return p3_out, p4_out, p5_out


# --------------------------- Detection head ------------------------------
class DetectHead(nn.Module):
    """
    For a single class, the head channels per anchor = 4 (bbox) + 1 (obj) + 1 (malignancy) = 6.
    Output shape: (B, A*6, H, W).
    """

    def __init__(self, in_ch: int, num_anchors: int):
        super().__init__()
        self.num_anchors = num_anchors
        self.channels_per_anchor = 6
        self.conv = nn.Conv2d(in_ch, num_anchors * self.channels_per_anchor, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# --------------------------- Full model -----------------------------------
class YoloLIDC(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        backbone: str = "cspdarknet_lite",
        num_anchors_per_scale: int = 3,
        strides: Sequence[int] = (8, 16, 32),
        base: int = 32,
    ):
        super().__init__()
        assert backbone in ("cspdarknet_lite", "resnet18"), backbone
        if backbone == "cspdarknet_lite":
            self.backbone = CSPDarknetLite(in_ch=in_channels, base=base)
            bb_channels = self.backbone.out_channels
        else:
            self.backbone = _ResNet18Adapter(in_channels=in_channels)
            bb_channels = self.backbone.out_channels
        self.neck = PANFPN(bb_channels)
        neck_ch = self.neck.out_channels
        self.strides = tuple(strides)
        self.num_anchors_per_scale = num_anchors_per_scale
        self.heads = nn.ModuleList(
            [DetectHead(c, num_anchors_per_scale) for c in neck_ch]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return [h(f) for h, f in zip(self.heads, (p3, p4, p5))]


# --------------------------- Optional ResNet18 backbone -------------------
class _ResNet18Adapter(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        from torchvision.models import resnet18
        net = resnet18(weights=None)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)  # /4
        self.layer1 = net.layer1                                             # /4
        self.layer2 = net.layer2                                             # /8
        self.layer3 = net.layer3                                             # /16
        self.layer4 = net.layer4                                             # /32
        self.out_channels = (128, 256, 512)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        p3 = self.layer2(x)   # /8
        p4 = self.layer3(p3)  # /16
        p5 = self.layer4(p4)  # /32
        return p3, p4, p5