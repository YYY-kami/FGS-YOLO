"""Network modules required by FGS-YOLO."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, RepConv


class LSKA(nn.Module):
    """Large Separable Kernel Attention with the effective 11x11 kernel used by FGS-YOLO."""

    def __init__(self, dim, k_size=11):
        super().__init__()
        if k_size != 11:
            raise ValueError("FGS-YOLO uses LSKA with k_size=11.")
        self.conv0h = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv0v = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        self.conv_spatial_h = nn.Conv2d(dim, dim, (1, 5), padding=(0, 4), groups=dim, dilation=2)
        self.conv_spatial_v = nn.Conv2d(dim, dim, (5, 1), padding=(4, 0), groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        residual = x.clone()
        attention = self.conv0h(x)
        attention = self.conv0v(attention)
        attention = self.conv_spatial_h(attention)
        attention = self.conv_spatial_v(attention)
        attention = self.conv1(attention)
        return residual * attention


class SPPF_LSKA(nn.Module):
    """SPPF followed by LSKA spatial reweighting."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden * 4, c2, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(hidden * 4, k_size=11)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        fused = torch.cat((x, y1, y2, self.pool(y2)), 1)
        return self.cv2(self.lska(fused))


class BasicBlock_3x3_Reverse(nn.Module):
    """RepConv-first reverse basic block used inside CSPStage."""

    def __init__(self, ch_in, ch_hidden_ratio, ch_out, shortcut=True):
        super().__init__()
        if ch_in != ch_out:
            raise ValueError("BasicBlock_3x3_Reverse requires ch_in == ch_out.")
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = Conv(ch_hidden, ch_out, 3, s=1)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, s=1)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(self.conv2(x))
        return x + y if self.shortcut else y


class CSPStage(nn.Module):
    """Cross-stage partial aggregation used at GDFPN fusion nodes."""

    def __init__(self, ch_in, ch_out, n, ch_hidden_ratio=1.0):
        super().__init__()
        ch_first = ch_out // 2
        ch_mid = ch_out - ch_first
        self.conv1 = Conv(ch_in, ch_first, 1)
        self.conv2 = Conv(ch_in, ch_mid, 1)
        self.blocks = nn.ModuleList(
            BasicBlock_3x3_Reverse(ch_mid, ch_hidden_ratio, ch_mid, shortcut=True) for _ in range(n)
        )
        self.conv3 = Conv(ch_mid * n + ch_first, ch_out, 1)

    def forward(self, x):
        y2 = self.conv2(x)
        outputs = [self.conv1(x)]
        for block in self.blocks:
            y2 = block(y2)
            outputs.append(y2)
        return self.conv3(torch.cat(outputs, dim=1))


class DySample(nn.Module):
    """Content-aware dynamic upsampling; FGS-YOLO uses scale=2 and style='lp'."""

    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale, self.style, self.groups = scale, style, groups
        if style not in {"lp", "pl"}:
            raise ValueError("style must be 'lp' or 'pl'.")
        if style == "pl" and (in_channels < scale**2 or in_channels % scale**2):
            raise ValueError("Invalid channels for pixel-first DySample.")
        if in_channels < groups or in_channels % groups:
            raise ValueError("in_channels must be divisible by groups.")
        offset_in = in_channels // scale**2 if style == "pl" else in_channels
        offset_out = 2 * groups if style == "pl" else 2 * groups * scale**2
        self.offset = nn.Conv2d(offset_in, offset_out, 1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.offset.bias, 0.0)
        if dyscope:
            self.scope = nn.Conv2d(offset_in, offset_out, 1)
            nn.init.constant_(self.scope.weight, 0.0)
            nn.init.constant_(self.scope.bias, 0.0)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        grid = torch.stack(torch.meshgrid(h, h, indexing="ij")).transpose(1, 2)
        return grid.repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        batch, _, height, width = offset.shape
        offset = offset.view(batch, 2, -1, height, width)
        coords_h = torch.arange(height, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(width, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij")).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([width, height], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(batch, -1, height, width), self.scale)
        coords = coords.view(batch, 2, -1, self.scale * height, self.scale * width)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        sampled = F.grid_sample(
            x.reshape(batch * self.groups, -1, height, width), coords, mode="bilinear",
            align_corners=False, padding_mode="border"
        )
        return sampled.reshape(batch, -1, self.scale * height, self.scale * width)

    def forward(self, x):
        if self.style == "pl":
            x_shuffled = F.pixel_shuffle(x, self.scale)
            raw = self.offset(x_shuffled)
            raw = raw * self.scope(x_shuffled).sigmoid() * 0.5 if hasattr(self, "scope") else raw * 0.25
            offset = F.pixel_unshuffle(raw, self.scale) + self.init_pos
        else:
            raw = self.offset(x)
            offset = raw * self.scope(x).sigmoid() * 0.5 + self.init_pos if hasattr(self, "scope") else raw * 0.25 + self.init_pos
        return self.sample(x, offset)
