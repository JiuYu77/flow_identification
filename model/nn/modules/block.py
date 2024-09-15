# -*- coding: UTF-8 -*-
from torch import nn
from conv import Conv1d
import torch
from utils.torch_utils import fuse_conv_and_bn

class C2f1d(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5) -> None:
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.conv1 = Conv1d(c1, 2 * self.c, 1, 1)
        self.conv2 = Conv1d((2 + n) * self.c, c2, 1)
        # self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(1, 1), e=1.0) for _ in range(n))
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f1d layer."""
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.conv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.conv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.conv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5) -> None:
        super().__init__()
        c_ = int(c2 * e) # hidden channels
        self.conv1 = Conv1d(c1, c_, k[0], 1)
        self.conv2 = Conv1d(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class RepVGGDW1d(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv1d(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv1d(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1

class CIB1d(nn.Module):
    """YOLOv10.  Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv1d(c1, c1, 3, g=c1),
            Conv1d(c1, 2 * c_, 1),
            Conv1d(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW1d(2 * c_),
            Conv1d(2 * c_, c2, 1),
            Conv1d(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)

class C2fCIB1d(C2f1d):
    """YOLOv10.  Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB1d(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))

class Attention(nn.Module):
    '''YOLOv10'''
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv1d(dim, h, 1, act=False)
        self.proj = Conv1d(dim, dim, 1, act=False)
        self.pe = Conv1d(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # B, H, W = x.shape; C = 1
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSA1d(nn.Module):
    '''YOLOv10'''
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv1d(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv1d(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            Conv1d(self.c, self.c*2, 1),
            Conv1d(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        # a, b = self.cv1(x).split((self.c, self.c), dim=1)
        a, b = self.cv1(x).split((self.c, self.c), dim=-2)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

class SCDown1d(nn.Module):
    '''YOLOv10'''
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = Conv1d(c1, c2, 1, 1)
        self.cv2 = Conv1d(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))
