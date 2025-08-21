import torch
import torch.nn.functional as F
from torch import nn


class TwoWayReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, temperature=1.0):
        ctx.save_for_backward(z)
        ctx.temperature = temperature
        return F.relu(z)

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        temp = ctx.temperature

        gate = F.sigmoid(z / temp)

        return grad_output * gate, None


class TwoWayReLU(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        return TwoWayReLUFunction.apply(x, self.temperature)

    def extra_repr(self):
        return f"temperature={self.temperature}"


class SoftMaxPool2d(nn.MaxPool2d):
    def __init__(self, *args, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, x):
        B, C, H, W = x.shape
        kH, kW = self.kernel_size, self.kernel_size

        # Unfold input to patches
        x_unf = F.unfold(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        x_unf = x_unf.view(B, C, kH * kW, -1)

        # Softmax pooling over spatial positions
        weights = F.softmax(x_unf / self.temperature, dim=2)
        pooled = (x_unf * weights).sum(dim=2)

        # Reshape back to image
        out_H = (H + 2 * self.padding - kH) // self.stride + 1
        out_W = (W + 2 * self.padding - kW) // self.stride + 1
        return pooled.view(B, C, out_H, out_W)

    def extra_repr(self):
        ret = super().extra_repr()
        return f"{ret}, temperature={self.temperature}"


class SurrogateSoftMaxPool2d(SoftMaxPool2d):
    def forward(self, x):
        soft = super().forward(x)
        hard = F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )

        return hard.detach() + (soft - soft.detach())
