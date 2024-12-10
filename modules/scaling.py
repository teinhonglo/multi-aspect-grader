import logging
import math
import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

class LimitParamValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        assert max >= min
        ctx.min = min
        ctx.max = max
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x,) = ctx.saved_tensors
        # where x < ctx.min, ensure all grads are negative (this will tend to make
        # x more positive).
        x_grad = x_grad * torch.where(
            torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0
        )
        # where x > ctx.max, ensure all grads are positive (this will tend to make
        # x more negative).
        x_grad *= torch.where(torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None

def limit_param_value(
    x: Tensor, min: float, max: float, prob: float = 0.6, training: bool = True
):
    # You apply this to (typically) an nn.Parameter during training to ensure that its
    # (elements mostly) stays within a supplied range.  This is done by modifying the
    # gradients in backprop.
    # It's not necessary to do this on every batch: do it only some of the time,
    # to save a little time.
    if training and random.random() < prob:
        return LimitParamValue.apply(x, min, max)
    else:
        return x
        
class BiasNormFunction(torch.autograd.Function):
    # This computes:
    #   scales = (torch.mean((x - bias) ** 2, keepdim=True)) ** -0.5 * log_scale.exp()
    #   return x * scales
    # (after unsqueezing the bias), but it does it in a memory-efficient way so that
    # it can just store the returned value (chances are, this will also be needed for
    # some other reason, related to the next operation, so we can save memory).
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        bias: Tensor,
        log_scale: Tensor,
        channel_dim: int,
        store_output_for_backprop: bool,
    ) -> Tensor:
        assert bias.ndim == 1
        if channel_dim < 0:
            channel_dim = channel_dim + x.ndim
        ctx.store_output_for_backprop = store_output_for_backprop
        ctx.channel_dim = channel_dim
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        scales = (
            torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
        ) * log_scale.exp()
        ans = x * scales
        ctx.save_for_backward(
            ans.detach() if store_output_for_backprop else x,
            scales.detach(),
            bias.detach(),
            log_scale.detach(),
        )
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        ans_or_x, scales, bias, log_scale = ctx.saved_tensors
        if ctx.store_output_for_backprop:
            x = ans_or_x / scales
        else:
            x = ans_or_x
        x = x.detach()
        x.requires_grad = True
        bias.requires_grad = True
        log_scale.requires_grad = True
        with torch.enable_grad():
            # recompute scales from x, bias and log_scale.
            scales = (
                torch.mean((x - bias) ** 2, dim=ctx.channel_dim, keepdim=True) ** -0.5
            ) * log_scale.exp()
            ans = x * scales
            ans.backward(gradient=ans_grad)
        return x.grad, bias.grad.flatten(), log_scale.grad, None, None

class BiasNorm(nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    Instead, we give the BiasNorm a trainable bias that it can use when
    computing the scale for normalization.  We also give it a (scalar)
    trainable scale on the output.


    Args:
       num_channels: the number of channels, e.g. 512.
       channel_dim: the axis/dimension corresponding to the channel,
         interpreted as an offset from the input's ndim if negative.
         This is NOT the num_channels; it should typically be one of
         {-2, -1, 0, 1, 2, 3}.
      log_scale: the initial log-scale that we multiply the output by; this
         is learnable.
      log_scale_min: FloatLike, minimum allowed value of log_scale
      log_scale_max: FloatLike, maximum allowed value of log_scale
      store_output_for_backprop: only possibly affects memory use; recommend
         to set to True if you think the output of this module is more likely
         than the input of this module to be required to be stored for the
         backprop.
    """

    def __init__(
        self,
        hidden_size: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        log_scale: float = 1.0,
        log_scale_min: float = -1.5,
        log_scale_max: float = 1.5,
        store_output_for_backprop: bool = False,
    ) -> None:
        super(BiasNorm, self).__init__()
        self.hidden_size = hidden_size
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(log_scale))
        self.bias = nn.Parameter(torch.empty(hidden_size).normal_(mean=0, std=1e-4))

        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

        self.store_output_for_backprop = store_output_for_backprop

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.hidden_size

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            channel_dim = self.channel_dim
            if channel_dim < 0:
                channel_dim += x.ndim
            bias = self.bias
            for _ in range(channel_dim + 1, x.ndim):
                bias = bias.unsqueeze(-1)
            scales = (
                torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
            ) * self.log_scale.exp()
            return x * scales

        log_scale = limit_param_value(
            self.log_scale,
            min=float(self.log_scale_min),
            max=float(self.log_scale_max),
            training=self.training,
        )

        return BiasNormFunction.apply(
            x, self.bias, log_scale, self.channel_dim, self.store_output_for_backprop
        )         

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)