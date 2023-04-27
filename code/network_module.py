import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
import functools
from typing import Iterable, Tuple, Union
from torch import Tensor, nn
import torch.fft as fft
from torch.fft import irfftn, rfftn
import torch.nn.functional as f
from functools import partial
#-----------------------------------------------
#                Normal ConvBlock
#-----------------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'replicate', activation = 'none', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'spherical':
            self.pad = SphericalPad(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0,inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False, scale_factor = 2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.conv2d(x)
        return x

#https://arxiv.org/pdf/2005.09704.pdf
class depth_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super(depth_separable_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
        
#https://arxiv.org/pdf/2005.09704.pdf
class sc_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super(sc_conv, self).__init__()
        self.single_channel_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1
        )

    def forward(self, input):
        out = self.single_channel_conv(input)
        return out


#-----------------------------------------------
#                Gated ConvBlock
#-----------------------------------------------
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'replicate', activation = 'elu', norm = 'none', sc=False, sn=False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'spherical':
            self.pad = SphericalPad(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sc:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            self.mask_conv2d = sc_conv(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            #self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = depth_separable_conv(in_channels, out_channels, kernel_size, stride, padding = 0,dilation=dilation)

        self.sigmoid = torch.nn.Sigmoid()

    
    def forward(self, x_in):
        x = self.pad(x_in)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        if self.norm:
            conv = self.norm(conv)
        if self.activation:
            conv = self.activation(conv)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask

        #x += self.shortcut(x_in)


        return x

class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sc = False, scale_factor = 2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sc)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x

def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)


def normLayer(out_channels, norm='instance'):
    if norm == 'bn':
        norm = nn.BatchNorm2d(out_channels)
    elif norm == 'instance':
        norm = nn.InstanceNorm2d(out_channels)
    elif norm == 'none':
        norm = None
    else:
        assert 0, "Unsupported normalization: {}".format(norm)
    return norm

def __pad_circular_nd(x: torch.Tensor, pad: int, dim) -> torch.Tensor:
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :param dim: the dimension over which the tensors are padded
    :return:
    """
    if isinstance(dim, int):
        dim = [dim]
    for d in dim:
        if d >= len(x.shape):
            raise IndexError(f"dim {d} out of range")
        idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x, x[idx]], dim=d)
        idx = tuple(slice(None if s != d else -2 * pad, None if s != d else -pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x[idx], x], dim=d)
        pass
    return x

horizontal_circular_pad2d = functools.partial(__pad_circular_nd, dim=[3])

class SphericalPad(torch.nn.Module):
    def __init__(self,
        padding
    ):
        super(SphericalPad, self).__init__()
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: # assumes [B, C, H, W] tensor inputs
        return torch.nn.functional.pad(
            horizontal_circular_pad2d(
                x, pad=self.padding
            ),
            pad=[0, 0, self.padding, self.padding], mode='replicate'
        )

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
    
    
##############################################################################
#######################################Fourier convs##########################
##############################################################################

def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    scalar_matmul = partial(torch.einsum, "agc..., gbc... -> agb...")
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    # Compute the real and imaginary parts independently, then manually insert them
    # into the output Tensor.  This is fairly hacky but necessary for PyTorch 1.7.0,
    # because Autograd is not enabled for complex matrix operations yet.  Not exactly
    # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.
    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple
    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    padding: Union[int, Iterable[int]] = 0,
    stride: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).
    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int]) Number of zero samples to pad the
            input on the last dimension.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.
    Returns:
        (Tensor) Convolved tensor
    """
    # Cast padding & stride to tuples.
    padding_ = to_ntuple(padding, n=signal.ndim - 2)
    stride_ = to_ntuple(stride, n=signal.ndim - 2)

    # Pad the input signal & kernel tensors
    signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    signal = f.pad(signal, signal_padding)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    if signal.size(-1) % 2 != 0:
        signal_ = f.pad(signal, [0, 1])
    else:
        signal_ = signal

    kernel_padding = [
        pad
        for i in reversed(range(2, signal_.ndim))
        for pad in [0, signal_.size(i) - kernel.size(i)]
    ]
    padded_kernel = f.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    # signal_ = signal_.reshape(signal_.size(0), groups, -1, *signal_.shape[2:])
    signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(padded_kernel, dim=tuple(range(2, signal.ndim)))

    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # Remove extra padded values
    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
        slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output


class _FFTConv(nn.Module):
    """Base class for PyTorch FFT convolution layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Iterable[int]],
        padding: Union[int, Iterable[int]] = 0,
        stride: Union[int, Iterable[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        ndim: int = 1,
    ):
        """
        Args:
            in_channels: (int) Number of channels in input tensors
            out_channels: (int) Number of channels in output tensors
            kernel_size: (Union[int, Iterable[int]) Square radius of the kernel
            padding: (Union[int, Iterable[int]) Number of zero samples to pad the
                input on the last dimension.
            stride: (Union[int, Iterable[int]) Stride size for computing output values.
            bias: (bool) If True, includes bias, which is added after convolution
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.use_bias = bias

        if in_channels % groups != 0:
            raise ValueError(
                "'in_channels' must be divisible by 'groups'."
                f"Found: in_channels={in_channels}, groups={groups}."
            )
        if out_channels % groups != 0:
            raise ValueError(
                "'out_channels' must be divisible by 'groups'."
                f"Found: out_channels={out_channels}, groups={groups}."
            )

        kernel_size = to_ntuple(kernel_size, ndim)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, *kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, signal):
        return fft_conv(
            signal,
            self.weight,
            bias=self.bias,
            padding=self.padding,
            stride=self.stride,
            groups=self.groups,
        )

class FFC(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin=0, ratio_gout=0,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l =  lact #self.act_l = lact(inplace=True)
        self.act_g = gact #self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g