import torch
import torch.nn as nn
from network_module import Conv2dLayer, normLayer
from featSegm import *


class FFC_generator(nn.Module):
    def __init__(self,  norm='bn', depth_enc=False, upsample_mode='upsample', resblocks = 9, ngf=32, use_attention = True, predict_transm=True):
        super().__init__()
        self.upsample_mode = upsample_mode
        self.predict_transm = predict_transm
        self.depth_enc = depth_enc
        self.use_attention = use_attention

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=7, padding=0)

        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=3, stride=2, padding=1)
        self.bn2 = normLayer(ngf*2, norm=norm)

        self.conv3 = nn.Conv2d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=3, stride=2, padding=1)
        self.bn3 = normLayer(ngf*4, norm=norm)

        self.conv4 = nn.Conv2d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=3, stride=2, padding=1)
        self.bn4 = normLayer(ngf*8, norm=norm)

        if self.use_attention:
            self.attention = ChannelAttention(ngf*4)
            self.att_conv =nn.Conv2d(ngf*8, ngf*4, 1)

        blocks = []
        ### resnet blocks
        for i in range(9):
            cur_resblock = ResnetBlock_remove_IN(ngf*8, 1, norm=nn.BatchNorm2d)
            blocks.append(cur_resblock)

        self.middle = nn.Sequential(*blocks)

        if self.upsample_mode == 'upsample':
            self.convt1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*8, ngf*4, kernel_size=3, stride=1, padding=1),
            normLayer(ngf*4, norm=norm),
            nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1),
        )
        else:
            self.convt1 = nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=3, stride=2, padding=1)
    
        self.bnt1 = normLayer(ngf*4, norm=norm)

        if self.upsample_mode == 'upsample':
            self.convt2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*4, ngf*2, kernel_size=3, stride=1, padding=1),
            normLayer(ngf*2, norm=norm),
            nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1),
        )
        else:
            self.convt2 = nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1)

        self.bnt2 = normLayer(ngf*2, norm=norm)
        
        if self.upsample_mode == 'upsample':
            self.convt3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=1),
        )
        else:
            self.convt3 = nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1)

        self.bnt3 = normLayer(ngf, norm=norm)

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, padding=0)
        self.act_last = nn.Tanh()

        if self.predict_transm:
            self.dims = [256, 128] #FFC2 #32
            #self.dims = [128, 64] #FFC2 #16
            #self.dims = [256, 128] #FFC2
            #self.dims = [128, 256] #FFC1
            self.layers = [0, 1]
            self.SE = EXTRACTOR_POOL['LSE'](n_class=1, dims=self.dims, layers=self.layers)

    def forward(self, input, rel_pos_emb=None, direct_emb=None, str_feats=None, depth=None):
        b, _, h, w = input.shape
        feats = []
        x = input
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)
        
        if rel_pos_emb is not None:
            try:
                #depth_embed = self.rel_depth_emb(depth.squeeze(0))
                rel_pos_emb = rel_pos_emb.reshape(b, h, w)
                rel_pos_embed = self.rel_depth_emb(rel_pos_emb) * self.alpha
                direct_pos_embed = self.multiLabel_emb(direct_emb.reshape(b, h, w, 4).float()) * self.beta
                x = x + rel_pos_embed.permute(0,3,1,2) + direct_pos_embed.permute(0,3,1,2)
            except:
                print("RuntimeError\n")
        x = self.conv2(x)
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)
        
        x = self.conv3(x)
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        #FFC1: features from layer3+layer4
        # if self.predict_transm:
        #     feats.append(x)

        x = self.conv4(x)
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        if self.predict_transm:
            feats.append(x)
            #FFC 1
            # ls_out = self.SE(feats, size=(input.shape[2], input.shape[3])) 
            # pred_map = ls_out[-1] 
        # FFC block
        x = self.middle(x)

        # Decoder
        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)

        #FFC2
        if self.predict_transm:
            feats.append(x)
            ls_out = self.SE(feats, size=(input.shape[2], input.shape[3]))
            pred_map = ls_out[-1] 

        #v1
        if self.use_attention:
            _x = self.attention(x)
            _pred_map = torch.nn.functional.interpolate(pred_map, size=[x.shape[2], x.shape[3]])
            tr_x = _pred_map * x
            _x = torch.cat((_x, tr_x), 1)
            x = self.att_conv(_x)
        #v2
        # if self.use_attention:
        #     x = self.attention(x)
        #     _pred_map = torch.nn.functional.interpolate(pred_map, size=[x.shape[2], x.shape[3]])
        #     x = _pred_map * x


        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32)) 
        x = self.act(x)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)
        x = self.act_last(x)
        x = (x + 1) / 2
        
        if self.predict_transm:
            return x, pred_map
        else:
            return x
            
class ResnetBlock_remove_IN(nn.Module):
    def __init__(self, dim, dilation=1, norm=nn.InstanceNorm2d):
        super(ResnetBlock_remove_IN, self).__init__()

        self.ffc1 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=dilation, groups=1, bias=False,
                               norm_layer=norm, activation_layer=nn.ReLU, enable_lfu=False)

        self.ffc2 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=1, groups=1, bias=False,
                               norm_layer=norm, activation_layer=nn.ReLU, enable_lfu=False)

    def forward(self, x):
        output = x
        _, c, _, _ = output.shape
        output = torch.split(output, [c - int(c * 0.75), int(c * 0.75)], dim=1)
        x_l, x_g = self.ffc1(output)
        output = self.ffc2((x_l, x_g))
        output = torch.cat(output, dim=1)
        output = x + output

        return output

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spectral_pos_encoding=False, fft_norm='ortho', norm='none'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.fft_norm = fft_norm

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        if norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(out_channels * 2)
        else:
            self.bn = None
            
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        if torch.__version__ > '1.7.1' and '1.7.1' not in torch.__version__:
            x = x.to(torch.float32)
            batch = x.shape[0]

            # (batch, c, h, w/2+1, 2)
            fft_dim = (-2, -1)
            ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
            if self.bn is not None:
                ffted = self.relu(self.bn(ffted.to(torch.float32)))
            else:
                ffted = self.relu(ffted.to(torch.float32))
            ffted = ffted.to(torch.float32)

            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
                0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])

            ifft_shape_slice = x.shape[-2:]
            output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        else:
            batch, c, h, w = x.size()
            r_size = x.size()

            # (batch, c, h, w/2+1, 2)
            ffted = torch.rfft(x, signal_ndim=2, normalized=True)
            # (batch, c, 2, h, w/2+1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
            if self.bn is not None:
                ffted = self.relu(self.bn(ffted))
            else:
                ffted = self.relu(ffted)
            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
                0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

            output = torch.irfft(ffted, signal_ndim=2,
                                 signal_sizes=r_size[2:], normalized=True)

        return output


class ChannelAttention(nn.Module):
    def __init__(self, n_channels) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ch_att = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 2, n_channels, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.gap(x)
        out = self.ch_att(x)
        return x * out 

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, separable_fu=False, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            #nn.BatchNorm2d(out_channels // 2),
            #nn.InstanceNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        fu_class = FourierUnit
        self.fu = fu_class(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = fu_class(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


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

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.InstanceNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 or norm_layer=='none' else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 or norm_layer=='none' else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l.to(torch.float32)))
        x_g = self.act_g(self.bn_g(x_g.to(torch.float32)))
        return x_l, x_g
    

class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)

    
def print_networks(net, verbose=True):
    print('---------- Networks initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        print(net)
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')