import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_tf_

from lib.wmsr_block import WAM


def _init_weights(module, name, scheme=''):
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()

    if act == 'relu':
        return nn.ReLU(inplace)
    if act == 'relu6':
        return nn.ReLU6(inplace)
    if act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    if act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    if act == 'gelu':
        return nn.GELU()
    if act == 'hswish':
        return nn.Hardswish(inplace)

    raise NotImplementedError(f'Activation layer [{act}] is not found.')


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class MiniDRP_MultiChannel(nn.Module):
    def __init__(self, in_channels, num_classes=9):
        super().__init__()

        hidden_channels = max(in_channels // 16, 1)

        self.conv_s = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

        self.freq_weight = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.mask_projector = nn.Sequential(
            nn.Conv2d(num_classes, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x, prior_cam):
        if prior_cam.shape[2:] != x.shape[2:]:
            prior_cam = F.interpolate(
                prior_cam,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        mask_features = self.mask_projector(prior_cam)

        x_s = self.conv_s(x)

        x_fft = torch.fft.fft2(x.float())
        weight = self.freq_weight(x_fft.real)
        x_f = torch.abs(torch.fft.ifft2(weight * x_fft))
        x_f = self.relu(self.norm(x_f))

        x_enhanced = x_s + x_f

        r_prior_cam_s = 1 - torch.sigmoid(mask_features)

        mask_fft = torch.fft.fft2(mask_features.float())
        r_prior_cam_f = torch.abs(mask_fft)
        r_prior_cam_f = 1 - torch.sigmoid(r_prior_cam_f)

        r_prior_cam = r_prior_cam_s + r_prior_cam_f
        out = x_enhanced * r_prior_cam

        return out


class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super().__init__()

        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=in_channels,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True)
            )
            for kernel_size in kernel_sizes
        ])

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = []

        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)

            if not self.dw_parallel:
                x = x + dw_out

        return outputs


class MSCB_With_DRP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        kernel_sizes=(1, 3, 5),
        expansion_factor=2,
        dw_parallel=True,
        add=True,
        activation='relu6',
        num_classes=9
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.add = add
        self.use_skip_connection = self.stride == 1

        ex_channels = int(self.in_channels * expansion_factor)
        if ex_channels % 3 != 0:
            ex_channels += 3 - ex_channels % 3

        self.ex_channels = ex_channels
        self.split_channels = self.ex_channels // 3

        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ex_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(activation, inplace=True)
        )

        self.msdc = MSDC(
            self.split_channels,
            kernel_sizes,
            stride,
            activation,
            dw_parallel=dw_parallel
        )

        self.wam = WAM(hidden_dim=self.split_channels)
        self.drp = MiniDRP_MultiChannel(self.split_channels, num_classes=num_classes)

        if self.add:
            combined_channels_in = self.split_channels * 3
        else:
            combined_channels_in = self.split_channels * len(kernel_sizes) + self.split_channels * 2

        self.pconv2 = nn.Sequential(
            nn.Conv2d(combined_channels_in, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )

        if self.use_skip_connection and self.in_channels != self.out_channels:
            self.conv1x1 = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                bias=False
            )

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x, prior_mask=None):
        identity = x

        x_ex = self.pconv1(x)
        x1, x2, x3 = torch.chunk(x_ex, chunks=3, dim=1)

        msdc_outs = self.msdc(x1)

        if self.add:
            out_branch1 = sum(msdc_outs)
        else:
            out_branch1 = torch.cat(msdc_outs, dim=1)

        out_branch2 = self.wam(x2)

        if prior_mask is not None:
            out_branch3 = self.drp(x3, prior_mask)
        else:
            out_branch3 = x3

        out = torch.cat([out_branch1, out_branch2, out_branch3], dim=1)
        out = channel_shuffle(out, groups=3)
        out = self.pconv2(out)

        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                identity = self.conv1x1(identity)
            out = identity + out

        return out


def MSCBLayer(
    in_channels,
    out_channels,
    n=1,
    stride=1,
    kernel_sizes=(1, 3, 5),
    expansion_factor=2,
    dw_parallel=True,
    add=True,
    activation='relu6',
    num_classes=9
):
    layers = [
        MSCB_With_DRP(
            in_channels,
            out_channels,
            stride,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            activation=activation,
            num_classes=num_classes
        )
    ]

    for _ in range(1, n):
        layers.append(
            MSCB_With_DRP(
                out_channels,
                out_channels,
                stride=1,
                kernel_sizes=kernel_sizes,
                expansion_factor=expansion_factor,
                dw_parallel=dw_parallel,
                add=add,
                activation=activation,
                num_classes=num_classes
            )
        )

    return nn.Sequential(*layers)


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super().__init__()

        self.in_channels = in_channels

        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            act_layer(activation, inplace=True)
        )

        self.pwc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)

        return x


class Sbam(nn.Module):
    def __init__(self, in_channel, out_channel, activation='relu'):
        super().__init__()

        self.hl_up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.concat_layer = nn.Sequential(
            nn.Conv2d(in_channel + out_channel, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel),
            act_layer(activation, inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channel),
            act_layer(activation, inplace=True)
        )

        self.hl_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            act_layer(activation, inplace=True)
        )

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, hl, ll):
        if hl.shape[2:] != ll.shape[2:]:
            hl = self.hl_up(hl)

        k = self.concat_layer(torch.cat([hl, ll], dim=1))
        hl = self.hl_layer(hl + k)
        out = ll + hl

        return out


class EMCAD(nn.Module):
    def __init__(
        self,
        channels=(512, 320, 128, 64),
        kernel_sizes=(1, 3, 5),
        expansion_factor=6,
        dw_parallel=True,
        add=True,
        activation='relu6',
        num_classes=1,
        **kwargs
    ):
        super().__init__()

        eucb_ks = 3

        self.mscb4 = MSCBLayer(
            channels[0],
            channels[0],
            n=1,
            stride=1,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            activation=activation,
            num_classes=num_classes
        )

        self.eucb3 = EUCB(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=eucb_ks,
            stride=eucb_ks // 2
        )

        self.sbam3 = Sbam(
            in_channel=channels[1],
            out_channel=channels[1],
            activation=activation
        )

        self.mscb3 = MSCBLayer(
            channels[1],
            channels[1],
            n=1,
            stride=1,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            activation=activation,
            num_classes=num_classes
        )

        self.eucb2 = EUCB(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=eucb_ks,
            stride=eucb_ks // 2
        )

        self.sbam2 = Sbam(
            in_channel=channels[2],
            out_channel=channels[2],
            activation=activation
        )

        self.mscb2 = MSCBLayer(
            channels[2],
            channels[2],
            n=1,
            stride=1,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            activation=activation,
            num_classes=num_classes
        )

        self.eucb1 = EUCB(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=eucb_ks,
            stride=eucb_ks // 2
        )

        self.sbam1 = Sbam(
            in_channel=channels[3],
            out_channel=channels[3],
            activation=activation
        )

        self.mscb1 = MSCBLayer(
            channels[3],
            channels[3],
            n=1,
            stride=1,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            activation=activation,
            num_classes=num_classes
        )

        self.head4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(channels[1], num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(channels[2], num_classes, kernel_size=1)
        self.head1 = nn.Conv2d(channels[3], num_classes, kernel_size=1)

    def forward(self, x, skips):
        d4 = x

        for module in self.mscb4:
            d4 = module(d4, prior_mask=None)

        p4 = self.head4(d4)
        mask4 = torch.sigmoid(p4)

        d3 = self.eucb3(d4)
        d3 = self.sbam3(hl=d3, ll=skips[0])

        for module in self.mscb3:
            d3 = module(d3, prior_mask=mask4)

        p3 = self.head3(d3)
        mask3 = torch.sigmoid(p3)

        d2 = self.eucb2(d3)
        d2 = self.sbam2(hl=d2, ll=skips[1])

        for module in self.mscb2:
            d2 = module(d2, prior_mask=mask3)

        p2 = self.head2(d2)
        mask2 = torch.sigmoid(p2)

        d1 = self.eucb1(d2)
        d1 = self.sbam1(hl=d1, ll=skips[2])

        for module in self.mscb1:
            d1 = module(d1, prior_mask=mask2)

        p1 = self.head1(d1)

        return [p4, p3, p2, p1]