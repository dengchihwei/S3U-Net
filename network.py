# -*- coding = utf-8 -*-
# @File Name : network
# @Date : 2023/5/21 12:54
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from loss import calc_local_contrast, vessel_loss


# following parts are components of 2D-UNet
class SingleConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, pad=1, act=nn.ELU()):
        super(SingleConv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=size, padding=pad, padding_mode='reflect', bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = act

    def forward(self, x):
        x = self.conv(x)
        try:
            assert torch.any(torch.isfinite(x))
        except AssertionError:
            print('data:', x.mean(), x.max(), x.min())
            print('w:', self.conv.weight.mean(), self.conv.weight.max(), self.conv.weight.min())
        x = self.norm(x)
        x = self.relu(x)
        return x


class DoubleConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, pad=1, act=nn.ELU()):
        super(DoubleConv2d, self).__init__()
        self.conv1 = SingleConv2d(in_ch, out_ch, size, pad, act)
        self.conv2 = SingleConv2d(out_ch, out_ch, size, pad, act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SingleUpSample2d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, stride=2, pad=1, out_pad=0):
        super(SingleUpSample2d, self).__init__()
        size, stride, pad = (size, size), (stride, stride), (pad, pad)
        self.up_sample = nn.ConvTranspose2d(in_ch, out_ch,
                                            kernel_size=size,
                                            stride=stride,
                                            padding=pad,
                                            output_padding=out_pad,
                                            bias=False)

    def forward(self, size, x):
        x = self.up_sample(x, size)
        return x


class SingleEncoder2d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, pool_size=2, pad=1, apply_pool=False):
        super(SingleEncoder2d, self).__init__()
        self.apply_pool = apply_pool
        if self.apply_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.conv = DoubleConv2d(in_ch, out_ch, size, pad)

    def forward(self, x):
        assert not torch.any(torch.isnan(x))
        if self.apply_pool:
            x = self.pool(x)
        x = self.conv(x)
        return x


class SingleDecoder2d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, stride=2, pad=1, out_pad=0):
        super(SingleDecoder2d, self).__init__()
        self.up_sample = SingleUpSample2d(in_ch-out_ch, in_ch-out_ch, size, stride, pad, out_pad)
        self.conv = DoubleConv2d(in_ch, out_ch, size, pad)

    def forward(self, encoder_features, x):
        assert not torch.any(torch.isnan(x))
        x = self.up_sample(encoder_features.size()[2:], x)
        x = torch.cat((encoder_features, x), dim=1)
        x = self.conv(x)
        return x


class SpatialAttention2d(nn.Module):
    def __init__(self, out_ch, min_scale, max_scale, size=3, pad=1, sample_num=16, sample_layers=4):
        super(SpatialAttention2d, self).__init__()
        self.min_scale, self.max_scale = min_scale, max_scale
        self.sample_num = sample_num
        self.sample_layer = sample_layers
        self.radius_conv = SingleConv2d(out_ch, sample_num, size, pad, act=nn.Sigmoid())
        self.attention_conv = SingleConv2d(4, 1, size, pad, act=nn.Tanh())

    def forward(self, image, x, radius=None):
        assert not torch.any(torch.isnan(x))
        if radius is None:
            radius = self.radius_conv(x) * self.max_scale + self.min_scale
        contrast = calc_local_contrast(image, radius, self.sample_num, self.sample_layer)
        assert not torch.any(torch.isnan(contrast))
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        min_x, _ = torch.min(x, dim=1, keepdim=True)
        attention = torch.cat([avg_x, max_x, min_x, contrast], dim=1)
        attention = self.attention_conv(attention)
        return attention


class OutConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, size=3, pad=1, act=nn.ELU()):
        super(OutConv, self).__init__()
        self.conv1 = SingleConv2d(in_ch, mid_ch, size, pad, act)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# following parts are components of the 3D-UNet
class SingleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, pad=1, act=nn.ELU()):
        super(SingleConv3d, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=size, padding=pad, padding_mode='reflect', bias=False)
        self.norm = nn.BatchNorm3d(out_ch)
        self.relu = act

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, pad=1, act=nn.ELU()):
        super(DoubleConv3d, self).__init__()
        self.conv1 = SingleConv3d(in_ch, out_ch, size, pad, act)
        self.conv2 = SingleConv3d(out_ch, out_ch, size, pad, act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNetConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, pad=1):
        super(ResNetConv3d, self).__init__()
        self.residual = nn.Conv3d(in_ch, out_ch, kernel_size=size, padding=pad, bias=False)
        self.norm = nn.BatchNorm3d(out_ch)
        self.conv1 = SingleConv3d(in_ch, out_ch, size, pad)
        self.conv2 = SingleConv3d(in_ch, out_ch, size, pad)

    def forward(self, x):
        x = self.norm(self.residual(x))
        res = x
        x = self.conv1(x)
        x = self.conv2(x + res)
        return x


class SingleUpSample3d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, stride=2, pad=1, out_pad=1):
        super(SingleUpSample3d, self).__init__()
        size, stride, pad = (size, size, size), (stride, stride, stride), (pad, pad, pad)
        self.up_sample = nn.ConvTranspose3d(in_ch, out_ch,
                                            kernel_size=size,
                                            stride=stride,
                                            padding=pad,
                                            output_padding=out_pad,
                                            bias=False)

    def forward(self, size, x):
        # output_size = encoder_features.size()[2:]
        x = self.up_sample(x, size)
        return x


class SingleEncoder3d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, pool_size=2, pad=1, apply_pool=False):
        super(SingleEncoder3d, self).__init__()
        self.apply_pool = apply_pool
        if self.apply_pool:
            self.pool = nn.MaxPool3d(kernel_size=pool_size)
        self.conv = DoubleConv3d(in_ch, out_ch, size, pad)

    def forward(self, x):
        assert not torch.any(torch.isnan(x))
        if self.apply_pool:
            x = self.pool(x)
        x = self.conv(x)
        return x


class SingleDecoder3d(nn.Module):
    def __init__(self, in_ch, out_ch, size=3, stride=2, pad=1, out_pad=0):
        super(SingleDecoder3d, self).__init__()
        self.up_sample = SingleUpSample3d(in_ch-out_ch, in_ch-out_ch, size, stride, pad, out_pad)
        self.conv = DoubleConv3d(in_ch, out_ch, size)

    def forward(self, encoder_features, x):
        assert not torch.any(torch.isnan(x))
        x = self.up_sample(encoder_features.size()[2:], x)
        x = torch.cat((encoder_features, x), dim=1)
        x = self.conv(x)
        return x


class SpatialAttention3d(nn.Module):
    def __init__(self, out_ch, min_scale, max_scale, size=3, pad=1, sample_num=16, sample_layers=3):
        super(SpatialAttention3d, self).__init__()
        self.min_scale, self.max_scale = min_scale, max_scale
        self.sample_num = sample_num
        self.sample_layer = sample_layers
        self.radius_conv = SingleConv3d(out_ch, 1, size, pad, act=nn.Sigmoid())
        self.attention_conv = SingleConv3d(3, 1, size, pad, act=nn.Sigmoid())

    def forward(self, image, x, radius=None):
        assert not torch.any(torch.isnan(x))
        if radius is None:
            radius = self.radius_conv(x) * self.max_scale + self.min_scale
        contrast = calc_local_contrast(image, radius, self.sample_num, self.sample_layer)
        assert not torch.any(torch.isnan(contrast))
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_x, max_x, contrast], dim=1)
        attention = self.attention_conv(attention)
        return attention


class Initializer(object):
    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            # m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)


class UNet2D(nn.Module):
    def __init__(self, in_ch, out_ch, feat_dims, size=3, pool_size=2, stride=2, pad=1, out_pad=0):
        super(UNet2D, self).__init__()
        # encoder part
        self.encoder1 = SingleEncoder2d(in_ch, feat_dims[0], size, pool_size, pad, apply_pool=False)
        self.encoder2 = SingleEncoder2d(feat_dims[0], feat_dims[1], size, pool_size, pad, apply_pool=True)
        self.encoder3 = SingleEncoder2d(feat_dims[1], feat_dims[2], size, pool_size, pad, apply_pool=True)
        self.encoder4 = SingleEncoder2d(feat_dims[2], feat_dims[3], size, pool_size, pad, apply_pool=True)
        # decoder part
        self.decoder1 = SingleDecoder2d(feat_dims[3] + feat_dims[2], feat_dims[2], size, stride, pad, out_pad)
        self.decoder2 = SingleDecoder2d(feat_dims[2] + feat_dims[1], feat_dims[1], size, stride, pad, out_pad)
        self.decoder3 = SingleDecoder2d(feat_dims[1] + feat_dims[0], feat_dims[0], size, stride, pad, out_pad)
        # final conv
        self.pred = nn.Conv2d(feat_dims[0], out_ch, 1)
        Initializer.weights_init(self)

    def freeze_unet(self):
        self.freeze_layer(self.encoder1)
        self.freeze_layer(self.encoder2)
        self.freeze_layer(self.encoder3)
        self.freeze_layer(self.encoder4)
        self.freeze_layer(self.decoder1)
        self.freeze_layer(self.decoder2)
        self.freeze_layer(self.decoder3)

    @staticmethod
    def freeze_layer(layer):
        for param in layer.parameters():
            param.requires_grad = False

    def forward(self, im, gt=None):
        x1 = self.encoder1(im)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        x = self.decoder1(x3, x4)
        x = self.decoder2(x2, x)
        x = self.decoder3(x1, x)

        # prediction
        pred = self.pred(x)

        # calculate the losses
        losses = {'total_loss': F.binary_cross_entropy_with_logits(pred, gt).unsqueeze(0)} if gt is not None else None
        return pred, losses


class LocalContrastNet2D(nn.Module):
    def __init__(self, in_ch, out_ch, min_scale, max_scale, radius_num, feat_dims,
                 spt_attn=True, size=3, pool_size=2, stride=2, pad=1, out_pad=0):
        super(LocalContrastNet2D, self).__init__()
        self.min_scale, self.max_scale, self.spt_attn = min_scale, max_scale, spt_attn
        # encoder part
        self.encoder1 = SingleEncoder2d(in_ch, feat_dims[0], size, pool_size, pad, apply_pool=False)
        self.encoder2 = SingleEncoder2d(feat_dims[0], feat_dims[1], size, pool_size, pad, apply_pool=True)
        self.encoder3 = SingleEncoder2d(feat_dims[1], feat_dims[2], size, pool_size, pad, apply_pool=True)
        self.encoder4 = SingleEncoder2d(feat_dims[2], feat_dims[3], size, pool_size, pad, apply_pool=True)
        # decoder part
        self.decoder1 = SingleDecoder2d(feat_dims[3] + feat_dims[2], feat_dims[2], size, stride, pad, out_pad)
        self.decoder2 = SingleDecoder2d(feat_dims[2] + feat_dims[1], feat_dims[1], size, stride, pad, out_pad)
        self.decoder3 = SingleDecoder2d(feat_dims[1] + feat_dims[0], feat_dims[0], size, stride, pad, out_pad)
        # spatial attention part
        if self.spt_attn:
            self.level_attention1 = SpatialAttention2d(feat_dims[0], min_scale, max_scale, size, pad)
            self.level_attention2 = SpatialAttention2d(feat_dims[1], min_scale, max_scale, size, pad)
            self.level_attention3 = SpatialAttention2d(feat_dims[2], min_scale, max_scale, size, pad)
            self.global_attention = SpatialAttention2d(feat_dims[0], min_scale, max_scale, size, pad, radius_num, 5)
        # final conv
        self.recon_conv = nn.Conv2d(feat_dims[0], in_ch, 1)
        self.direction_conv = nn.Conv2d(feat_dims[0], out_ch, 1)
        self.radius_conv = nn.Conv2d(feat_dims[0], radius_num, 1)
        Initializer.weights_init(self)

    def forward(self, im, loss_config=None):
        # encoding
        x1 = self.encoder1(im)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # apply attention
        attentions = []
        if self.spt_attn:
            attention1 = self.level_attention1(im, x1)
            x1 = torch.mul(attention1, x1)
            attentions.append(attention1)

            x = F.interpolate(im, scale_factor=0.5, mode='bilinear')
            attention2 = self.level_attention2(x, x2)
            x2 = torch.mul(attention2, x2)
            attentions.append(attention2)

            x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
            attention3 = self.level_attention3(x, x3)
            x3 = torch.mul(attention3, x3)
            attentions.append(attention3)

        # decoding
        x = self.decoder1(x3, x4)
        x = self.decoder2(x2, x)
        x = self.decoder3(x1, x)

        # final conv
        recon = self.recon_conv(x)
        direction = self.direction_conv(x)
        radius = torch.sigmoid(self.radius_conv(x)) * self.max_scale + self.min_scale

        # output dicts
        output = {
            'recon': recon,
            'radius': radius,
            'vessel': direction
        }

        # global attention
        if self.spt_attn:
            global_attn = self.global_attention(im, x, radius).squeeze(1)
            attentions.append(global_attn)
            output['attentions'] = attentions

        # calculate the losses
        losses = vessel_loss(im, output, loss_config) if loss_config is not None else None
        return output, losses


class UNet3D(nn.Module):
    def __init__(self, in_ch, out_ch,  feat_dims, size=3, pool_size=2, stride=2, pad=1, out_pad=0):
        super(UNet3D, self).__init__()
        # encoder part
        self.encoder1 = SingleEncoder3d(in_ch, feat_dims[0], size, pool_size, pad, apply_pool=False)
        self.encoder2 = SingleEncoder3d(feat_dims[0], feat_dims[1], size, pool_size, pad, apply_pool=True)
        self.encoder3 = SingleEncoder3d(feat_dims[1], feat_dims[2], size, pool_size, pad, apply_pool=True)
        self.encoder4 = SingleEncoder3d(feat_dims[2], feat_dims[3], size, pool_size, pad, apply_pool=True)
        # decoder part
        self.decoder1 = SingleDecoder3d(feat_dims[3] + feat_dims[2], feat_dims[2], size, stride, pad, out_pad)
        self.decoder2 = SingleDecoder3d(feat_dims[2] + feat_dims[1], feat_dims[1], size, stride, pad, out_pad)
        self.decoder3 = SingleDecoder3d(feat_dims[1] + feat_dims[0], feat_dims[0], size, stride, pad, out_pad)
        # final conv
        self.pred = nn.Conv3d(feat_dims[0], out_ch, 1)
        Initializer.weights_init(self)

    def freeze_unet(self):
        self.freeze_layer(self.encoder1)
        self.freeze_layer(self.encoder2)
        self.freeze_layer(self.encoder3)
        self.freeze_layer(self.encoder4)
        self.freeze_layer(self.decoder1)
        self.freeze_layer(self.decoder2)
        self.freeze_layer(self.decoder3)

    @staticmethod
    def freeze_layer(layer):
        for param in layer.parameters():
            param.requires_grad = False

    def forward(self, im, gt=None):
        x1 = self.encoder1(im)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        x = self.decoder1(x3, x4)
        x = self.decoder2(x2, x)
        x = self.decoder3(x1, x)

        # prediction
        pred = self.pred(x)

        # calculate the losses
        # losses = {'total_loss': F.binary_cross_entropy_with_logits(pred, gt).unsqueeze(0)} if gt is not None else None
        # return pred, losses
        return pred


class LocalContrastNet3D(nn.Module):
    def __init__(self, in_ch, out_ch, min_scale, max_scale, radius_num, feat_dims,
                 spt_attn=True, size=3, pool_size=2, stride=2, pad=1, out_pad=0):
        super(LocalContrastNet3D, self).__init__()
        self.min_scale, self.max_scale, self.spt_attn = min_scale, max_scale, spt_attn
        # encoder part
        self.encoder1 = SingleEncoder3d(in_ch, feat_dims[0], size, pool_size, pad, apply_pool=False)
        self.encoder2 = SingleEncoder3d(feat_dims[0], feat_dims[1], size, pool_size, pad, apply_pool=True)
        self.encoder3 = SingleEncoder3d(feat_dims[1], feat_dims[2], size, pool_size, pad, apply_pool=True)
        self.encoder4 = SingleEncoder3d(feat_dims[2], feat_dims[3], size, pool_size, pad, apply_pool=True)
        # decoder part
        self.decoder1 = SingleDecoder3d(feat_dims[3] + feat_dims[2], feat_dims[2], size, stride, pad, out_pad)
        self.decoder2 = SingleDecoder3d(feat_dims[2] + feat_dims[1], feat_dims[1], size, stride, pad, out_pad)
        self.decoder3 = SingleDecoder3d(feat_dims[1] + feat_dims[0], feat_dims[0], size, stride, pad, out_pad)
        # spatial attention part
        if self.spt_attn:
            self.level_attention1 = SpatialAttention3d(feat_dims[0], min_scale, max_scale, size, pad)
            self.level_attention2 = SpatialAttention3d(feat_dims[1], min_scale, max_scale, size, pad)
            self.level_attention3 = SpatialAttention3d(feat_dims[2], min_scale, max_scale, size, pad)
            self.global_attention = SpatialAttention3d(feat_dims[0], min_scale, max_scale, size, pad, radius_num, 3)
        # final conv
        self.recon_conv = nn.Conv3d(feat_dims[0], in_ch, 1)
        self.direction_conv = nn.Conv3d(feat_dims[0], out_ch, 1)
        self.radius_conv = nn.Conv3d(feat_dims[0], radius_num, 1)
        Initializer.weights_init(self)

    def forward(self, im, loss_config=None):
        # encoding
        x1 = self.encoder1(im)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        features = [x1, x2, x3, x4]

        # apply attention
        attentions = []
        if self.spt_attn:
            attention1 = self.level_attention1(im, x1)
            x1 = torch.mul(attention1, x1)
            attentions.append(attention1)

            x = F.interpolate(im, scale_factor=0.5, mode='trilinear')
            attention2 = self.level_attention2(x, x2)
            x2 = torch.mul(attention2, x2)
            attentions.append(attention2)

            x = F.interpolate(x, scale_factor=0.5, mode='trilinear')
            attention3 = self.level_attention3(x, x3)
            x3 = torch.mul(attention3, x3)
            attentions.append(attention3)

        # decoding
        x = self.decoder1(x3, x4)
        features.append(x.clone())
        x = self.decoder2(x2, x)
        features.append(x.clone())
        x = self.decoder3(x1, x)
        features.append(x.clone())

        # final conv
        recon = self.recon_conv(x)
        direction = self.direction_conv(x)
        radius = torch.sigmoid(self.radius_conv(x)) * self.max_scale + self.min_scale

        # output dicts
        output = {
            'feat': self.process_feats(features),
            'recon': recon,
            'radius': radius,
            'vessel': direction
        }

        # global attention
        if self.spt_attn:
            global_attn = self.global_attention(im, x, radius).squeeze(1)
            attentions.append(global_attn)
            output['attentions'] = attentions
        losses = vessel_loss(im, output, loss_config) if loss_config is not None else None
        return output, losses

    @staticmethod
    def process_feats(layer_features):
        """
        process the feature lists output by the network
        :param layer_features: a list of U-Net features of different layers
        :return: final features [b, c1+c2+c3+c4+c5+c6+c7, h, w, d]
        """
        x1, x2, x3, x4, x5, x6, x7 = layer_features
        x2 = F.interpolate(x2, scale_factor=2, mode='trilinear')
        x3 = F.interpolate(x3, scale_factor=4, mode='trilinear')
        x4 = F.interpolate(x4, scale_factor=8, mode='trilinear')
        x5 = F.interpolate(x5, scale_factor=4, mode='trilinear')
        x6 = F.interpolate(x6, scale_factor=2, mode='trilinear')
        final_feats = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=1)
        return final_feats


if __name__ == '__main__':
    from torchinfo import summary

    feature_dims = [64, 128, 256, 512]
    _loss_config = {
        "lambda_flux": 5.0,
        "lambda_recon": 1.0,
        "lambda_direction": 1.0,
        "lambda_intensity": 1.0,
        "flux_sample_num": 128,
        "sample_num": 16
    }

    """ 3D UNet Test """
    unet_3d = UNet3D(1, 1, feature_dims)
    print(type(unet_3d).__name__)
    summary(unet_3d, input_size=[(8, 1, 16, 16, 16), (8, 1, 16, 16, 16)])

    contrast_model_3d = LocalContrastNet3D(1, 3, 0.1, 10.5, 128, feature_dims)
    print(type(contrast_model_3d).__name__)
    summary(contrast_model_3d, input_data=[torch.randn((8, 1, 16, 16, 16)), _loss_config])

    """ 2D UNet Test """
    unet_2d = UNet2D(1, 1, feature_dims)
    print(type(unet_2d).__name__)
    summary(unet_2d, input_size=[(8, 1, 16, 16), (8, 1, 16, 16)])

    contrast_model_2d = LocalContrastNet2D(1, 2, 0.1, 10.5, 128, feature_dims)
    print(type(contrast_model_2d).__name__)
    summary(contrast_model_2d, input_data=[torch.randn((8, 1, 16, 16)), _loss_config])
