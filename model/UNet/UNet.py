import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)

    
    
class DeepLabV3(nn.Module):
    def __init__(self, backbone, dilations, nclass):
        super(DeepLabV3, self).__init__()

        if 'resnet' in backbone:
            self.backbone = resnet.__dict__[backbone](pretrained=True, replace_stride_with_dilation= [False, False, True])
        else:
            assert backbone == 'xception'
            self.backbone = xception(pretrained=True)

        low_channels = 256
        high_channels = 2048

        self.classifier = nn.Sequential(
            ASPPModule(high_channels, dilations),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, nclass, 1)
        )


    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.backbone.base_forward(x)
        c4 = feats[-1]
        out = self.classifier(c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out


class FCN(nn.Module):
    def __init__(self, backbone, nclass):
        super(FCN, self).__init__()

        if 'resnet' in backbone:
            self.backbone = resnet.__dict__[backbone](pretrained=True, replace_stride_with_dilation=[False, False, True])
        else:
            assert backbone == 'xception'
            self.backbone = xception(pretrained=True)

        high_channels = 2048

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(1024)

        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)

        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.classifier = nn.Conv2d(64, nclass, kernel_size=1)


    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.backbone.base_forward(x)
        c1, c2, c3, c4 = feats[0], feats[1], feats[2], feats[3]
        
        x = self.relu(self.bn1(self.deconv1(c4)))
        x = x + c3
        
        x = self.relu(self.bn2(self.deconv2(x)))
        x = x + c2
        
        x = self.relu(self.bn3(self.deconv3(x)))
        x = x + c1
        
        x = self.relu(self.bn4(self.deconv4(x)))
        
        x = self.classifier(x)
        out = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return out

class UNet(nn.Module):
    def __init__(self, backbone, nclass):
        super(UNet, self).__init__()

        if 'resnet' in backbone:
            self.backbone = resnet.__dict__[backbone](pretrained=True, replace_stride_with_dilation=[False, False, True])
        else:
            assert backbone == 'xception'
            self.backbone = xception(pretrained=True)

        # Feature map channels from encoder
        ch_c1, ch_c2, ch_c3, ch_c4 = 256, 512, 1024, 2048

        # Decoder
        self.up3 = self._upsample_block(ch_c4 + ch_c3, 512)   # c4 + c3
        self.up2 = self._upsample_block(512 + ch_c2, 256)     # up3 + c2
        self.up1 = self._upsample_block(256 + ch_c1, 128)     # up2 + c1
        self.up0 = self._upsample_block(128, 64)              # last up

        self.classifier = nn.Conv2d(64, nclass, kernel_size=1)

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[-2:]

        c1, c2, c3, c4 = self.backbone.base_forward(x)

        u3 = self.up3(torch.cat([F.interpolate(c4, size=c3.shape[-2:], mode='bilinear', align_corners=True), c3], dim=1))
        u2 = self.up2(torch.cat([F.interpolate(u3, size=c2.shape[-2:], mode='bilinear', align_corners=True), c2], dim=1))
        u1 = self.up1(torch.cat([F.interpolate(u2, size=c1.shape[-2:], mode='bilinear', align_corners=True), c1], dim=1))
        u0 = self.up0(F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=True))

        out = self.classifier(u0)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        return out

class PSPModule(nn.Module):
    """
    Pyramid Pooling Module
    - sizes: AdaptiveAvgPool2d의 출력 크기 목록 (예: (1, 2, 3, 6))
    - out_features: bottleneck 1x1 conv의 출력 채널 수
    """
    def __init__(self, in_features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([self._make_stage(in_features, s) for s in sizes])
        self.bottleneck = nn.Conv2d(in_features * (len(sizes) + 1), out_features, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, in_features, size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(in_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        priors = [F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=True)
                  for stage in self.stages]
        priors.append(x)
        y = torch.cat(priors, dim=1)
        y = self.relu(self.bn(self.bottleneck(y)))
        return y


class PSPUpsample(nn.Module):
    """
    2x 업샘플 + 3x3 conv 블록
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.conv(x)


class PSPNet(nn.Module):
    """
    너의 DeepLab/FCN/UNet과 같은 조건:
      - backbone: resnet* 또는 xception (self.backbone.base_forward 사용)
      - 출력: raw logits (LogSoftmax/aux 없음)
      - 최종 업샘플: bilinear + align_corners=True 로 입력 해상도 복원
    """
    def __init__(self, backbone, nclass, psp_sizes=(1, 2, 3, 6)):
        super().__init__()

        if 'resnet' in backbone:
            # 마지막 stage에 dilation 적용해 output stride ~8
            self.backbone = resnet.__dict__[backbone](pretrained=True,
                                                      replace_stride_with_dilation=[False, False, True])
        else:
            assert backbone == 'xception'
            self.backbone = xception(pretrained=True)

        high_channels = 2048  # c4 채널

        # PSP 모듈: c4 -> 1024
        self.psp = PSPModule(in_features=high_channels, out_features=1024, sizes=psp_sizes)
        self.drop1 = nn.Dropout2d(p=0.3)

        # 디코더 업블록: 1024 -> 256 -> 64 -> 64 (입력 해상도까지 2x씩 3번 업샘플)
        self.up1 = PSPUpsample(1024, 256)
        self.drop2 = nn.Dropout2d(p=0.15)

        self.up2 = PSPUpsample(256, 64)
        self.drop3 = nn.Dropout2d(p=0.15)

        self.up3 = PSPUpsample(64, 64)
        self.drop4 = nn.Dropout2d(p=0.15)

        # 최종 분류 헤드 (raw logits)
        self.classifier = nn.Conv2d(64, nclass, kernel_size=1, bias=True)

    def forward(self, x):
        h, w = x.shape[-2:]
        # encoder features
        feats = self.backbone.base_forward(x)
        c4 = feats[-1]  # high-level feature (B, 2048, H/8, W/8) 정도

        p = self.psp(c4)
        p = self.drop1(p)

        p = self.up1(p)
        p = self.drop2(p)

        p = self.up2(p)
        p = self.drop3(p)

        p = self.up3(p)
        p = self.drop4(p)

        out = self.classifier(p)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        return out
