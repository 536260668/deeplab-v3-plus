"""
Created on July 31 2022
@author: Liu Ziheng
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.freeze import set_trainable
"""
ResNet
"""


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)

        self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)



    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features
"""
Xception
Pretrained model from https://github.com/Cadene/pretrained-models.pytorch
by Remi Cadene

"""


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding,
                               dilation=dilation, groups=in_channels, bias=bias)
        self.bn = BatchNorm(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, exit_flow=False, use_1st_relu=True):
        super(Block, self).__init__()

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        self.relu = nn.ReLU()

        rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=stride, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        if exit_flow:
            rep[3:6] = rep[:3]
            rep[:3] = [
                self.relu,
                SeparableConv2d(in_channels, in_channels, 3, 1, dilation),
                nn.BatchNorm2d(in_channels)]

        if not use_1st_relu: rep = rep[1:]
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        output = self.rep(x)
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        x = output + skip
        return x


class Xception(nn.Module):
    def __init__(self,pretrained_path, output_stride=16, in_channels=3, pretrained=True):
        super(Xception, self).__init__()

        # Stride for block 3 (entry flow), and the dilation rates for middle flow and exit flow
        if output_stride == 16: b3_s, mf_d, ef_d = 2, 1, (1, 2)
        if output_stride == 8: b3_s, mf_d, ef_d = 1, 2, (2, 4)

        # Entry Flow
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, stride=2, dilation=1, use_1st_relu=False)
        self.block2 = Block(128, 256, stride=2, dilation=1)
        self.block3 = Block(256, 728, stride=b3_s, dilation=1)

        # Middle Flow
        for i in range(16):
            exec(f'self.block{i + 4} = Block(728, 728, stride=1, dilation=mf_d)')

        # Exit flow
        self.block20 = Block(728, 1024, stride=1, dilation=ef_d[0], exit_flow=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=ef_d[1])
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=ef_d[1])
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=ef_d[1])
        self.bn5 = nn.BatchNorm2d(2048)

        if pretrained: self._load_pretrained_model(pretrained_path)

    def _load_pretrained_model(self,pretrained_path):
        pretrained_weights = torch.load(pretrained_path)
        state_dict = self.state_dict()
        model_dict = {}

        for k, v in pretrained_weights.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)  # [C, C] -> [C, C, 1, 1]
                if k.startswith('block11'):
                    # In Xception there is only 8 blocks in Middle flow
                    model_dict[k] = v
                    for i in range(8):
                        model_dict[k.replace('block11', f'block{i + 12}')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.block1(x)
        low_level_features = x
        x = F.relu(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_features


"""
ASPP
"""
class ASPP(nn.Module):
    def __init__(self, in_channels,output_stride,out_channels = 256):
        super(ASPP, self).__init__()
        assert (output_stride == 8 or output_stride == 16)
        if output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            dilations = [1, 6, 12, 18]


        self.Aspp1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=0,
                          dilation=dilations[0], bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())

        self.Aspp2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                          padding=dilations[1], dilation=dilations[1], bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        self.Aspp3 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                          padding=dilations[2], dilation=dilations[2], bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        self.Aspp4 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                          padding=dilations[3], dilation=dilations[3], bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(2048, 256, kernel_size=(1, 1), stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        self.conv1 = nn.Conv2d(1280, 256, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print("X1.shape", x.size())
        x1 = self.Aspp1(x);
        # print("X1.shape", x1.size())
        x2 = self.Aspp2(x);
        # print("X2.shape", x2.size())
        x3 = self.Aspp3(x);
        # print("X3.shape", x3.size())
        x4 = self.Aspp4(x);
        # print("X4.shape", x4.size())
        x5 = self.global_avg_pool(x);
        # print('x5.shape', x5.size())
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        # print('x5.shape', x5.size())
        cat = torch.cat((x1, x2, x3, x4, x5), dim=1);
            # print('cat.shape', cat.size())

        x = self.conv1(cat)
        x = self.bn1(x);
            # print('output.shape', output.size())
        x = self.dropout(self.relu(x))
        return x


"""
decoder
"""
class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()

        self.conv =nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )

    def forward(self, x, low_level_features):
        low_level_features = self.conv(low_level_features)
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x



class deeplabv3_plus(nn.Module):
    def __init__(self,num_classes,in_channels = 3,backbone='xception',pretrained = True,
                 output_stride = 16, freeze_bn=True,freeze_backbone=True):
        super(deeplabv3_plus, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 256
        else:
            self.backbone = Xception(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained,
                                     pretrained_path='./pretrained/xception-b5690688.pth')
            low_level_channels = 128
        self.ASPP = ASPP(in_channels=2048, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)

        # print([self.backbone])
        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.backbone], False)

    def forward(self, x):

        torch.autograd.set_detect_anomaly(True)

        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        x = self.ASPP(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == '__main__':
    model = deeplabv3_plus(2,3)
    print(1)