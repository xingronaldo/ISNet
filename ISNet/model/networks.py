import torch
import torch.nn as nn
import torch.nn.init as init

from dcn_v2 import DCN
from model.resnet import resnet18, resnet34, resnet50, resnext50_32x4d, wide_resnet50_2


def init_method(net, init_type='normal'):
    def init_func(m): 
        classname = m.__class__.__name__
        if hasattr(m, 'resnet'):
            pass
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=0.02)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=0.02)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


#func called to initialize a net
def init_net(net, init_type='normal', initialize=True, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if initialize:
        init_method(net, init_type)
    else:
        pass
    return net


#print #parameters
def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



def define_model(model_type='PartialMM', resnet='resnet18', init_type='normal', initialize=True, gpu_ids=[]):

    if model_type == 'FullMM':
        net = Model_FullMM(resnet=resnet)
    elif model_type == 'PartialMM':
        net = Model_PartialMM(resnet=resnet)
    else:
        raise NotImplementedError
    print_network(net)

    return init_net(net, init_type, initialize, gpu_ids)


class ChannelAttention(nn.Module):
    def __init__(self, input_nc, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(input_nc, input_nc // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(input_nc // ratio, input_nc, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MarginMaximization(nn.Module):
    def __init__(self, num_filters=128):
        super().__init__()
        self.offset = nn.Sequential(nn.Conv2d(num_filters*2, num_filters, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(num_filters))
        self.dcpack_L2 = DCN(num_filters, num_filters, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                                extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fea1, fea2):

        offset = self.offset(torch.cat([fea1, fea2], dim=1))
        fea_mm = self.relu(self.dcpack_L2([fea2, offset], None))
        fea = torch.cat([fea1, fea_mm], dim=1)

        return fea


class Model_FullMM(nn.Module):

    def __init__(self, resnet='resnet18'):
        super().__init__()

        if resnet == 'resnet18':
            self.resnet = resnet18(pretrained=True,
                                   replace_stride_with_dilation=[False, False, True])
        elif resnet == 'resnet34':
            self.resnet = resnet34(pretrained=True,
                                   replace_stride_with_dilation=[False, False, True])
        elif resnet == 'resnet50':
            self.resnet = resnet50(pretrained=True,
                                   replace_stride_with_dilation=[False, False, True])
        elif resnet == 'resnext50_32x4d':
            self.resnet = resnext50_32x4d(pretrained=True,
                                   replace_stride_with_dilation=[False, False, True])

        elif resnet == 'wide_resnet50_2':
            self.resnet = wide_resnet50_2(pretrained=True,
                                          replace_stride_with_dilation=[False, False, True])
        else:
            raise NotImplementedError

        if resnet in ['resnet18', 'resnet34']:
            self.ca_1 = ChannelAttention(input_nc=64, ratio=8)
            self.ca_2 = ChannelAttention(input_nc=128, ratio=8)
            self.ca_3 = ChannelAttention(input_nc=256, ratio=8)
            self.ca_4 = ChannelAttention(input_nc=512, ratio=8)

            self.align_1 = MarginMaximization(num_filters=64)
            self.align_2 = MarginMaximization(num_filters=128)
            self.align_3 = MarginMaximization(num_filters=256)
            self.align_4 = MarginMaximization(num_filters=512)

            self.sa_1 = SpatialAttention(kernel_size=3)
            self.sa_2 = SpatialAttention(kernel_size=3)
            self.sa_3 = SpatialAttention(kernel_size=3)
            self.sa_4 = SpatialAttention(kernel_size=3)

            self.conv_4to3 = nn.Conv2d(1024, 512, 1, 1, padding=0)
            self.up_3to2 = nn.PixelShuffle(upscale_factor=2)
            self.up_2to1 = nn.PixelShuffle(upscale_factor=2)

            self.classifier = nn.Sequential(nn.PixelShuffle(2),
                                            nn.Conv2d(64, 8, 3, 1, padding=1),
                                            nn.BatchNorm2d(8),
                                            nn.ReLU(),
                                            nn.PixelShuffle(2)
                                            )

        elif resnet in ['resnet50', 'resnext50_32x4d', 'wide_resnet50_2']:
            self.ca_1 = ChannelAttention(input_nc=256, ratio=8)
            self.ca_2 = ChannelAttention(input_nc=512, ratio=8)
            self.ca_3 = ChannelAttention(input_nc=1024, ratio=8)
            self.ca_4 = ChannelAttention(input_nc=2048, ratio=8)

            self.align_1 = MarginMaximization(num_filters=256)
            self.align_2 = MarginMaximization(num_filters=512)
            self.align_3 = MarginMaximization(num_filters=1024)
            self.align_4 = MarginMaximization(num_filters=2048)

            self.sa_1 = SpatialAttention(kernel_size=3)
            self.sa_2 = SpatialAttention(kernel_size=3)
            self.sa_3 = SpatialAttention(kernel_size=3)
            self.sa_4 = SpatialAttention(kernel_size=3)

            self.conv_4to3 = nn.Conv2d(4096, 2048, 1, 1, padding=0)
            self.up_3to2 = nn.PixelShuffle(upscale_factor=2)
            self.up_2to1 = nn.PixelShuffle(upscale_factor=2)

            self.classifier = nn.Sequential(nn.PixelShuffle(2),
                                            nn.Conv2d(256, 64, 3, 1, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(),
                                            nn.Conv2d(64, 8, 1, 1, padding=0),
                                            nn.PixelShuffle(2)
                                            )
        else:
            raise NotImplementedError

    def resnet_forward(self, x):
        fea0 = self.resnet.conv1(x)
        fea0 = self.resnet.bn1(fea0)
        fea0 = self.resnet.relu(fea0)
        fea0 = self.resnet.maxpool(fea0)

        fea1 = self.resnet.layer1(fea0)
        fea1 = self.ca_1(fea1) * fea1
        fea2 = self.resnet.layer2(fea1)
        fea2 = self.ca_2(fea2) * fea2
        fea3 = self.resnet.layer3(fea2)
        fea3 = self.ca_3(fea3) * fea3
        fea4 = self.resnet.layer4(fea3)
        fea4 = self.ca_4(fea4) * fea4

        return fea1, fea2, fea3, fea4


    def classifier_forward(self, fea):
        pred = self.classifier(fea)

        return pred

    def forward(self, t1_img, t2_img):
        t1_fea1, t1_fea2, t1_fea3, t1_fea4 = self.resnet_forward(t1_img)
        t2_fea1, t2_fea2, t2_fea3, t2_fea4 = self.resnet_forward(t2_img)

        fea4 = self.align_4(t1_fea4, t2_fea4)
        fea4 = self.sa_4(fea4) * fea4
        fea4 = self.conv_4to3(fea4)

        fea3 = self.align_3(t1_fea3, t2_fea3)
        fea3 = self.sa_3(fea3) * fea3
        fea3 = torch.cat([fea3, fea4], dim=1)
        fea3 = self.up_3to2(fea3)

        fea2 = self.align_2(t1_fea2, t2_fea2)
        fea2 = self.sa_2(fea2) * fea2
        fea2 = torch.cat([fea2, fea3], dim=1)
        fea2 = self.up_2to1(fea2)

        fea1 = self.align_1(t1_fea1, t2_fea1)
        fea1 = self.sa_1(fea1) * fea1
        fea = torch.cat([fea1, fea2], dim=1)

        pred = self.classifier_forward(fea)

        return pred


class Model_PartialMM(nn.Module):

    def __init__(self, resnet='resnet18'):
        super().__init__()

        if resnet == 'resnet18':
            self.resnet = resnet18(pretrained=True,
                                   replace_stride_with_dilation=[False, False, True])
        elif resnet == 'resnet34':
            self.resnet = resnet34(pretrained=True,
                                   replace_stride_with_dilation=[False, False, True])
        elif resnet == 'resnet50':
            self.resnet = resnet50(pretrained=True,
                                   replace_stride_with_dilation=[False, False, True])
        elif resnet == 'resnext50_32x4d':
            self.resnet = resnext50_32x4d(pretrained=True,
                                          replace_stride_with_dilation=[False, False, True])

        elif resnet == 'wide_resnet50_2':
            self.resnet = wide_resnet50_2(pretrained=True,
                                          replace_stride_with_dilation=[False, False, True])
        else:
            raise NotImplementedError

        if resnet in ['resnet18', 'resnet34']:
            self.ca_1 = ChannelAttention(input_nc=64, ratio=8)
            self.ca_2 = ChannelAttention(input_nc=128, ratio=8)
            self.ca_3 = ChannelAttention(input_nc=256, ratio=8)
            self.ca_4 = ChannelAttention(input_nc=512, ratio=8)

            self.align_1 = MarginMaximization(num_filters=64)
            self.align_2 = MarginMaximization(num_filters=128)

            self.sa_1 = SpatialAttention(kernel_size=3)
            self.sa_2 = SpatialAttention(kernel_size=3)
            self.sa_3 = SpatialAttention(kernel_size=3)
            self.sa_4 = SpatialAttention(kernel_size=3)

            self.conv_4to3 = nn.Conv2d(1024, 512, 1, 1, padding=0)
            self.up_3to2 = nn.PixelShuffle(upscale_factor=2)
            self.up_2to1 = nn.PixelShuffle(upscale_factor=2)

            self.classifier = nn.Sequential(nn.PixelShuffle(2),
                                            nn.Conv2d(64, 8, 3, 1, padding=1),
                                            nn.BatchNorm2d(8),
                                            nn.ReLU(),
                                            nn.PixelShuffle(2)
                                            )

        elif resnet in ['resnet50', 'resnext50_32x4d', 'wide_resnet50_2']:
            self.ca_1 = ChannelAttention(input_nc=256, ratio=8)
            self.ca_2 = ChannelAttention(input_nc=512, ratio=8)
            self.ca_3 = ChannelAttention(input_nc=1024, ratio=8)
            self.ca_4 = ChannelAttention(input_nc=2048, ratio=8)

            self.align_1 = MarginMaximization(num_filters=256)
            self.align_2 = MarginMaximization(num_filters=512)

            self.sa_1 = SpatialAttention(kernel_size=3)
            self.sa_2 = SpatialAttention(kernel_size=3)
            self.sa_3 = SpatialAttention(kernel_size=3)
            self.sa_4 = SpatialAttention(kernel_size=3)

            self.conv_4to3 = nn.Conv2d(4096, 2048, 1, 1, padding=0)
            self.up_3to2 = nn.PixelShuffle(upscale_factor=2)
            self.up_2to1 = nn.PixelShuffle(upscale_factor=2)

            self.classifier = nn.Sequential(nn.PixelShuffle(2),
                                            nn.Conv2d(256, 64, 3, 1, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(),
                                            nn.Conv2d(64, 8, 1, 1, padding=0),
                                            nn.PixelShuffle(2)
                                            )

    def resnet_forward(self, x):
        fea0 = self.resnet.conv1(x)
        fea0 = self.resnet.bn1(fea0)
        fea0 = self.resnet.relu(fea0)
        fea0 = self.resnet.maxpool(fea0)

        fea1 = self.resnet.layer1(fea0)
        fea1 = self.ca_1(fea1) * fea1
        fea2 = self.resnet.layer2(fea1)
        fea2 = self.ca_2(fea2) * fea2
        fea3 = self.resnet.layer3(fea2)
        fea3 = self.ca_3(fea3) * fea3
        fea4 = self.resnet.layer4(fea3)
        fea4 = self.ca_4(fea4) * fea4

        return fea1, fea2, fea3, fea4

    def classifier_forward(self, fea):
        pred = self.classifier(fea)

        return pred

    def forward(self, t1_img, t2_img):
        t1_fea1, t1_fea2, t1_fea3, t1_fea4 = self.resnet_forward(t1_img)
        t2_fea1, t2_fea2, t2_fea3, t2_fea4 = self.resnet_forward(t2_img)

        fea4 = torch.cat([t1_fea4, t2_fea4], dim=1)
        fea4 = self.sa_4(fea4) * fea4
        fea4 = self.conv_4to3(fea4)

        fea3 = torch.cat([t1_fea3, t2_fea3], dim=1)
        fea3 = self.sa_3(fea3) * fea3
        fea3 = torch.cat([fea3, fea4], dim=1)
        fea3 = self.up_3to2(fea3)

        fea2 = self.align_2(t1_fea2, t2_fea2)
        fea2 = self.sa_2(fea2) * fea2
        fea2 = torch.cat([fea2, fea3], dim=1)
        fea2 = self.up_2to1(fea2)

        fea1 = self.align_1(t1_fea1, t2_fea1)
        fea1 = self.sa_1(fea1) * fea1
        fea = torch.cat([fea1, fea2], dim=1)

        pred = self.classifier_forward(fea)

        return pred


