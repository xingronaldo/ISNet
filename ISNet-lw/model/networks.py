import torch
import torch.nn as nn
import torch.nn.init as init
from dcn_v2 import DCN


def init_method(net, init_type='normal'):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
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


def define_model(dim=128, init_type='normal', initialize=True, gpu_ids=[]):

    if dim == 128:
        weights = 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth'
    elif dim == 192:
        weights = 'https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth'
    elif dim == 256:
        weights = 'https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth'
    elif dim == 384:
        weights = 'https://dl.fbaipublicfiles.com/LeViT/LeViT-384-9bdaf2e2.pth'
    else:
        raise NotImplementedError

    net = Model(dim=dim)

    print_network(net)
    init_net(net, init_type, initialize, gpu_ids)

    checkpoint = torch.hub.load_state_dict_from_url(weights, map_location='cpu')
    checkpoint_model = checkpoint['model']
    all_pre_keys = checkpoint_model.keys()
    re_str = ['patch_embed.0', 'patch_embed.2', 'patch_embed.4', 'patch_embed.6']
    new_str = ['cnn1.0', 'cnn2.0', 'cnn3.0', 'cnn4.0']

    for i, search_str in enumerate(re_str):
        for item in list(all_pre_keys):
            if search_str in item:
                replace_name = item.replace(re_str[i], new_str[i])
                checkpoint_model[replace_name] = checkpoint_model.pop(item)

    ## preposcess the pretrained model
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in new_str}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    return net



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
        fea_align = self.relu(self.dcpack_L2([fea2, offset], None))
        fea = torch.cat([fea1, fea_align], dim=1)

        return fea


class Model(nn.Module):

    def __init__(self, dim=192):
        super().__init__()


        self.cnn1 = torch.nn.Sequential(nn.Conv2d(3, dim // 8, 3, 2, 1),
                                        nn.BatchNorm2d(dim // 8),
                                        nn.Hardswish()
                                        )
        self.cnn2 = torch.nn.Sequential(nn.Conv2d(dim // 8, dim // 4, 3, 2, 1),
                                        nn.BatchNorm2d(dim // 4),
                                        nn.Hardswish()
                                        )
        self.cnn3 = torch.nn.Sequential(nn.Conv2d(dim // 4, dim // 2, 3, 2, 1),
                                        nn.BatchNorm2d(dim // 2),
                                        nn.Hardswish()
                                        )
        self.cnn4 = torch.nn.Sequential(nn.Conv2d(dim // 2, dim, 3, 2, 1),
                                        nn.BatchNorm2d(dim),
                                        nn.Hardswish()
                                        )


        self.ca_2 = ChannelAttention(input_nc=dim // 4, ratio=8)
        self.ca_3 = ChannelAttention(input_nc=dim // 2, ratio=8)
        self.ca_4 = ChannelAttention(input_nc=dim, ratio=8)

        self.align_2 = MarginMaximization(num_filters=dim // 4)
        self.align_3 = MarginMaximization(num_filters=dim // 2)
        self.align_4 = MarginMaximization(num_filters=dim)

        self.sa_2 = SpatialAttention(kernel_size=3)
        self.sa_3 = SpatialAttention(kernel_size=3)
        self.sa_4 = SpatialAttention(kernel_size=3)

        self.up_4to3 = nn.PixelShuffle(upscale_factor=2)
        self.up_3to2 = nn.PixelShuffle(upscale_factor=2)

        self.classifier = nn.Sequential(nn.PixelShuffle(2),
                                        nn.Conv2d(7*dim//32, 8, 3, 1, padding=1),
                                        nn.BatchNorm2d(8),
                                        nn.ReLU(),
                                        nn.PixelShuffle(2)
                                        )

    def cnn_forward(self, x):
        fea1 = self.cnn1(x)
        fea2 = self.cnn2(fea1)
        fea3 = self.cnn3(fea2)
        fea4 = self.cnn4(fea3)

        return fea1, fea2, fea3, fea4

    def classifier_forward(self, fea):
        pred = self.classifier(fea)

        return pred

    def forward(self, t1_img, t2_img):
        t1_fea1, t1_fea2, t1_fea3, t1_fea4 = self.cnn_forward(t1_img)
        t2_fea1, t2_fea2, t2_fea3, t2_fea4 = self.cnn_forward(t2_img)

        fea4 = self.align_4(t1_fea4, t2_fea4)
        fea4 = self.sa_4(fea4) * fea4
        fea4 = self.up_4to3(fea4)

        fea3 = self.align_3(t1_fea3, t2_fea3)
        fea3 = self.sa_3(fea3) * fea3
        fea3 = torch.cat([fea3, fea4], dim=1)
        fea3 = self.up_3to2(fea3)

        fea2 = self.align_2(t1_fea2, t2_fea2)
        fea2 = self.sa_2(fea2) * fea2
        fea = torch.cat([fea2, fea3], dim=1)

        pred = self.classifier_forward(fea)

        return pred

