import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """
    # phase는 train or test
    def __init__(self, phase, size, base, extras, head, num_classes, 
                tiny=False, box_size_change=False, minmax=False):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.tiny = tiny

        self.box_size_change = box_size_change
        self.cfg = (coco, voc)[num_classes == 21] # self.cfg = coco
        # cfg is data info
        self.priorbox = PriorBox(self.cfg, self.box_size_change, minmax)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
        # if phase == 'val':
        #     self.softmax = nn.Softmax(dim=-1)
        #     self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        if self.tiny:
            # print('original image', x.size()) # 7이나 4가 나오는 이유는 Multi-GPU!
            # apply vgg11 up to conv4_2 relu
            for k in range(15):
                x = self.vgg[k](x)
                # print('{}번째'.format(k), x.size())
            s = self.L2Norm(x)
            sources.append(s)
            # conv4_2에서 나온 피처맵 추가

            # apply vgg up to fc7
            for k in range(15, len(self.vgg)):
                x = self.vgg[k](x)
                # print('{}번째'.format(k), x.size())
            sources.append(x)
            # conv7에서 나온 피처맵 추가
        else:
            # apply vgg16 up to conv4_3 relu
            for k in range(23):
                x = self.vgg[k](x)
                # print('{}번째'.format(k), x.size())
            s = self.L2Norm(x)
            sources.append(s)
            # conv4_3에서 나온 피처맵 추가

            # apply vgg up to fc7
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)
                # print('{}번째'.format(k), x.size())
            sources.append(x)
            # conv7에서 나온 피처맵 추가

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                # print('extra feature map', x.size())
                sources.append(x)
                # 나머지 4개의 피처 맵 추가

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # sources의 피처맵을 loc과 conf 레이어 input으로 취한다

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg11():
    # cfg는 300, i는 3
    layers = []

    # 이렇게 만든 모델은 추후에 eval 시 문제가 날 수도 있다 => vgg10.pth load 해결?
    vgg_pretrained = models.vgg11(pretrained=True).features[:-1]
    vgg_pretrained[10] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    for i in range(len(vgg_pretrained)):
        layers += [vgg_pretrained[i]]

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    print('vgg11을 선택!')

    return layers

    # 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    # 300: [64, 'M', 128, 'M', 256, 256, 'C', 512, 512, 'M', 512, 512]
    # conv1 -> conv2 -> conv3-1 -> conv3-2 -> conv4-1 -> conv4-2 -> conv5-1 -< conv5-2
    # 64, Relu, 'M', 128, Relu, 'M', 256, Relu, 256, Relu, 'M', 512, Relu, 512, Relu, 'M', | len is 16
    # 512, Relu, 512, Relu, 'M' | len is 5
    # VGG Network의 총 길이는 21
    # 64, Relu, 'M', 128, Relu, 'M', 256, Relu, 256, Relu, 'C', 512, Relu, 512, Relu[14], 'M', | len is 16
    # 512, Relu, 512, Relu, 'M' 1024, Relu, 1024, Relu | len is 9
    # BASE Network의 총 길이는 25

def vgg(cfg, i, batch_norm=False):
    # cfg는 300, i는 3
    layers = []
    # in_channels = i

    # 이렇게 만든 모델은 추후에 eval 시 문제가 날 수도 있다 => vgg16.pth load 해결?
    vgg_pretrained = models.vgg16(pretrained=True).features[:-1]
    vgg_pretrained[16] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    for i in range(len(vgg_pretrained)):
        layers += [vgg_pretrained[i]]

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6) # conv6
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1) # conv7

    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers

    # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'] list len 18
    # '300'[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512] list len 17
    # Pytorch VGG에는 Conv 뒤에 Relu가 붙기 때문에 인덱싱할 때 주의
    # 64, Relu, 64, Relu, 'M', 128, Relu, 128, Relu, 'M', 256, Relu, 256, Relu, 256, Relu, 'M', 512, Relu, 512, | len 20
    # Relu, 512, Relu, 'M', 512, Relu, 512, Relu, 512, Relu,'M' | len 11
    # VGG Network의 총 길이는 31
    # 64, Relu, 64, Relu, 'M', 128, Relu, 128, Relu, 'M', 256, Relu, 256, Relu, 256, Relu, 'C', 512, Relu, 512, | len 20
    # Relu, 512, Relu[22], 'M', 512, Relu, 512, Relu, 512, Relu, 'M', 1024, Relu, 1024, Relu | len 15
    # SSD BASE VGG Network 총 길이는 35

    # vgg_pretrained = models.vgg16(pretrained=True).features[:-1]
    # check! vgg_pretrained[16] = nn.MaxPool2d(kernel_size=2, stride=2,padding=0, dilation=1 , ceil_mode=True)
    # for i in range(len(vgg_pretrained)) :
    #     vgg += [vgg_pretrained[i]]
    # vgg += [nn.MaxPool2d(kernel_size=3, stride=1,padding=1, dilation=1 , ceil_mode=False)]
    # vgg += [nn.Conv2d(512,1024,kernel_size=(3,3),stride=(1,1),padding=(6,6),dilation=(6,6))]
    # vgg += [nn.ReLU(inplace=True)]
    # vgg += [nn.Conv2d(1024,1024,kernel_size=(1,1),stride=(1,1))]
    # vgg += [nn.ReLU(inplace=True)]


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    # flag는 False와 True가 반복된다
    for k, v in enumerate(cfg):
        # 'S'의 의미는 stride가 2!
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

    # '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]

    # extra_layers = []
    # 1024-256 extra_layers += [nn.Conv2d(1024, 256, kernel_size=(1, 1),stride=(1,1))]
    # 'S', 256-512 extra_layers += [nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2,2), padding=(1,1))]
    # 512-128 extra_layers += [nn.Conv2d(512, 128, kernel_size=(1, 1),stride=(1,1))]
    # 'S', 128-256 extra_layers += [nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2,2), padding=(1,1))]
    # 256-128 extra_layers += [nn.Conv2d(256, 128, kernel_size=(1, 1),stride=(1,1))]
    # 128-256 extra_layers += [nn.Conv2d(128, 256, kernel_size=(3, 3),stride=(1,1))]
    # 256-128 extra_layers += [nn.Conv2d(256, 128, kernel_size=(1, 1),stride=(1,1))]
    # 128-256 extra_layers += [nn.Conv2d(128, 256, kernel_size=(3, 3),stride=(1,1))]
    # EXTRA layer 길이는 8

def multibox(vgg, extra_layers, cfg, num_classes, tiny=False):
    loc_layers = []
    conf_layers = []
    if tiny:
        vgg_source = [13, -2]
    else:
        vgg_source = [21, -2] # VGG, 21 INDEX is Conv(512,512), -2 INDEX is Conv7(1024,1024)
    # VGG에서 각각 2개의 conf와 loc Conv layer 생성
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        # cfg[k] * 4는 xmin, ymin, xmax, ymas 때문
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
        # cfg[k] * num_classess는 class 개수를 고려하기 때문

    # EXTRA에서 각각 2개의 conf와 loc Conv layer 생성
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
c_mbox = {
    '300': [3, 5, 5, 5, 3, 3],  # number of boxes per feature map location
    '512': [],
}
o_mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}
# 3 is 1, 2, 1/2 | 5 is 1, 2, 3, 1/2, 1/3


def build_ssd(phase, size=300, num_classes=21, 
                tiny=False, box_size_change=False, minmax=False):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return

    # 추가 가능
    if tiny:
        if box_size_change: # tiny_ssd
            print('totaly tiny ssd!')
            base_, extras_, head_ = multibox(vgg11(),
                                        add_extras(extras[str(size)], 1024),
                                        c_mbox[str(size)], num_classes, tiny)
        else: # tiny_param_ssd
            print('tiny parameters ssd!')
            base_, extras_, head_ = multibox(vgg11(),
                                        add_extras(extras[str(size)], 1024),
                                        o_mbox[str(size)], num_classes, tiny)
    else:
        if box_size_change: # tiny_size_ssd
            print('tiny size ssd!')
            base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                            add_extras(extras[str(size)], 1024),
                                            c_mbox[str(size)], num_classes, tiny)
        else: # ssd
            print('ssd!')
            base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                            add_extras(extras[str(size)], 1024),
                                            o_mbox[str(size)], num_classes, tiny)

    return SSD(phase, size, base_, extras_, head_, num_classes, tiny, box_size_change, minmax)
    
    # 여기 까지는 전부 이해
    # return SSD(phase, size, base_, extras_, head_, num_classes, False)