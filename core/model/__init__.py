import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import resnet50


from .resnet import  ResNet, Bottleneck, BasicBlock

from .wrn import wrn

def build_model_res50gn(group_norm, num_classes):
    print('Building model...')
    def gn_helper(planes):
        return nn.GroupNorm(group_norm, planes)
    net = ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=num_classes, norm_layer=gn_helper)
    return net

def build_model_res18bn(num_classes):
    print('Building model...')
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, norm_layer=nn.BatchNorm2d)

def build_model_wrn2810bn(num_classes):
    print('Building model...')
    return wrn(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)

def create_resnet_gn(NUM_CLASSES):
    def group_norm_factory(channels):
        return nn.GroupNorm(32, channels)

    model = resnet50(
        pretrained=False,
        num_classes=NUM_CLASSES,
        norm_layer=group_norm_factory,
    )
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    return model