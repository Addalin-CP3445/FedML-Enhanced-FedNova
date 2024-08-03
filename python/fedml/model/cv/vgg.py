"Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"

import torch
import torch.nn as nn
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from backpack.extensions.firstorder.base import FirstOrderModuleExtension


__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]

class Conv2dBatchGrad(FirstOrderModuleExtension):
    """Extract individual gradients for Conv2d layers."""

    def __init__(self):
        super().__init__(params=["weight", "bias"])

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        return g_out[0].flatten(start_dim=1).sum(axis=1).unsqueeze(-1)

    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        if module.bias is not None:
            return g_out[0].sum(axis=0)

class LinearBatchGrad(FirstOrderModuleExtension):
    """Extract individual gradients for Linear layers."""

    def __init__(self):
        super().__init__(params=["weight", "bias"])

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        return g_out[0].flatten(start_dim=1).sum(axis=1).unsqueeze(-1)

    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        if module.bias is not None:
            return g_out[0].sum(axis=0)


# class VGG(nn.Module):
#     def __init__(
#         self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True
#     ) -> None:
#         super(VGG, self).__init__()
#         self.features = features
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             extend(nn.Linear(512 * 7 * 7, 4096)),
#             nn.ReLU(inplace=False),
#             nn.Dropout(),
#             extend(nn.Linear(4096, 4096)),
#             nn.ReLU(inplace=False),
#             nn.Dropout(),
#             extend(nn.Linear(4096, num_classes)),
#         )
#         if init_weights:
#             self._initialize_weights()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

#     def _initialize_weights(self) -> None:
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             # elif isinstance(m, nn.BatchNorm2d):
#             elif isinstance(m, nn.GroupNorm):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)


# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == "M":
#             layers = layers + [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             v = int(v)
#             conv2d = extend(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
#             if batch_norm:
#                 # layers = layers + [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
#                 layers = layers + [conv2d, nn.GroupNorm(32, v), nn.ReLU(inplace=False)]
#             else:
#                 layers = layers + [conv2d, nn.ReLU(inplace=False)]
#             in_channels = v
#     return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = self._extend_layers(features)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self._extend_layers(nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        ))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _extend_layers(self, module):
        for name, layer in module.named_children():
            module._modules[name] = extend(layer)
        return module

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [extend(conv2d), nn.GroupNorm(32, v), nn.ReLU(inplace=False)]
            else:
                layers += [extend(conv2d), nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def vgg11(num_classes = 1000):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    batch_grad_extension = BatchGrad()
    batch_grad_extension.set_module_extension(nn.Conv2d, Conv2dBatchGrad())
    batch_grad_extension.set_module_extension(nn.Linear, LinearBatchGrad())
    return VGG(make_layers(cfgs["A"]), num_classes=num_classes)


def vgg11_bn():
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs["A"], batch_norm=True))


def vgg13():
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs["B"]))


def vgg13_bn():
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs["A"], batch_norm=True))


def vgg16():
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs["D"]))


def vgg16_bn():
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs["D"], batch_norm=True))


def vgg19():
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs["E"]))


def vgg19_bn():
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG(make_layers(cfgs["E"], batch_norm=True))
