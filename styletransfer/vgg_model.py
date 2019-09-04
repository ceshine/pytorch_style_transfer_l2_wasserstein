from collections import namedtuple

import torch
import torchvision.models.vgg as vgg

LossOutput = namedtuple(
    "LossOutput", ["conv1", "conv2", "conv3", "conv4", "conv5"])


class VggNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(VggNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '2': "conv1",
            '7': "conv2",
            '16': "conv3",
            '25': "conv4",
            '34': "conv5",
        }

    def forward(self, x) -> namedtuple:
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)
