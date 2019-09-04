from pathlib import Path
import logging

import numpy as np
import torch.nn as nn
import torch
from torch.optim import LBFGS
from torchvision import transforms
from PIL import Image

from .vgg_model import VggNetwork
from .loss_function import calc_style_desc, calc_l2_wass_dist


LAYERS_TO_USE = ["conv1", "conv3"]
WEIGHTS = np.array([1, 1]) * 1e2
DEVICE = torch.device("cuda")
LOG_DIR = Path("log")
LOG_DIR.mkdir(exist_ok=True)

LOGGER = logging.getLogger(__name__)


def tensor_normalizer():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


def recover_image(img):
    return (
        (
            img *
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) +
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        ).transpose(0, 2, 3, 1) *
        255.
    ).clip(0, 255).astype(np.uint8)


class StyleTransfer:
    def __init__(self, log_interval=0):
        self.vgg = VggNetwork().to(DEVICE)
        self.n_iter = 0
        self.log_interval = log_interval

    def infer_loss(self, content_tensor, style_desc_dict):
        content_activations = self.vgg(content_tensor.unsqueeze(0))._asdict()
        losses = []
        for layer, weight in zip(LAYERS_TO_USE, WEIGHTS):
            content_style_desc = calc_style_desc(
                content_activations[layer].squeeze(0).permute(1, 2, 0))
            losses.append(calc_l2_wass_dist(
                content_style_desc, style_desc_dict[layer]) * weight)
        LOGGER.debug(
            ", ".join(["%.2f" % x.detach().cpu().numpy() for x in losses]))
        return torch.sum(torch.stack(losses))

    def synthesize(self, source, style, output_path, steps=50):
        transform = transforms.Compose([
            transforms.ToTensor(), tensor_normalizer()])
        content_tensor = nn.Parameter(
            transform(source.copy()).to(DEVICE))
        style_tensor = transform(style).to(DEVICE)
        style_activations = self.vgg(style_tensor.unsqueeze(0))._asdict()
        style_desc_dict = {}
        for layer in LAYERS_TO_USE:
            print(layer, style_activations[layer].size())
            style_desc_dict[layer] = calc_style_desc(
                style_activations[layer].squeeze(0).permute(1, 2, 0),
                take_root=True)
        result = self.optimize(
            content_tensor, style_desc_dict, steps).to("cpu").detach().numpy()
        self.save_images(result, output_path)

    @staticmethod
    def save_images(img_arr, output_path):
        result = recover_image(img_arr)[0]
        Image.fromarray(result).save(output_path)

    def optimize(self, content_tensor, style_desc_dict, steps):
        optimizer = LBFGS([content_tensor], lr=0.8, max_iter=steps)
        self.n_iter = 0

        def closure():
            self.n_iter += 1
            optimizer.zero_grad()
            loss = self.infer_loss(
                content_tensor, style_desc_dict)
            LOGGER.info("Step %d: loss %.2f", self.n_iter, loss)
            loss.backward(retain_graph=True)
            if self.log_interval > 0 and self.n_iter % self.log_interval == 0:
                self.save_images(content_tensor.unsqueeze(0).to(
                    "cpu").detach().numpy(), LOG_DIR / f"{self.n_iter:03d}.jpg")
            return loss

        optimizer.step(closure)
        return content_tensor.unsqueeze(0)
