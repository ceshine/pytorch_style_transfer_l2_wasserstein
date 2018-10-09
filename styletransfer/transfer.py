import numpy as np
import torch.nn as nn
import torch
from torch.optim import LBFGS
from torchvision import transforms
from PIL import Image

from .vgg_model import VggNetwork
from .loss_function import calc_style_desc, calc_l2_wass_dist


LAYERS_TO_USE = ["relu1", "relu2", "relu3", "relu4"]
WEIGHTS = [1, 1, 1, 1]
DEVICE = torch.device("cuda")


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
    def __init__(self):
        self.vgg = VggNetwork().to(DEVICE)
        self.n_iter = 0

    def infer_loss(self, content_tensor, style_desc_dict):
        content_activations = self.vgg(content_tensor.unsqueeze(0))._asdict()
        losses = []
        for layer, weight in zip(LAYERS_TO_USE, WEIGHTS):
            content_style_desc = calc_style_desc(
                content_activations[layer].squeeze(0).permute(1, 2, 0))
            losses.append(calc_l2_wass_dist(
                content_style_desc, style_desc_dict[layer]) * weight)
        print(["%.2f" % x.detach().cpu().numpy() for x in losses])
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
        optimizer = LBFGS([content_tensor], lr=0.8, max_iter=10)
        self.n_iter = 0
        while self.n_iter < steps:
            def closure():
                self.n_iter += 1
                optimizer.zero_grad()
                loss = self.infer_loss(
                    content_tensor, style_desc_dict)
                print(f"Step {self.n_iter}: loss {loss:.2f}")
                loss.backward(retain_graph=True)
                return loss
            optimizer.step(closure)
            self.save_images(content_tensor.unsqueeze(0).to(
                "cpu").detach().numpy(), f"log/{self.n_iter}.jpg")
        return content_tensor.unsqueeze(0)
