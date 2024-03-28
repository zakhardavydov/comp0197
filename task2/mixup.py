import torch

from PIL import Image
from torch.distributions import Beta
from torchvision.utils import make_grid, save_image


class BatchMixUp:

    def __init__(self, sampling_method: int = 1, alpha=0.2):
        self._sampling_method = sampling_method
        self._alpha = alpha

    def sample(self):
        if self._sampling_method == 1:
            return Beta(self._alpha, self._alpha).sample()
        return torch.rand(1)

    def mix_up(self, batch):
        img, labels = batch

        batch_size = img.size()[0]
        index = torch.randperm(batch_size)

        l = self.sample()

        mixed_img = l * img + (1 - l) * img[index, :]
        mixed_labels = l * labels + (1 - l) * labels[index]
        return mixed_img, mixed_labels

    def visualize(self, batch, save_path: str = "mixup.png"):
        images, _ = batch
        grid = make_grid(images, nrow=4)
        save_image(grid, save_path)
        return save_path
