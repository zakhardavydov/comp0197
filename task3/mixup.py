import torch

from torch.distributions import Beta
from torchvision.utils import make_grid, save_image


class BatchMixUp:

    def __init__(self, sampling_method: int = 1, alpha=0.2):
        """
        Mix up can be configured: by sampling method and by alpha when sampling method == 1
        """
        self._sampling_method = sampling_method
        self._alpha = alpha

    def sample(self):
        """
        Generate lambda based on the sampling method initiated
        """
        if self._sampling_method == 1:
            return Beta(self._alpha, self._alpha).sample()
        return torch.rand(1)

    def mix_up(self, batch):
        """
        Mix up that works within the batch
        """
        img, labels = batch

        batch_size = img.size()[0]
        index = torch.randperm(batch_size)

        l = self.sample()

        mixed_img = l * img + (1 - l) * img[index, :]
        mixed_labels = l * labels + (1 - l) * labels[index]

        return mixed_img, mixed_labels

    def visualize(self, batch, save_path: str = "mixup.png"):
        """
        Mix up one batch, visualize in grid and save to file
        """
        images, _ = self.mix_up(batch)
        grid = make_grid(images, nrow=4)
        save_image(grid, save_path)
        return save_path
