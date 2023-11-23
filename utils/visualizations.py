import torch
from torch import Tensor
from torchvision import transforms
from torchvision.utils import make_grid

from base.base_dataloader import BaseDataLoader
from utils import transforms as local_transforms
from utils.helpers import colorize_mask


class Visualization:
    def __init__(self, train_loader: BaseDataLoader):
        self.train_loader = train_loader
        self.restore_transform = transforms.Compose(
            [
                local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
                transforms.ToPILImage(),
            ]
        )
        self.viz_transform = transforms.Compose(
            [transforms.Resize((400, 400)), transforms.ToTensor()]
        )
        self.visual = []

    def add_to_visual(self, targets: Tensor, outputs: Tensor) -> None:
        if len(self.visual) < 15:
            target_np = targets.data.cpu().numpy()
            output_np = outputs.data.max(1)[1].cpu().numpy()
            self.visual.append([outputs[0].data.cpu(), target_np[0], output_np[0]])

    def flush_visual(self) -> Tensor:
        images = []
        palette = self.train_loader.dataset.palette
        for d, t, o in self.visual:
            d = self.restore_transform(d)
            t, o = colorize_mask(t, palette), colorize_mask(o, palette)
            d, t, o = d.convert("RGB"), t.convert("RGB"), o.convert("RGB")
            [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
            images.extend([d, t, o])
        images = torch.stack(images, 0)
        images = make_grid(images.cpu(), nrow=3, padding=5)
        self.visual = []
        return images
