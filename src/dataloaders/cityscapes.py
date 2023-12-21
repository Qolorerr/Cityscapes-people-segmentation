import os
from glob import glob

import numpy as np
from PIL import Image

from src.base.base_dataloader import BaseDataLoader
from src.base.base_dataset import BaseDataSet


class CityScapesDataset(BaseDataSet):
    def __init__(self, id_to_train_id: dict[int, int], mode: str = "fine", **kwargs):
        self.num_classes = len(set(id_to_train_id.values()))
        self.mode = mode
        self.id_to_train_id = id_to_train_id
        super().__init__(**kwargs)

    def _set_files(self):
        assert (self.mode == "fine" and self.split in ["train", "val"]) or (
            self.mode == "coarse" and self.split in ["train", "train_extra", "val"]
        )

        SUFIX = "_gtFine_labelIds.png"
        if self.mode == "coarse":
            img_dir_name = (
                "leftImg8bit_trainextra"
                if self.split == "train_extra"
                else "leftImg8bit_trainvaltest"
            )
            label_path = os.path.join(self.root, "gtCoarse", "gtCoarse", self.split)
        else:
            img_dir_name = "leftImg8bit_trainvaltest"
            label_path = os.path.join(self.root, "gtFine_trainvaltest", "gtFine", self.split)
        image_path = os.path.join(self.root, img_dir_name, "leftImg8bit", self.split)
        assert os.listdir(image_path) == os.listdir(label_path)

        image_paths, label_paths = [], []
        for city in os.listdir(image_path):
            image_paths.extend(sorted(glob(os.path.join(image_path, city, "*.png"))))
            label_paths.extend(sorted(glob(os.path.join(label_path, city, f"*{SUFIX}"))))
        self.files = list(zip(image_paths, label_paths))

    def _load_data(self, index: int):
        image_path, label_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        for k, v in self.id_to_train_id.items():
            label[label == k] = v
        return image, label, image_id


class CityScapes(BaseDataLoader):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        split: str,
        palette: list[int],
        id_to_train_id: dict[int, int],
        crop_size: int | None = None,
        base_size: int | None = None,
        scale: bool = True,
        num_workers: int = 1,
        mode: str = "fine",
        val: bool = False,
        shuffle: bool = False,
        flip: bool = False,
        rotate: bool = False,
        blur: bool = False,
        augment: bool = False,
        val_split: float | None = None,
        return_id: bool = False,
    ):
        self.MEAN = [0.28689529, 0.32513294, 0.28389176]
        self.STD = [0.17613647, 0.18099176, 0.17772235]

        kwargs = {
            "root": data_dir,
            "split": split,
            "mean": self.MEAN,
            "std": self.STD,
            "augment": augment,
            "crop_size": crop_size,
            "base_size": base_size,
            "scale": scale,
            "flip": flip,
            "blur": blur,
            "rotate": rotate,
            "return_id": return_id,
            "val": val,
            "palette": palette
        }

        self.dataset = CityScapesDataset(id_to_train_id=id_to_train_id, mode=mode, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
