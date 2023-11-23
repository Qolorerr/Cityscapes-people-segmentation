import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

import dataloaders
import models
from base.base_dataloader import BaseDataLoader
from base.base_model import BaseModel
from trainer import Trainer
from utils import Logger, losses


def get_instance(module, name: str, config: DictConfig, *args):
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    if cfg.resume:
        cfg = DictConfig(torch.load(cfg.resume)["config"])

    train_logger = Logger()

    # DATA LOADERS
    train_loader: BaseDataLoader = get_instance(dataloaders, "train_loader", cfg)
    val_loader: BaseDataLoader = get_instance(dataloaders, "val_loader", cfg)

    # MODEL
    model: BaseModel = get_instance(models, "arch", cfg, train_loader.dataset.num_classes)
    print(f"\n{model}\n")

    # LOSS
    loss: nn.Module = getattr(losses, cfg["loss"])(ignore_index=cfg["ignore_index"])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=cfg.resume,
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger,
    )

    trainer.train()


if __name__ == "__main__":
    main()
