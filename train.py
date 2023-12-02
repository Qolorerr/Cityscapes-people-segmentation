import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.base.base_dataloader import BaseDataLoader
from src.base.base_model import BaseModel
from src.trainer import Trainer
from src.utils import Logger


@hydra.main(version_base='1.1', config_path="config", config_name="config")
def main(cfg: DictConfig):
    if cfg.resume:
        cfg = DictConfig(torch.load(cfg.resume)["config"])

    train_logger = Logger()

    # DATA LOADERS
    train_loader: BaseDataLoader = instantiate(cfg.train_loader)
    val_loader: BaseDataLoader = instantiate(cfg.val_loader)

    # MODEL
    model: BaseModel = instantiate(cfg.model, train_loader.dataset.num_classes)
    print(f"\n{model}\n")

    # LOSS
    loss: nn.Module = instantiate(cfg.loss)

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
