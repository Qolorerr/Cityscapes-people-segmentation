import datetime
import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils import tensorboard

from src.base.base_dataloader import BaseDataLoader
from src.base.base_model import BaseModel
from src.utils import Logger


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])


class BaseTrainer:
    def __init__(
        self,
        model: BaseModel,
        loss: nn.Module,
        resume: str,
        config: DictConfig,
        train_loader: BaseDataLoader,
        val_loader: BaseDataLoader | None = None,
        train_logger: Logger | None = None,
    ):
        self.model = model
        self.loss = loss
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.do_validation = self.config["trainer"]["val"]
        self.start_epoch = 1
        self.improved = False
        self.accelerator = Accelerator()

        # SETTING THE DEVICE
        self.model = self.accelerator.prepare(self.model)

        # CONFIGS
        cfg_trainer = self.config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]

        # OPTIMIZER
        if self.config["differential_lr"]:
            if isinstance(self.model, torch.nn.DataParallel):
                decoder_params = self.model.module.get_decoder_params()
                backbone_params = self.model.module.get_backbone_params()
            else:
                decoder_params = self.model.get_decoder_params()
                backbone_params = self.model.get_backbone_params()
            trainable_params = [
                {"params": filter(lambda p: p.requires_grad, decoder_params)},
                {
                    "params": filter(lambda p: p.requires_grad, backbone_params),
                    "lr": config["optimizer"]["lr"] / 10,
                },
            ]
        else:
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = instantiate(self.config.optimizer, trainable_params)
        self.lr_scheduler = instantiate(self.config.lr_scheduler, self.optimizer, self.epochs, len(train_loader))
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )

        # MONITORING
        self.monitor = cfg_trainer.get("monitor", "off")
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]
            self.mnt_best = -math.inf if self.mnt_mode == "max" else math.inf
            self.early_stoping = cfg_trainer.get("early_stop", math.inf)

        # CHECKPOINTS & TENSORBOARD
        start_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        writer_dir = os.path.join(cfg_trainer["log_dir"], self.config["name"], start_time)
        self.writer = tensorboard.SummaryWriter(writer_dir)

        if resume:
            self._resume_checkpoint(resume)

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.epochs + 1):
            # RUN TRAIN (AND VAL)
            results = self._train_epoch(epoch)
            if self.do_validation and epoch % self.config["trainer"]["val_per_epochs"] == 0:
                results = self._valid_epoch(epoch)

                # LOGGING INFO
                self.logger.info(f"\n         ## Info for epoch {epoch} ## ")
                for k, v in results.items():
                    self.logger.info(f"         {str(k):15s}: {v}")

            if self.train_logger is not None:
                log = {"epoch": epoch, **results}
                self.train_logger.add_entry(log)

            # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
            if self.mnt_mode != "off" and epoch % self.config["trainer"]["val_per_epochs"] == 0:
                try:
                    if self.mnt_mode == "min":
                        self.improved = log[self.mnt_metric] < self.mnt_best
                    else:
                        self.improved = log[self.mnt_metric] > self.mnt_best
                except KeyError:
                    self.logger.warning(
                        f"The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops."
                    )
                    break

                if self.improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stoping:
                    self.logger.info(
                        f"\nPerformance didn't improve for {self.early_stoping} epochs"
                    )
                    self.logger.warning("Training Stoped")
                    break

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)

    def _save_checkpoint(self, epoch, save_best=False) -> None:
        state = {
            "arch": type(self.model).__name__,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = os.path.join(self.checkpoint_dir, f"checkpoint-epoch{epoch}.pth")
        self.logger.info(f"\nSaving a checkpoint: {filename} ...")
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f"best_model.pth")
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")

    def _resume_checkpoint(self, resume_path: str) -> None:
        self.logger.info(f"Loading checkpoint : {resume_path}")
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]
        self.not_improved_count = 0

        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                {"Warning! Current model is not the same as the one in the checkpoint"}
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        if checkpoint["config"]["optimizer"]["_target_"] != self.config["optimizer"]["_target_"]:
            self.logger.warning(
                {"Warning! Current optimizer is not the same as the one in the checkpoint"}
            )
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(f"Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded")

    def _train_epoch(self, epoch: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _valid_epoch(self, epoch) -> dict[str, np.ndarray]:
        raise NotImplementedError
