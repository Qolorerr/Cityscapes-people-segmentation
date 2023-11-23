import time

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from base.base_dataloader import BaseDataLoader
from base.base_model import BaseModel
from base.base_trainer import BaseTrainer
from utils import Logger
from utils.metrics import AverageMeter, eval_metrics
from utils.visualizations import Visualization


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: BaseModel,
        loss: nn.Module,
        resume: str,
        config: DictConfig,
        train_loader: BaseDataLoader,
        val_loader: BaseDataLoader | None = None,
        train_logger: Logger | None = None,
        prefetch: bool = True,
    ):
        super(Trainer, self).__init__(
            model, loss, resume, config, train_loader, val_loader, train_logger
        )

        self.wrt_mode, self.wrt_step = "train_", 0
        self.log_step = config["trainer"].get(
            "log_per_iter", int(np.sqrt(self.train_loader.batch_size))
        )
        if config["trainer"]["log_per_iter"]:
            self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        self.train_loader, self.val_loader = self.accelerator.prepare(
            self.train_loader, self.val_loader
        )

        # INITIALISATION VISUALIZATION
        self.visualization = Visualization(train_loader)

    def _log_step(self, epoch: int, batch_idx: int, loss_item) -> None:
        if batch_idx % self.log_step == 0:
            self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
            self.writer.add_scalar(f"{self.wrt_mode}/loss", loss_item, self.wrt_step)

    def _log_seg_metrics(self) -> dict[str, np.ndarray]:
        seg_metrics = self._get_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]:
            self.writer.add_scalar(f"{self.wrt_mode}/{k}", v, self.wrt_step)
        return seg_metrics

    def _print_metrics(self, tbar: tqdm, epoch: int, mode: str = "TRAIN") -> None:
        pixAcc, mIoU, _ = self._get_seg_metrics().values()

        # PRINT INFO
        message = "{} ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |"
        message = message.format(mode, epoch, self.total_loss.average, pixAcc, mIoU)
        if mode == "EVAL":
            message += " B {:.2f} D {:.2f} |".format(
                self.batch_time.average, self.data_time.average
            )
        tbar.set_description(message)

    def _train_epoch(self, epoch: int) -> dict[str, np.ndarray]:
        self.model.train()
        if self.config["arch"]["args"]["freeze_bn"]:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.freeze_bn()
            else:
                self.model.freeze_bn()
        self.wrt_mode = "train"

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (inputs, targets) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            self.lr_scheduler.step(epoch=epoch - 1)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if self.config["arch"]["type"][:3] == "PSP":
                outputs = outputs[0]
            loss = self.loss(outputs, targets, self.config["arch"]["type"], self.num_classes)

            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
            self.accelerator.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            self._log_step(epoch, batch_idx, loss.item())

            # FOR EVAL
            seg_metrics = eval_metrics(outputs, targets, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            self._print_metrics(tbar, epoch, "TRAIN")

        # METRICS TO TENSORBOARD
        seg_metrics = self._log_seg_metrics()
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f"{self.wrt_mode}/Learning_rate_{i}", opt_group["lr"], self.wrt_step
            )
            # self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {"loss": self.total_loss.average, **seg_metrics}

        # if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch: int) -> dict[str, np.ndarray]:
        if self.val_loader is None:
            self.logger.warning(
                "Not data loader was passed for the validation step, No validation is performed !"
            )
            return {}
        self.logger.info("\n###### EVALUATION ######")

        self.model.eval()
        self.wrt_mode = "val"

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tbar):
                # LOSS
                output = self.model(data)
                loss = self.loss(output, targets, self.config["arch"]["type"], self.num_classes)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, targets, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                self.visualization.add_to_visual(targets, output)

                # PRINT INFO
                self._print_metrics(tbar, epoch, "EVAL")

            # WRITING & VISUALIZING THE MASKS
            val_img = self.visualization.flush_visual()
            self.writer.add_image(
                f"{self.wrt_mode}/inputs_targets_predictions", val_img, self.wrt_step
            )

            # METRICS TO TENSORBOARD
            self.wrt_step = epoch * len(self.val_loader)
            self.writer.add_scalar(f"{self.wrt_mode}/loss", self.total_loss.average, self.wrt_step)
            seg_metrics = self._log_seg_metrics()

            log = {"val_loss": self.total_loss.average, **seg_metrics}

        return log

    def _reset_metrics(self) -> None:
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union) -> None:
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self) -> dict[str, np.ndarray | dict]:
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3))),
        }
