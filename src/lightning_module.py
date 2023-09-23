from typing import Dict, List

import timm
import torch
import torch.nn.functional as func
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import ConfusionMatrix, MeanMetric

from src.metrics import get_metrics


class ClassificationLightningModule(LightningModule):
    def __init__(self, cfg, class_to_idx: Dict[str, int]):
        super().__init__()
        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()

        metrics = get_metrics(
            num_classes=len(class_to_idx),
            num_labels=len(class_to_idx),
            task='multiclass',
            average='macro',
        )
        self.frequency = cfg.trainer_config.check_val_every_n_epoch
        self.lr = cfg.learning_rate
        self.patient = cfg.scheduler_patient
        self.model = timm.create_model(cfg.model, pretrained=True, num_classes=len(class_to_idx))

        self._valid_metrics = metrics.clone(prefix='valid_')
        self._test_metrics = metrics.clone(prefix='test_')
        self.save_hyperparameters()

    def forward(self, images: Tensor) -> Tensor:
        # return self.model(images.float())
        return self.model(images)

    def training_step(self, batch: List[Tensor]):  # noqa: WPS210
        images, targets = batch
        logits = self.forward(images)
        loss = func.cross_entropy(logits, targets)
        self._train_loss(loss)
        self.log('step_loss', loss, on_step=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': logits, 'target': targets}

    def validation_step(self, batch: List[Tensor], batch_index: int):  # noqa: WPS210
        images, targets = batch
        logits = self.forward(images)
        loss = func.cross_entropy(logits, targets)
        self._valid_loss(loss)

        self._valid_metrics(logits, targets)
        self.log('val_loss', loss, on_step=False, prog_bar=False, logger=True)

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        images, targets = batch
        logits = self(images)

        self._test_metrics(logits, targets)
        return logits

    def on_train_epoch_end(self) -> None:
        self.log('mean_train_loss', self._train_loss, on_step=False, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.log('mean_valid_loss', self._valid_loss, on_step=False, prog_bar=True, on_epoch=True)

        self.log_dict(self._valid_metrics, prog_bar=True, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics, prog_bar=True, on_epoch=True)

    def configure_optimizers(self) -> dict:
        # TODO: parametrize optimizer and lr scheduler.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  # noqa: WPS432 will be parametrized
        scheduler = ReduceLROnPlateau(optimizer, patience=self.patient, verbose=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': self.frequency,
                'monitor': 'mean_valid_loss',
            },
        }
