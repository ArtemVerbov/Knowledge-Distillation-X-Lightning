from typing import TYPE_CHECKING, Dict, List, Optional

import timm
import torch.nn.functional as func
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric

from src.config import ModelConfig
from src.metrics import get_metrics

if TYPE_CHECKING:
    from torch.optim import Adam


class ClassificationLightningModule(LightningModule):  # noqa: WPS214
    def __init__(
        self,
        model_cfg: ModelConfig,
        class_to_idx: Dict[str, int],
        optimizer: 'Adam',
        scheduler: Optional['ReduceLROnPlateau'] = None,
    ):
        super().__init__()
        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()
        self.optimizer = optimizer
        self.scheduler = scheduler

        metrics = get_metrics(
            num_classes=len(class_to_idx),
            num_labels=len(class_to_idx),
            task='multiclass',
            average='macro',
        )
        self.frequency = model_cfg.optimizer_frequency
        self.model_cfg = model_cfg
        self.model = timm.create_model(self.model_cfg.model, pretrained=True, num_classes=len(class_to_idx))

        self._valid_metrics = metrics.clone(prefix='valid_')
        self._test_metrics = metrics.clone(prefix='test_')
        self.save_hyperparameters()

    def forward(self, images: Tensor) -> Tensor:
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

    # noinspection PyCallingNonCallable
    def configure_optimizers(self) -> Dict:
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler:
            scheduler = self.scheduler(optimizer)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': self.model_cfg.interval,
                    'frequency': self.model_cfg.optimizer_frequency,
                    'monitor': self.model_cfg.monitor,
                },
            }
        return {'optimizer': optimizer}
