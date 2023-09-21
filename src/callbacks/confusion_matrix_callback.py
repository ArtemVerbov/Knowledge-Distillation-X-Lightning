from typing import Dict, List

import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torchmetrics import ConfusionMatrix


class ConfusionMatrixLogging(Callback):
    def __init__(self, class_to_idx: Dict):
        super().__init__()
        self.labels = list(class_to_idx.keys())
        self._confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=len(self.labels))
        self.predicts: List[Tensor] = []
        self.targets: List[Tensor] = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int = 0):  # WPS211
        self.predicts.append(outputs)
        self.targets.append(batch[1])

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        targets = torch.cat(self.targets, dim=0)
        predicts = torch.cat(self.predicts, dim=0)

        self._confusion_matrix(predicts, targets)
        fig, _ = self._confusion_matrix.plot(labels=self.labels)
        trainer.logger.experiment.add_figure(
            'Matrix',
            figure=fig,
            global_step=trainer.current_epoch,
        )
