from lightning import Callback, LightningModule, Trainer
from torchvision.utils import make_grid

from src.transforms import inv_trans


class VisualizeBatch(Callback):
    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.batch = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch = batch[0]

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.current_epoch % self.every_n_epochs == 0:
            visualizations = [inv_trans(img) for img in self.batch]
            grid = make_grid(visualizations, normalize=False)
            trainer.logger.experiment.add_image(
                'Batch preview',
                img_tensor=grid,
                global_step=trainer.current_epoch,
            )
