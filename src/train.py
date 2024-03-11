from dataclasses import asdict
from typing import TYPE_CHECKING

import hydra
import lightning
from clearml import Task
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.callbacks.confusion_matrix_callback import ConfusionMatrixLogging
from src.callbacks.debug import VisualizeBatch
from src.config import ExperimentConfig
from src.constants import CONFIG_PATH
from src.datamodule import ClassificationDataModule

if TYPE_CHECKING:
    from lightning import LightningModule
    from omegaconf import DictConfig


# noinspection PyDataclass
@hydra.main(config_path=str(CONFIG_PATH), config_name='conf', version_base='1.2')
def train(cfg: 'DictConfig') -> None:  # noqa: WPS210

    experiment_config: ExperimentConfig = hydra.utils.instantiate(cfg.experiment_config)
    module: 'LightningModule' = hydra.utils.instantiate(cfg.lightning_module.module)

    lightning.seed_everything(0)
    datamodule = ClassificationDataModule(cfg=experiment_config.data_config)

    if experiment_config.project_config.track_in_clearml:
        Task.force_requirements_env_freeze()
        task = Task.init(
            project_name=experiment_config.project_config.project_name,
            task_name=experiment_config.project_config.experiment_name,
            # If `output_uri=True` uses default ClearML output URI,
            # can use string value to specify custom storage URI like S3.
            output_uri=True,
        )
        # Stores yaml config as a dictionary in clearml
        task.connect(asdict(experiment_config))
        task.connect_configuration(datamodule.transforms.get_train_transforms(), name='transformations')

    model = module(
        experiment_config.model_config,
        class_to_idx=datamodule.class_to_idx,
    )

    lr_logger = LearningRateMonitor(logging_interval='epoch')
    visualize = VisualizeBatch(every_n_epochs=5)
    matrix_logger = ConfusionMatrixLogging(datamodule.class_to_idx)
    early_stopping = EarlyStopping(monitor='mean_valid_loss', patience=5)
    check_points = ModelCheckpoint(monitor='valid_f1', mode='max', verbose=True, save_top_k=1)

    trainer = Trainer(
        **asdict(experiment_config.trainer_config),
        callbacks=[
            lr_logger,
            visualize,
            matrix_logger,
            early_stopping,
            check_points,
        ],
        # overfit_batches=10,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path='best')


if __name__ == '__main__':
    train()
