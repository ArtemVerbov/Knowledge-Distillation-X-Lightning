import os

import lightning
from clearml import Task
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor

from src.callbacks.confusion_matrix_callback import ConfusionMatrixLogging
from src.callbacks.debug import VisualizeBatch
from src.config import ExperimentConfig
from src.constants import PROJECT_ROOT
from src.datamodule import ClassificationDataModule
from src.lightning_module import ClassificationLightningModule


def train(cfgs: ExperimentConfig):
    lightning.seed_everything(0)
    Task.force_requirements_env_freeze()
    task = Task.init(
        project_name=cfgs.project_config.project_name,
        task_name=cfgs.project_config.experiment_name,
        # If `output_uri=True` uses default ClearML output URI,
        # can use string value to specify custom storage URI like S3.
        output_uri=True,
    )
    task.connect(cfgs.model_dump())

    datamodule = ClassificationDataModule(cfg=cfgs.data_config)
    task.connect_configuration(datamodule.transforms.get_train_transforms(), name='transformations')

    model = ClassificationLightningModule(cfgs, class_to_idx=datamodule.class_to_idx)

    lr_logger = LearningRateMonitor(logging_interval='epoch')
    visualize = VisualizeBatch(every_n_epochs=1)
    matrix_logger = ConfusionMatrixLogging(datamodule.class_to_idx)

    trainer = Trainer(
        **dict(cfgs.trainer_config),
        callbacks=[
            lr_logger,
            visualize,
            matrix_logger,
        ],
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    cfgs_path = os.getenv('TRAIN_CFG_PATH', PROJECT_ROOT / 'configs' / 'conf.yaml')
    train(cfgs=ExperimentConfig.from_yaml(cfgs_path))
