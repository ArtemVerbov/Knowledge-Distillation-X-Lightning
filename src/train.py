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
    # logger = TensorBoardLogger('tb_logs1', name='model')
    # tb_log_dir = Path(os.path.join("tb_logs1", 'profiler0'))
    # profiler = PyTorchProfiler(torch.profiler.tensorboard_trace_handler(str(tb_log_dir)),
    #                            schedule=torch.profiler.schedule(active=1, warmup=0, wait=0))
    # trainer = Trainer(**dict(cfgs.trainer_config), callbacks=[], profiler=profiler, logger=logger)
    # lr_finder = LearningRateFinder()
    lr_logger = LearningRateMonitor(logging_interval='epoch')
    visualize = VisualizeBatch(every_n_epochs=1)
    matrix_logger = ConfusionMatrixLogging(datamodule.class_to_idx)

    trainer = Trainer(
        **dict(cfgs.trainer_config),
        callbacks=[
            lr_logger,
            # lr_finder,
            visualize,
            matrix_logger,
        ]
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    cfgs_path = os.getenv('TRAIN_CFG_PATH', PROJECT_ROOT / 'configs' / 'conf.yaml')
    train(cfgs=ExperimentConfig.from_yaml(cfgs_path))
    # datamodule = ClassificationDataModule(cfg=ExperimentConfig.from_yaml(cfgs_path).data_config)
    # datamodule.prepare_data()
    # datamodule.setup('fit')
    # train_dataloader = datamodule.train_dataloader()
    # image = next(iter(train_dataloader))[0][:5]
    # for i in image:
    #     plt.imshow(i.permute(1,2,0).cpu())
    #     plt.show()
