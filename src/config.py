from dataclasses import field
from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra='forbid', validate_assignment=True))
class DataConfig:
    dataset_name: str = 'classification_data'
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    data_split: Tuple[float, ...] = (0.8, 0.2)
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True

    @model_validator(mode='after')
    def splits_add_up_to_one(self) -> 'DataConfig':
        epsilon = 1e-5
        total = sum(self.data_split)
        assert (abs(total - 1) < epsilon), f'Split should add up to 1, got {total}'
        return self


@dataclass(config=ConfigDict(extra='forbid', validate_assignment=True))
class TrainerConfig:
    min_epochs: int = 7  # prevents early stopping
    max_epochs: int = 20

    # perform a validation loop every N training epochs
    check_val_every_n_epoch: int = 3

    log_every_n_steps: int = 50

    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[Literal['norm', 'value']] = None

    # set True to ensure deterministic results
    # makes training slower but gives more reproducibility than just setting seeds
    deterministic: bool = False

    fast_dev_run: bool = False
    default_root_dir: Optional[Path] = None

    detect_anomaly: bool = False
    accelerator: str = 'auto'


@dataclass(config=ConfigDict(extra='forbid', validate_assignment=True))
class ProjectConfig:
    project_name: str = 'image_classification'
    experiment_name: str = 'hydra_config'
    track_in_clearml: bool = True


@dataclass(config=ConfigDict(extra='forbid', validate_assignment=True))
class ModelConfig:
    model: str
    optimizer_frequency: int = 3
    interval: str = 'epoch'
    monitor: str = 'mean_valid_loss'


@dataclass(config=ConfigDict(extra='forbid', validate_assignment=True))
class ExperimentConfig:
    model_config: ModelConfig = field(default=ModelConfig)
    data_config: DataConfig = field(default=DataConfig)
    trainer_config: TrainerConfig = field(default=TrainerConfig)
    project_config: ProjectConfig = field(default=ProjectConfig)

    @model_validator(mode='after')
    def scheduler_monitor_check(self) -> 'ExperimentConfig':
        if 'valid' in self.model_config.monitor:
            assert (self.model_config.optimizer_frequency % self.trainer_config.check_val_every_n_epoch == 0
                    ), """If "monitor" references validation metric/loss "model_config.optimizer_frequency" parameter
             should be set to a multiple of "trainer_config.check_val_every_n_epoch."""

        if 'mean' in self.model_config.monitor:
            assert (self.model_config.interval == 'epoch'
                    ), """If "monitor" references any "mean" metric/loss "model_config.interval"
                    should be set to "epoch"""
        return self
