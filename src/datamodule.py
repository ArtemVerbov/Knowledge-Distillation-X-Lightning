from pathlib import Path
from typing import Dict, Optional

import torch
from clearml import Dataset as ClearmlDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.config import DataConfig
from src.dataset import ClassificationDataset
from src.transforms import Transforms


class ClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DataConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.transforms = Transforms(*cfg.img_size)

        # Prevent hyperparameters from being stored in checkpoints.
        self.save_hyperparameters(logger=False)

        self.data_path: Optional[Path] = None
        self.initialized = False

        self.data_train: Optional[ClassificationDataset] = None
        self.data_val: Optional[ClassificationDataset] = None
        self.data_test: Optional[ClassificationDataset] = None

    @property
    def class_to_idx(self) -> Dict[str, int]:
        if not self.initialized:
            self.prepare_data()
            self.setup('test')
        return self.data_test.class_to_idx

    def prepare_data(self):
        self.data_path = Path(ClearmlDataset.get(dataset_name=self.cfg.dataset_name).get_local_copy())

    def setup(self, stage: str):
        if stage == 'fit':
            all_data = ClassificationDataset(
                str(self.data_path / 'Classification_data' / 'train'),
                transform=self.transforms.compose(stage),
            )
            length = len(all_data)
            train_length = round(length * self.cfg.data_split[0])
            val_length = round(length * self.cfg.data_split[1])
            self.data_train, self.data_val = torch.utils.data.random_split(
                all_data,
                [train_length, val_length],
            )
            self.data_val.transform = self.transforms.compose('test')
        elif stage == 'test':
            self.data_test = ClassificationDataset(
                str(self.data_path / 'Classification_data' / 'test'),
                transform=self.transforms.compose(stage),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
        )
