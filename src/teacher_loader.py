from typing import TYPE_CHECKING, Callable, Dict

import hydra
import torch

from src.constants import CONFIG_PATH, WEIGHTS_PATH

if TYPE_CHECKING:
    from omegaconf import DictConfig


class TeacherLoader:

    def __init__(self, model_to_create: Callable, weights_name: str, num_classes: int) -> None:
        self.model = model_to_create(num_classes=num_classes)
        self.weights_path = WEIGHTS_PATH / weights_name

    def load_model(self) -> Callable:
        self.model.load_state_dict(self._update_state_dict_keys())
        return self.model

    def _update_state_dict_keys(self) -> Dict:
        if torch.cuda.is_available():
            state_dict = torch.load(self.weights_path)
        else:
            state_dict = torch.load(self.weights_path, map_location=torch.device('cpu'))
        return {key.removeprefix('model.'): value for key, value in state_dict['state_dict'].items()}


@hydra.main(config_path=str(CONFIG_PATH / 'lightning_module'), config_name='knowledge_distillation', version_base='1.2')
def train(cfg: 'DictConfig') -> None:
    model = hydra.utils.instantiate(cfg.module.teacher_model.model_to_create)
    teacher = TeacherLoader(model, weights_name=cfg.module.teacher_model.weights_name, num_classes=6)


if __name__ == '__main__':
    train()
