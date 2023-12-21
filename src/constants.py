import os
from pathlib import Path

_DEFAULT_PROJECT_PATH = Path(__file__).resolve().parents[1]

PROJECT_ROOT = Path(os.getenv('PROJ_ROOT', _DEFAULT_PROJECT_PATH))

CONFIG_PATH = os.getenv('TRAIN_CFG_PATH', PROJECT_ROOT / 'configs')
