import os
from pathlib import Path
from typing import Any, Sequence, Union

import yaml
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange,
    GridSearch,
    HyperParameterOptimizer,
)

from src.constants import PROJECT_ROOT


def optimize() -> Sequence[Any]:
    Task.init(project_name='test', task_name='optimizer', task_type=Task.TaskTypes.optimizer)

    optimizer = HyperParameterOptimizer(
        base_task_id='39afd29fdc844a9aa15fa2b8a2696448',
        hyper_parameters=[DiscreteParameterRange('General/data_config/batch_size', [16, 32, 64])],
        objective_metric_title='step_loss',
        objective_metric_series='step_loss',
        objective_metric_sign='min',
        optimizer_class=GridSearch,
    )

    optimizer.start_locally()
    optimizer.wait()
    top_exp = optimizer.get_top_experiments_details(top_k=1)
    optimizer.stop()

    return top_exp


def dict_to_yaml(params: Sequence, path: Union[str, Path]):
    hyper_parameters = params[0]['hyper_parameters']
    key = list(hyper_parameters)[0]
    hyper_parameters[key.split('/')[-1]] = hyper_parameters.pop(key)

    with open(path, 'w') as out_file:
        yaml.safe_dump(hyper_parameters, out_file, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    exp = optimize()
    optimization_path = os.getenv('TRAIN_CFG_PATH', PROJECT_ROOT / 'configs' / 'optimization' / 'opt.yaml')
    dict_to_yaml(exp, optimization_path)
