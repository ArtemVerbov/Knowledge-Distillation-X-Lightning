# Image classification template 

```
All experiments are tracked with clearml: https://clear.ml/docs/latest/docs/

Environment created with poetry: https://python-poetry.org/docs/

To install dependeces: poetry install
To run training: poetry run python3.11 -m src.train   
```

1. Setup ClearML: clearml-init

2. Migrate dataset to ClearML: make migrate_dataset

## Result example:
https://app.clear.ml/projects/27f17385bf22464fa816703a1659f67f/experiments/7ba229d3112b4910a78032df3faa4a8f/output/execution

## Test data confusion matrix

![alt text](https://github.com/ArtemVerbov/ImageClassification/blob/main/media/confusion_matrix.png?raw=true)
