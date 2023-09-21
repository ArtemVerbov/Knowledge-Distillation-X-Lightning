from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection(
        {
            'f1': F1Score(**kwargs),
            'precision': Precision(**kwargs),
            'recall': Recall(**kwargs),
            'accuracy': Accuracy(**kwargs),
        },
    )
