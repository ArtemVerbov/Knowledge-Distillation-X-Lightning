import torch
import torch.nn as nn
from torch import Tensor


def knowledge_distillation_loss(
    teacher_logits: Tensor,
    student_logits: Tensor,
    label_loss: Tensor,
    temperature: float,
    soft_target_loss_weight: float,
    ce_loss_weight: float,
):

    # Soften the student logits by applying softmax first and log() second
    soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = nn.functional.log_softmax(student_logits / temperature, dim=-1)

    # Calculate the soft targets loss.
    # Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
    soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (temperature ** 2)
    return soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
