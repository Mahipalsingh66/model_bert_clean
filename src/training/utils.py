import torch
import numpy as np

def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)
