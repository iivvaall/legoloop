from pathlib import Path

import torch


def path_join(path, *args):
    res = Path(path)
    for arg in args:
        res /= arg
    return res


def to_device(data, device):
    if isinstance(data, dict):
        return {
            key: to_device(value, device)
            for key, value in data.items()
        }
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data