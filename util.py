import numpy as np
import random
import os
import torch


def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def print_config(args):
    config_path = os.path.join(args.base_path, args.dataset, "output", args.file_id, "config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            print(k, '=', v, file=f)


def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def mat_padding(inputs, length=None, padding=0):
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]
    if length is None:
        length = max([x.shape[0] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[0] = (0, length - x.shape[0])
        pad_width[1] = (0, length - x.shape[0])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)


def sequence_padding(inputs, dim=0, length=None, padding=0):
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]
    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[dim] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)
