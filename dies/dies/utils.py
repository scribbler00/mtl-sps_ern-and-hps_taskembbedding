from collections import Iterable
import numpy as np
import pandas as pd
import random
import torch

np_int_dtypes = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.intp,
    np.uintp,
]


def listify(p):
    "Make `p` listy."
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    elif isinstance(p, pd.DataFrame):
        p = [p]

    return list(p)


def get_structure(initial_size, percental_reduce, min_value, final_outputs=1):
    ann_structure = [initial_size]
    final_outputs = listify(final_outputs)

    if 0 in final_outputs or (None in final_outputs):
        raise ValueError(
            "Invalid parameters: final_outputs should not contain 0 or None"
        )

    if percental_reduce >= 1.0:
        percental_reduce = percental_reduce / 100.0

    while True:
        new_size = int(ann_structure[-1] - ann_structure[-1] * percental_reduce)

        if new_size <= min_value:
            new_size = min_value
            ann_structure.append(new_size)
            break
        else:
            ann_structure.append(new_size)

    return ann_structure + final_outputs


def set_random_states(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
