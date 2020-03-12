# State modifier
import torch
import numpy as np
import cv2

import os
from collections import OrderedDict


def renameKeys(source: torch.nn.Module.state_dict, target: str) -> torch.nn.Module.state_dict:
    new_key = list(source)
    state = torch.load(target)
    new_state = OrderedDict()

    i = 0
    for key, value in state.items():
        new_state[new_key[i]] = value
        i += 1

    return new_state
