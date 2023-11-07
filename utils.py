import json
import numpy as np
import matplotlib.pyplot as plt
import torch


def save_obj(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def get_obj(path):
    try:
        with open(path, 'r') as f:
            obj = json.load(f)
            print(obj)
            return obj
    except FileNotFoundError:
        return {}


def visualize_tensor(pixel_tensor: torch.Tensor):
    pixel_tensor = np.array(pixel_tensor.permute(1, 2, 0))
    plt.imshow(pixel_tensor)
    plt.show()
