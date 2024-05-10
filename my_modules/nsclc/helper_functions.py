import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np


# Function to load TIFFs
def load_tiff(load_fns_and_img_paths):
    X = transforms.ToTensor()(Image.open(load_fns_and_img_paths[1])).float()
    return X


# Function to load ASCs
def load_asc(load_fns_and_img_paths):
    x = torch.from_numpy(np.genfromtxt(load_fns_and_img_paths[1])).float()
    return x


# Function to simplify getting mean lifetime from raw parameters
def load_weighted_average(load_fns_and_img_paths):
    x = torch.empty(0, dtype=torch.float32)
    for fp in load_fns_and_img_paths[1]:
        x = torch.cat((x, torch.unsqueeze(fp[0](fp), dim=0)), dim=0)
    x = (x[0] * x[1] + x[2] * x[3]) / (x[0] + x[2])
    x[x < 0] = 0
    return x


# Function to simplify getting mean lifetime from raw parameters
def load_bound_fraction(load_fns_and_img_paths):
    x = torch.Tensor([])
    for fp in load_fns_and_img_paths[1]:
        x = torch.cat((x, torch.unsqueeze(fp[0](fp), dim=0)), dim=0)
    X = x[1] / (x[0] + x[1])
    X[X < 0] = 0
    X[X > 1] = 1
    return X


def convert_mp_to_torch(mp_array, shape):
    np_array = np.ctypeslib.as_array(mp_array.get_obj())
    np_array = np.reshape(np_array, shape)
    torch_array = torch.from_numpy(np_array)
    return torch_array
