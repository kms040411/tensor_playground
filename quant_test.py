import numpy as np
import torch
import random

def tensor_gen(dim_m, dim_n, mean, std, outlier_num):
    tensor = torch.empty(dim_m, dim_n, dtype=torch.float)
    torch.nn.init.normal_(tensor, mean, std)

    outlier_list = list()
    for i in range(outlier_num):
        outlier_list.append((random.randint(0, dim_m-1), random.randint(0, dim_n-1)))
    for outlier in outlier_list:
        tensor[outlier[0], outlier[1]] = random.uniform(-20, 20)
    return tensor

def proposed_quant(tensor, scaling_factor):
    a_vec = torch.max(tensor, dim=1)[0] # size of DIM_M
    b_vec = torch.max(tensor, dim=0)[0] # size of DIM_N

    a_vec = a_vec.unsqueeze(dim=1)
    b_vec = b_vec.unsqueeze(dim=0)

    tensor_proposed = tensor * scaling_factor
    tensor_proposed = (tensor_proposed / a_vec) / b_vec
    tensor_proposed = tensor_proposed.to(dtype=torch.int8)
    return tensor_proposed, a_vec, b_vec

def naive_quant(tensor, scaling_factor):
    naive_tensor = tensor * scaling_factor
    naive_tensor = naive_tensor.to(dtype=torch.int8)
    return naive_tensor

def proposed_restore(quant_tensor, a_vec, b_vec, scaling_factor):
    restore_proposed = quant_tensor.to(dtype=torch.float16)
    restore_proposed = restore_proposed / scaling_factor
    restore_proposed = ((restore_proposed * b_vec) * a_vec)
    return restore_proposed

def naive_restore(quant_tensor, scaling_factor):
    restore_naive = quant_tensor.to(dtype=torch.float16)
    restore_naive = restore_naive / scaling_factor
    return restore_naive

def get_mse(original, quant):
    return torch.nn.functional.mse_loss(original, quant)