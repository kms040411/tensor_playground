import numpy as np
import torch
from tensor_viewer.ops import print_dist
import random

DIM_M = 1024
DIM_N = 3072
NORM_MEAN = 0.0
NORM_STD = 1.0

OUTLIER_NUM = int(DIM_M * DIM_N * 0.005)

# Generate Original Tensor
tensor = torch.empty(DIM_M, DIM_N, dtype=torch.float)
torch.nn.init.normal_(tensor, mean=NORM_MEAN, std=NORM_STD)
#print_dist(tensor, None)

outlier_list = list()
for i in range(OUTLIER_NUM):
    outlier_list.append((random.randint(0, DIM_M-1), random.randint(0, DIM_N-1)))
for outlier in outlier_list:
    tensor[outlier[0], outlier[1]] = random.uniform(-20, 20)

# Proposed Solution
SCALING_FACTOR_PROPOSED = 1200
a_vec = torch.max(tensor, dim=1)[0] # size of DIM_M
b_vec = torch.max(tensor, dim=0)[0] # size of DIM_N

a_vec = a_vec.unsqueeze(dim=1)
b_vec = b_vec.unsqueeze(dim=0)

tensor_proposed = tensor * SCALING_FACTOR_PROPOSED
tensor_proposed = (tensor_proposed / a_vec) / b_vec
tensor_proposed = tensor_proposed.to(dtype=torch.int8)

# Naive Solution
SCALING_FACTOR_NAIVE = 50

naive_tensor = tensor * SCALING_FACTOR_NAIVE
naive_tensor = naive_tensor.to(dtype=torch.int8)

# Get error of proposed solution
restore_proposed = tensor_proposed.to(dtype=torch.float16)
restore_proposed = restore_proposed / SCALING_FACTOR_PROPOSED
restore_proposed = ((restore_proposed * b_vec) * a_vec)

# Get error of naive solution
restore_naive = naive_tensor.to(dtype=torch.float16)
restore_naive = restore_naive / SCALING_FACTOR_NAIVE

naive_error = torch.nn.functional.mse_loss(tensor, restore_naive)
print(f"naive_error: {naive_error}")
proposed_error = torch.nn.functional.mse_loss(tensor, restore_proposed)
print(f"proposed_error: {proposed_error}")

# print_dist(tensor, None)
# print_dist(restore_naive, None)
# print_dist(restore_proposed, None)