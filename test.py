from quant_test import *
import sys

DIM_M = 1024
DIM_N = 3072
NORM_MEAN = 0.0
NORM_STD = 1.0

OUTLIER_NUM = int(DIM_M * DIM_N * 0.001)
MAX_SWEEP_NUM = 10000

#SCALING_FACTOR_PROPOSED = 1200
#SCALING_FACTOR_NAIVE = 50

model = load("/data/hyperaccel/model/megatron-gpt2-345m/ckpt/pytorch_model.bin")
tensor = model["lm_head.weight"] #tensor_gen(DIM_M, DIM_N, NORM_MEAN, NORM_STD, OUTLIER_NUM)

proposed_results_x = list()
proposed_results_y = list()
for scaling_factor in range(300, 1000, 50):
    tensor_proposed, a_vec, b_vec = proposed_quant(tensor, scaling_factor)
    restore_proposed = proposed_restore(tensor_proposed, a_vec, b_vec, scaling_factor)
    proposed_error = get_mse(tensor, restore_proposed)

    proposed_results_x.append(scaling_factor)
    proposed_results_y.append(proposed_error)

naive_results_x = list()
naive_results_y = list()
for scaling_factor in range(0, 20, 1):
    tensor_naive = naive_quant(tensor, scaling_factor)
    restore_naive = naive_restore(tensor_naive, scaling_factor)
    naive_error = get_mse(tensor, restore_naive)

    naive_results_x.append(scaling_factor)
    naive_results_y.append(naive_error)

# scaling_factor_proposed = 10.0
# current_jump = 10.0
# current_err = sys.float_info.max
# counter = 0
# while True:
#     tensor_proposed, a_vec, b_vec = proposed_quant(tensor, scaling_factor_proposed)
#     restore_proposed = proposed_restore(tensor_proposed, a_vec, b_vec, scaling_factor_proposed)
#     proposed_error = get_mse(tensor, restore_proposed)

#     if proposed_error < current_err:
#         scaling_factor_proposed += current_jump
#         current_jump = current_jump * 1.5
#         current_err = proposed_error
#     elif proposed_error == current_err:
#         print(f"current count: {counter}, current jump: {current_jump}, current err: {current_err}, current scaling factor: {scaling_factor_proposed}")
#         break
#     else:
#         current_jump = current_jump * -0.5
#         scaling_factor_proposed += current_jump
#         current_err = proposed_error

#     if counter % 100 == 0:
#         print(f"current count: {counter}, current jump: {current_jump}, current err: {current_err}, current scaling factor: {scaling_factor_proposed}")

#     counter += 1
#     if counter > MAX_SWEEP_NUM: break
    
# print(f"Final proposed error: {current_err}, Final scaling factor: {scaling_factor_proposed}")
# print(f"")

# scaling_factor_naive = 10.0
# current_jump = 1.0
# current_err = sys.float_info.max
# counter = 0
# while True:
#     tensor_naive = naive_quant(tensor, scaling_factor_naive)
#     restore_naive = naive_restore(tensor_naive, scaling_factor_naive)
#     naive_error = get_mse(tensor, restore_naive)

#     if naive_error < current_err:
#         scaling_factor_naive += current_jump
#         current_jump = current_jump * 1.5
#         current_err = naive_error 
#     elif naive_error == current_err:
#         print(f"current count: {counter}, current jump: {current_jump}, current err: {current_err}, current scaling factor: {scaling_factor_naive}")
#         break
#     else:
#         current_jump = current_jump * -0.5
#         scaling_factor_naive += current_jump
#         current_err = naive_error

#     if counter % 100 == 0:
#         print(f"current count: {counter}, current jump: {current_jump}, current err: {current_err}, current scaling factor: {scaling_factor_naive}")

#     counter += 1
#     if counter > MAX_SWEEP_NUM: break

# print(f"Final naive error: {current_err}, Final scaling factor: {scaling_factor_naive}")
