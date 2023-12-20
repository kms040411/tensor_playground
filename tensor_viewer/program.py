import torch
from typing import Optional
from inspect import getmembers, isfunction
import ops
import readline
POSSIBLE_OPS = list(map(lambda x: x[0], getmembers(ops, isfunction)))

def program(model: dict):
    current_tensor_a: Optional[torch.Tensor] = None
    current_tensor_b: Optional[torch.Tensor] = None
    while True:
        if current_tensor_a is None and current_tensor_b is None:
            print("")
            print("=================== List of Tensors ===================")
            print(*model.keys(), sep="\n")
            print("=======================================================")
            print("")

            user_tensor = input("Enter tensor name: ")
            if user_tensor == "": break
            elif user_tensor not in model.keys():
                print(f"There is no tensor with name {user_tensor}")
                continue
            current_tensor_a = model[user_tensor]
            continue
        
        print("")
        print("================= Possible operations: ================")
        print(*POSSIBLE_OPS, sep="\n")
        print("=======================================================")
        print("")

        user_op = input("Enter operation to apply: ")
        if user_op == "": break
        elif user_op not in POSSIBLE_OPS:
            print(f"There is no operation with name {user_op}")
            continue

        current_tensor_a, current_tensor_b = eval(f"ops.{user_op}(current_tensor_a, current_tensor_b)")

        print("")
        print("=================== Current Tensor ===================")
        if current_tensor_a is not None:
            print("= Tensor A")
            print(current_tensor_a)
            print(current_tensor_a.shape)
            print("======================================================")
        if current_tensor_b is not None:
            print("= Tensor B")
            print(current_tensor_b)
            print(current_tensor_b.shape)
            print("======================================================")
        print("")