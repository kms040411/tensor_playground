import sys
import torch
import program

def load(filename: str) -> dict:
    ckpt = torch.load(filename)
    return ckpt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [model_weight]")
        exit(0)
        
    model = load(sys.argv[1])
    program.program(model)