import torch
import progressbar

def print_tensor(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    return tensor_a, tensor_b

def clear(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    return None, None

def swap(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    return tensor_b, tensor_a

def print_dist(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    if tensor_a is None:
        print("This operation works with tensor_a")
        return tensor_a, tensor_b
    
    max_elem = torch.max(tensor_a).item()
    min_elem = torch.min(tensor_a).item()
    print(f"Max: {max_elem}")
    print(f"Min: {min_elem}")
    print(f"Avg: {torch.mean(tensor_a)}")
    print(f"Std: {torch.std(tensor_a)}")

    while True:
        try:
            num_bin = int(input("How many bin?: "))
            break
        except:
            continue
    
    # Draw histogram
    gap = (max_elem - min_elem) / num_bin
    bins = []
    counts = []
    for i in range(num_bin):
        bins.append(min_elem + i * gap)
        counts.append(0)
    
    flattened = tensor_a.reshape(-1).tolist()
    total_elem = len(flattened)
    print("total elements: ", total_elem)

    bar = progressbar.ProgressBar(maxval=total_elem).start()
    for idx, i in enumerate(flattened):
        bar.update(idx)
        reach_end = True
        for idx in range(num_bin - 1):
            if bins[idx] <= i < bins[idx + 1]:
                counts[idx] += 1
                reach_end = False
        if reach_end: counts[num_bin - 1] += 1
    bar.finish()
    
    for i in range(num_bin):
        print(f"{counts[i] * 1.0 / total_elem:.4f}\t", end="")
    print()
    for i in range(num_bin):
        print(f"{bins[i]:.4f}\t", end="")
    print()

    return tensor_a, tensor_b

def qr_decomposition(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    user_ok = input("This function will erase tensor_b. Are you okay to continue? [y/N]: ")
    if user_ok not in ("y", "Y"):
        return tensor_a, tensor_b
    
    user_option = input("Enter QR decomposition option [reduced, complete, r]: ")
    if user_option not in ("reduced", "complete", "r"):
        return tensor_a, tensor_b
    
    q, r = torch.linalg.qr(tensor_a.float(), mode=user_option)
    return q.half(), r.half()