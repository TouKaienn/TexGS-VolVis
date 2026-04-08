import torch
import compute_normal_cuda

def compute_normal_world_space_cuda(quaternions: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    assert quaternions.is_cuda and scales.is_cuda
    assert quaternions.shape[1] == 4 and scales.shape[1] == 3
    return compute_normal_cuda.compute_normal_cuda(quaternions.contiguous(), scales.contiguous())
