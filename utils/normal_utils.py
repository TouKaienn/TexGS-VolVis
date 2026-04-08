
import torch
import numpy as np
from torch.nn import functional as F
from compute_normal_cuda import compute_normal_cuda

def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.zeros((batch_size, 3,3)).cuda()
    
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]

    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)

    return R

def scale_to_mat(scales):
    """Create scaling matrices."""
    N = scales.size(0)
    scale_mat = torch.eye(3).unsqueeze(0).repeat(N, 1, 1).to(scales.device)
    scale_mat[:, 0, 0] = scales[:, 0]
    scale_mat[:, 1, 1] = scales[:, 1]
    scale_mat[:, 2, 2] = 1.0  # No scaling in z-direction
    return scale_mat

def compute_normal_world_space(quaternions, scales, use_cuda=True):
    """Compute normal vectors from quaternions and scaling factors.
    Exactly the same implementation in rasterizer."""
    if use_cuda:
        return compute_normal_cuda(quaternions, scales)
    else:
        R = quat_to_rot(quaternions)
        S = scale_to_mat(scales)
        RS = torch.bmm(R, S)
        tn = RS[:, :,2]
        normal_vectors = tn / (tn.norm(dim=1, keepdim=True)+0.000001)
        return normal_vectors


    