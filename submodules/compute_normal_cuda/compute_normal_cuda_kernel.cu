#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void compute_normal_kernel(
    const float* __restrict__ q,     // [N, 4]
    const float* __restrict__ s,     // [N, 3]
    float* __restrict__ out,         // [N, 3]
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Read quaternion
    float qr = q[idx * 4 + 0];
    float qi = q[idx * 4 + 1];
    float qj = q[idx * 4 + 2];
    float qk = q[idx * 4 + 3];

    // Normalize quaternion
    float norm = sqrtf(qr*qr + qi*qi + qj*qj + qk*qk) + 1e-6f;
    qr /= norm; qi /= norm; qj /= norm; qk /= norm;

    // Rotation matrix R * S * [0, 0, 1]^T = RS[:,2]
    float sx = s[idx * 3 + 0];
    float sy = s[idx * 3 + 1];
    float sz = 1.0f;  // no scaling in z

    float x = 2 * (qi * qk + qr * qj) * sz;
    float y = 2 * (qj * qk - qi * qr) * sz;
    float z = (1 - 2 * (qi * qi + qj * qj)) * sz;

    float len = sqrtf(x * x + y * y + z * z) + 1e-6f;
    out[idx * 3 + 0] = x / len;
    out[idx * 3 + 1] = y / len;
    out[idx * 3 + 2] = z / len;
}

torch::Tensor compute_normal_cuda(torch::Tensor quats, torch::Tensor scales) {
    const int N = quats.size(0);
    auto output = torch::zeros({N, 3}, quats.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    compute_normal_kernel<<<blocks, threads>>>(
        quats.data_ptr<float>(),
        scales.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_normal_cuda", &compute_normal_cuda, "Compute Normal from Quat+Scale (CUDA)");
}
