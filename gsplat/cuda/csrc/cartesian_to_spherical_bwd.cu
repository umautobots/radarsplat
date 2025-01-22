#include "bindings.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Cartesian to Spherical Backward Pass
 ****************************************************************************/

template <typename T>
__global__ void cartesian_to_spherical_bwd_kernel(
    const uint32_t N,
    const T *__restrict__ means,       // [N, 3]
    const T *__restrict__ covars,      // [N, 6]
    const T *__restrict__ sph_means,   // [N, 3]
    const T *__restrict__ sph_covars,  // [N, 6]
    const T *__restrict__ grad_sph_means, // [N, 3]
    const T *__restrict__ grad_sph_covars, // [N, 6]
    T *__restrict__ grad_means,        // [N, 3]
    T *__restrict__ grad_covars        // [N, 6]
) {
    // using OpT = typename OpType<T>::type;

    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    // Shift pointers to the current Gaussian
    const T *mean = means + idx * 3;
    const T *covar = covars + idx * 6;
    const T *grad_sph_mean = grad_sph_means + idx * 3;
    const T *grad_sph_covar = grad_sph_covars + idx * 6;

    T x = mean[0];
    T y = mean[1];
    T z = mean[2];

    // Compute Jacobian and inverse Jacobian for backward pass
    T r = sph_means[idx * 3];
    T theta = sph_means[idx * 3 + 1];
    T phi = sph_means[idx * 3 + 2];

    T r2 = r * r;
    T xy2 = x * x + y * y;
    T r_xy = sqrt(xy2);

    // Compute Jacobian
    T J[3][3];

    // d(r)/d(x, y, z)
    J[0][0] = x / r;
    J[0][1] = y / r;
    J[0][2] = z / r;

    // d(theta)/d(x, y, z)
    J[1][0] = -y / max(xy2, T(1e-7));
    J[1][1] = x / max(xy2, T(1e-7));
    J[1][2] = 0;

    // d(phi)/d(x, y, z)
    J[2][0] = -x * z / (r2 * max(r_xy, T(1e-7)));
    J[2][1] = -y * z / (r2 * max(r_xy, T(1e-7)));
    J[2][2] = r_xy / r2;

    // Compute inverse Jacobian
    T J_inv[3][3];

    // d(x, y, z)/d(r, theta, phi)
    J_inv[0][0] = sin(phi) * cos(theta);
    J_inv[0][1] = sin(phi) * sin(theta);
    J_inv[0][2] = cos(phi);

    J_inv[1][0] = cos(phi) * cos(theta);
    J_inv[1][1] = cos(phi) * sin(theta);
    J_inv[1][2] = -sin(phi);

    J_inv[2][0] = -sin(theta);
    J_inv[2][1] = cos(theta);
    J_inv[2][2] = 0;

    // Gradient for means
    T grad_mean[3] = {0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            grad_mean[i] += J_inv[i][j] * grad_sph_mean[j];
        }
    }

    T *grad_mean_out = grad_means + idx * 3;
    grad_mean_out[0] = grad_mean[0];
    grad_mean_out[1] = grad_mean[1];
    grad_mean_out[2] = grad_mean[2];

    // Gradient for covariances
    T grad_covar_matrix[3][3] = {0};
    T covar_matrix[3][3] = {
        {covar[0], covar[1], covar[2]},
        {covar[1], covar[3], covar[4]},
        {covar[2], covar[4], covar[5]}
    };

    T sph_grad_covar_matrix[3][3] = {
        {grad_sph_covar[0], grad_sph_covar[1], grad_sph_covar[2]},
        {grad_sph_covar[1], grad_sph_covar[3], grad_sph_covar[4]},
        {grad_sph_covar[2], grad_sph_covar[4], grad_sph_covar[5]}
    };

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                grad_covar_matrix[i][j] += J[k][i] * sph_grad_covar_matrix[k][j] * J[j][k];
            }
        }
    }

    T *grad_covar_ptr = grad_covars + idx * 6;
    grad_covar_ptr[0] = grad_covar_matrix[0][0];
    grad_covar_ptr[1] = grad_covar_matrix[0][1];
    grad_covar_ptr[2] = grad_covar_matrix[0][2];
    grad_covar_ptr[3] = grad_covar_matrix[1][1];
    grad_covar_ptr[4] = grad_covar_matrix[1][2];
    grad_covar_ptr[5] = grad_covar_matrix[2][2];
}

std::tuple<torch::Tensor, torch::Tensor> cartesian_to_spherical_bwd_tensor(
    const torch::Tensor &means,         // [N, 3]
    const torch::Tensor &covars,        // [N, 6]
    const torch::Tensor &sph_means,     // [N, 3]
    const torch::Tensor &sph_covars,    // [N, 6]
    const torch::Tensor &grad_sph_means,// [N, 3]
    const torch::Tensor &grad_sph_covars// [N, 6]
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(covars);
    GSPLAT_CHECK_INPUT(sph_means);
    GSPLAT_CHECK_INPUT(sph_covars);
    GSPLAT_CHECK_INPUT(grad_sph_means);
    GSPLAT_CHECK_INPUT(grad_sph_covars);

    uint32_t N = means.size(0);

    torch::Tensor grad_means = torch::empty_like(means);
    torch::Tensor grad_covars = torch::empty_like(covars);

    if (N > 0) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            means.scalar_type(), "cartesian_to_spherical_bwd", [&]() {
                cartesian_to_spherical_bwd_kernel<<<
                    (N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.data_ptr<scalar_t>(),
                    sph_means.data_ptr<scalar_t>(),
                    sph_covars.data_ptr<scalar_t>(),
                    grad_sph_means.data_ptr<scalar_t>(),
                    grad_sph_covars.data_ptr<scalar_t>(),
                    grad_means.data_ptr<scalar_t>(),
                    grad_covars.data_ptr<scalar_t>());
            });
    }

    return std::make_tuple(grad_means, grad_covars);
}

} // namespace gsplat
