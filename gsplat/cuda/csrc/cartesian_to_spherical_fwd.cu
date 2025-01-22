#include "bindings.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Cartesian to Spherical Forward Pass
 ****************************************************************************/

template <typename T>
__global__ void cartesian_to_spherical_fwd_kernel(
    const uint32_t N,
    const T *__restrict__ means,  // [N, 3]
    const T *__restrict__ covars, // [N, 6]
    T *__restrict__ sph_means,   // [N, 3]
    T *__restrict__ sph_covars   // [N, 6]
) {
    // using OpT = typename OpType<T>::type;

    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    // Shift pointers to the current Gaussian
    const T *mean = means + idx * 3;
    const T *covar = covars + idx * 6;

    T x = mean[0];
    T y = mean[1];
    T z = mean[2];

    // Compute spherical means
    T r = sqrt(x * x + y * y + z * z);
    T theta = atan2(y, x);
    T phi = acos(z / max(r, T(1e-7)));

    T *sph_mean = sph_means + idx * 3;
    sph_mean[0] = r;
    sph_mean[1] = theta;
    sph_mean[2] = phi;

    // Compute Jacobian
    T r2 = r * r;
    T xy2 = x * x + y * y;
    T r_xy = sqrt(xy2);

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

    // Transform covariance matrix
    T covar_matrix[3][3] = {
        {covar[0], covar[1], covar[2]},
        {covar[1], covar[3], covar[4]},
        {covar[2], covar[4], covar[5]}
    };

    // First temp_matrix = J * covar_matrix
    T sph_covar_matrix[3][3] = {0};
    T temp_matrix[3][3] = {0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                temp_matrix[i][j] += J[i][k] * covar_matrix[k][j];
            }
        }
    }

    // Now multiply temp_matrix by J^T: sph_covar_matrix = (J * covar_matrix) * J^T
    // (J^T)[k][j] = J[j][k]
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                sph_covar_matrix[i][j] += temp_matrix[i][k] * J[j][k];
            }
        }
    }

    T *sph_covar = sph_covars + idx * 6;
    sph_covar[0] = sph_covar_matrix[0][0];
    sph_covar[1] = sph_covar_matrix[0][1];
    sph_covar[2] = sph_covar_matrix[0][2];
    sph_covar[3] = sph_covar_matrix[1][1];
    sph_covar[4] = sph_covar_matrix[1][2];
    sph_covar[5] = sph_covar_matrix[2][2];
}

std::tuple<torch::Tensor, torch::Tensor> cartesian_to_spherical_fwd_tensor(
    const torch::Tensor &means,  // [N, 3]
    const torch::Tensor &covars // [N, 6]
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(covars);

    uint32_t N = means.size(0);

    torch::Tensor sph_means = torch::empty({N, 3}, means.options());
    torch::Tensor sph_covars = torch::empty({N, 6}, covars.options());

    if (N > 0) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            means.scalar_type(), "cartesian_to_spherical_fwd", [&]() {
                cartesian_to_spherical_fwd_kernel<<<
                    (N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.data_ptr<scalar_t>(),
                    sph_means.data_ptr<scalar_t>(),
                    sph_covars.data_ptr<scalar_t>());
            });
    }

    return std::make_tuple(sph_means, sph_covars);
}
} // namespace gsplat
