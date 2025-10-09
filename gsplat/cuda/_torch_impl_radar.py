from typing import Optional, Tuple

import torch
from torch import Tensor

from gsplat.cuda._torch_impl import _quat_scale_to_matrix
from gsplat.utils import upper_triangular_to_matrices, matrices_to_upper_triangular

def accumulate(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    # opacities_w_reflectance: Tensor,  # [C, N]
    opacities: Tensor,  # [C, N]
    # noise_probs: Tensor,  # [C, N]
    gaussian_ids: Tensor,  # [M]
    pixel_ids: Tensor,  # [M]
    camera_ids: Tensor,  # [M]
    image_width: int,
    image_height: int,
) -> Tuple[Tensor, Tensor]:
    """Alpah compositing of 2D Gaussians in Pure Pytorch.

    This function performs alpha compositing for Gaussians based on the pair of indices
    {gaussian_ids, pixel_ids, camera_ids}, which annotates the intersection between all
    pixels and Gaussians. These intersections can be accquired from
    `gsplat.rasterize_to_indices_in_range`.

    .. note::

        This function exposes the alpha compositing process into pure Pytorch.
        So it relies on Pytorch's autograd for the backpropagation. It is much slower
        than our fully fused rasterization implementation and comsumes much more GPU memory.
        But it could serve as a playground for new ideas or debugging, as no backward
        implementation is needed.

    .. warning::

        This function requires the `nerfacc` package to be installed. Please install it
        using the following command `pip install nerfacc`.

    Args:
        means2d: Gaussian means in 2D. [C, N, 2]
        conics: Inverse of the 2D Gaussian covariance, Only upper triangle values. [C, N, 3]
        opacities: Per-view Gaussian opacities (for example, when antialiasing is
            enabled, Gaussian in each view would efficiently have different opacity). [C, N]
        gaussian_ids: Collection of Gaussian indices to be rasterized. A flattened list of shape [M].
        pixel_ids: Collection of pixel indices (row-major) to be rasterized. A flattened list of shape [M].
        camera_ids: Collection of camera indices to be rasterized. A flattened list of shape [M].
        image_width: Image width.
        image_height: Image height.

    Returns:
        A tuple:

        - **renders**: Accumulated opacities, but extend to rgb 3 channel image. [C, image_height, image_width, 3]
        - **alphas**: Accumulated opacities. [C, image_height, image_width, 1]
    """

    # try:
    #     from nerfacc import accumulate_along_rays, render_weight_from_alpha
    # except ImportError:
    #     raise ImportError("Please install nerfacc package: pip install nerfacc")

    C, N = means2d.shape[:2]

    pixel_ids_x = pixel_ids % image_width
    pixel_ids_y = pixel_ids // image_width
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5  # [M, 2]
    deltas = pixel_coords - means2d[camera_ids, gaussian_ids]  # [M, 2]
    c = conics[camera_ids, gaussian_ids]  # [M, 3]
    sigmas = (
        0.5 * (c[:, 0] * deltas[:, 0] ** 2 + c[:, 2] * deltas[:, 1] ** 2)
        + c[:, 1] * deltas[:, 0] * deltas[:, 1]
    )  # [M]
    # alphas_w_reflectance = torch.clamp_max(
    #     opacities_w_reflectance[camera_ids, gaussian_ids] * torch.exp(-sigmas), 0.999
    # )
    alphas = torch.clamp_max(
        opacities[camera_ids, gaussian_ids] * torch.exp(-sigmas), 0.999
    )
    # alphas_noise_probs = torch.clamp_max(
    #     noise_probs[camera_ids, gaussian_ids] * torch.exp(-sigmas), 0.999
    # )

    indices = camera_ids * image_height * image_width + pixel_ids
    total_pixels = C * image_height * image_width

    # TODO: Update this with radar antenna profile.
    # alphas_w_reflectance, counts_w_reflectance = sum_weights(
    #     alphas_w_reflectance, indices, total_pixels
    # )
    alphas, counts = sum_weights(
        alphas, indices, total_pixels
    )
    # alphas_noise_probs, counts = sum_weights(
    #     alphas_noise_probs, indices, total_pixels
    # )

    # alphas_w_reflectance = alphas_w_reflectance.reshape(C, image_height, image_width, 1)
    alphas = alphas.reshape(C, image_height, image_width, 1)
    # alphas_noise_probs = alphas_noise_probs.reshape(C, image_height, image_width, 1)
    counts = counts.reshape(C, image_height, image_width, 1)

    return alphas #alphas_w_reflectance, alphas, alphas_noise_probs #, counts

    # weights, trans = render_weight_from_alpha(
    #     alphas, ray_indices=indices, n_rays=total_pixels
    # )
    # renders = accumulate_along_rays(
    #     weights,
    #     colors[camera_ids, gaussian_ids],
    #     ray_indices=indices,
    #     n_rays=total_pixels,
    # ).reshape(C, image_height, image_width, channels)
    # alphas = accumulate_along_rays(
    #     weights, None, ray_indices=indices, n_rays=total_pixels
    # ).reshape(C, image_height, image_width, 1)
    # return renders, alphas


# For radar
def sum_weights(weights: torch.Tensor, indices: torch.Tensor, total_pixels: int) -> torch.Tensor:
    """
    Compute a tensor of size [total_pixels] where each index is the average of all the weights
    that have the same index value in `indices`.

    Args:
        weights (torch.Tensor): A tensor of size [N] representing weights.
        indices (torch.Tensor): A tensor of size [N] representing indices.
        total_pixels (int): The size of the output tensor.

    Returns:
        summed_weights (torch.Tensor): A tensor of size [total_pixels] with sum weights.
        counts (torch.Tensor): A tensor of size [total_pixels] with gaussian counts.
    """
    # Initialize an output tensor for summing weights
    summed_weights = torch.zeros(total_pixels, device=weights.device, dtype=weights.dtype)

    # Initialize a tensor to count occurrences for each index
    counts = torch.zeros(total_pixels, device=weights.device, dtype=weights.dtype)

    # Accumulate weights and counts for each index
    summed_weights.index_add_(0, indices, weights)
    counts.index_add_(0, indices, torch.ones_like(weights))

    # # Avoid division by zero by replacing zero counts with 1 (no contribution to average)
    # counts = torch.where(counts == 0, torch.ones_like(counts), counts)

    # Compute the average weights
    # averaged_weights = summed_weights / counts

    return summed_weights, counts

def _rasterize_to_radar_pixels(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [C, N]
    # noise_probs: Tensor,  # [C, N]
    # reflectance: Tensor,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    batch_per_iter: int = 100,
):
    """Pytorch implementation of `gsplat.cuda._wrapper.rasterize_to_pixels()`.

    This function rasterizes 2D Gaussians to pixels in a Pytorch-friendly way. It
    iteratively accumulates the renderings within each batch of Gaussians. The
    interations are controlled by `batch_per_iter`.

    .. note::
        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.

    .. note::

        This function relies on Pytorch's autograd for the backpropagation. It is much slower
        than our fully fused rasterization implementation and comsumes much more GPU memory.
        But it could serve as a playground for new ideas or debugging, as no backward
        implementation is needed.

    .. warning::

        This function requires the `nerfacc` package to be installed. Please install it
        using the following command `pip install nerfacc`.
    """
    from ._wrapper import rasterize_to_indices_in_range_radargs

    # Combined occuapncy and reflectance
    # opacities_w_reflectance = torch.clamp(opacities + noise_probs, min=1e-6, max=1) * reflectance

    C, N = means2d.shape[:2]
    n_isects = len(flatten_ids)
    device = means2d.device

    # render_powers = torch.zeros((C, image_height, image_width, 1), device=device)
    render_occupancy = torch.zeros((C, image_height, image_width, 1), device=device)
    # render_noise_probs = torch.zeros((C, image_height, image_width, 1), device=device)

    # Split Gaussians into batches and iteratively accumulate the renderings
    block_size = tile_size * tile_size
    isect_offsets_fl = torch.cat(
        [isect_offsets.flatten(), torch.tensor([n_isects], device=device)]
    )
    max_range = (isect_offsets_fl[1:] - isect_offsets_fl[:-1]).max().item()
    num_batches = (max_range + block_size - 1) // block_size
    for step in range(0, num_batches, batch_per_iter):
        
        # summed_weights_w_reflectance = torch.zeros((C, image_height, image_width, 1), device=device)
        summed_weights = torch.zeros((C, image_height, image_width, 1), device=device)
        # summed_noise_weights = torch.zeros((C, image_height, image_width, 1), device=device)
        # counts = torch.zeros((C, image_height, image_width, 1), device=device)

        # transmittances = 1.0 - render_alphas[..., 0]
        
        transmittances = torch.ones((C, image_height, image_width, 1), device=device) # transmittances is not used in rasterize_to_indices_in_range_radargs
        # TODO: check why this transmittances has to be ones instead of zeros.
        
        # Find the M intersections between pixels and gaussians.
        # Each intersection corresponds to a tuple (gs_id, pixel_id, camera_id)
        gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range_radargs(
            step,
            step + batch_per_iter,
            transmittances,
            means2d,
            conics,
            # opacities_w_reflectance, 
            opacities,
            image_width,
            image_height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )  # [M], [M]
        if len(gs_ids) == 0:
            break

        # Accumulate the renderings within this batch of Gaussians.
        # summed_weights_w_reflectance_step, summed_weights_step, summed_noise_weights_step = 
        summed_weights_step = accumulate(
            means2d,
            conics,
            # opacities_w_reflectance,
            opacities,
            # noise_probs,
            gs_ids,
            pixel_ids,
            camera_ids,
            image_width,
            image_height,
        )
        # summed_weights_w_reflectance = summed_weights_w_reflectance + summed_weights_w_reflectance_step
        summed_weights = summed_weights + summed_weights_step
        # summed_noise_weights = summed_noise_weights + summed_noise_weights_step

        # counts = counts + counts_step

    # render_powers = summed_weights_w_reflectance #/ counts
    render_occupancy = summed_weights #/ counts
    # render_noise_probs = summed_noise_weights #/ counts

    return render_occupancy #render_powers, render_occupancy, render_noise_probs


class _cartesian_to_spherical_fixed_bwd(torch.nn.Module):
    def __init__(self):
        super(_cartesian_to_spherical_fixed_bwd, self).__init__()

    def forward(self, means, covars):
        return _cartesian_to_spherical_fixed_bwd_func.apply(means, covars)
    
class _cartesian_to_spherical_fixed_bwd_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, covars):
        """
        Forward pass: Converts Cartesian means and covariance matrices to spherical space.
        
        Parameters:
            means (torch.Tensor): Cartesian means, shape (N, 3).
            covars (torch.Tensor): Upper triangular Cartesian covariance elements, shape (N, 6).
        
        Returns:
            sph_means (torch.Tensor): Spherical means, shape (N, 3) [r, azimuth, inclination].
            sph_covars (torch.Tensor): Spherical covariances, shape (N, 6) (upper triangular).
        """
        x, y, z = means[:, 0], means[:, 1], means[:, 2]

        # Means transformation
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.atan2(y, x)  # Azimuth
        phi = torch.acos(z / r.clamp(min=1e-7))  # Inclination

        sph_means = torch.stack((r, theta, phi), dim=-1)

        # Reconstruct full Cartesian covariance matrices
        cov_matrices = torch.zeros(covars.shape[0], 3, 3, device=covars.device)
        cov_matrices[:, 0, 0] = covars[:, 0]  # xx
        cov_matrices[:, 0, 1] = cov_matrices[:, 1, 0] = covars[:, 1]  # xy
        cov_matrices[:, 0, 2] = cov_matrices[:, 2, 0] = covars[:, 2]  # xz
        cov_matrices[:, 1, 1] = covars[:, 3]  # yy
        cov_matrices[:, 1, 2] = cov_matrices[:, 2, 1] = covars[:, 4]  # yz
        cov_matrices[:, 2, 2] = covars[:, 5]  # zz

        # Jacobian calculation
        r2 = r**2
        xy2 = x**2 + y**2
        r_xy = torch.sqrt(xy2)

        J = torch.zeros(means.shape[0], 3, 3, device=means.device)

        # d(r)/d(x, y, z)
        J[:, 0, 0] = x / r
        J[:, 0, 1] = y / r
        J[:, 0, 2] = z / r

        # d(theta)/d(x, y, z)
        J[:, 1, 0] = -y / xy2.clamp(min=1e-7)
        J[:, 1, 1] = x / xy2.clamp(min=1e-7)
        J[:, 1, 2] = 0

        # d(phi)/d(x, y, z)
        J[:, 2, 0] = -x * z / (r2 * r_xy.clamp(min=1e-7))
        J[:, 2, 1] = -y * z / (r2 * r_xy.clamp(min=1e-7))
        J[:, 2, 2] = r_xy / r2

        # Covariance transformation
        sph_cov_matrices = torch.matmul(J, torch.matmul(cov_matrices, J.transpose(-1, -2)))

        # Extract upper triangular elements from spherical covariance matrices
        sph_covars = torch.stack((
            sph_cov_matrices[:, 0, 0],  # rr
            sph_cov_matrices[:, 0, 1],  # r-theta
            sph_cov_matrices[:, 0, 2],  # r-phi
            sph_cov_matrices[:, 1, 1],  # theta-theta
            sph_cov_matrices[:, 1, 2],  # theta-phi
            sph_cov_matrices[:, 2, 2],  # phi-phi
        ), dim=-1)

        # Save tensors for backward computation
        ctx.save_for_backward(means, cov_matrices, J, sph_means)

        return sph_means, sph_covars

    @staticmethod
    def backward(ctx, grad_sph_means, grad_sph_covars):
        means, cov_matrices, J, sph_means = ctx.saved_tensors
        grad_means = torch.zeros_like(means)
        grad_covars = torch.zeros_like(cov_matrices)

        # Reconstruct full gradient matrix for spherical covariances
        grad_sph_cov_matrices = torch.zeros_like(cov_matrices)
        grad_sph_cov_matrices[:, 0, 0] = grad_sph_covars[:, 0]  # rr
        grad_sph_cov_matrices[:, 0, 1] = grad_sph_cov_matrices[:, 1, 0] = grad_sph_covars[:, 1]  # r-theta
        grad_sph_cov_matrices[:, 0, 2] = grad_sph_cov_matrices[:, 2, 0] = grad_sph_covars[:, 2]  # r-phi
        grad_sph_cov_matrices[:, 1, 1] = grad_sph_covars[:, 3]  # theta-theta
        grad_sph_cov_matrices[:, 1, 2] = grad_sph_cov_matrices[:, 2, 1] = grad_sph_covars[:, 4]  # theta-phi
        grad_sph_cov_matrices[:, 2, 2] = grad_sph_covars[:, 5]  # phi-phi

        # Backprop through covariance transformation
        grad_covars = torch.matmul(J.transpose(-1, -2), torch.matmul(grad_sph_cov_matrices, J))

        # Backprop through means transformation
        r, theta, phi = sph_means[:, 0], sph_means[:, 1], sph_means[:, 2]
        J_inv = torch.zeros_like(J)
        J_inv[:, 0, 0] = torch.sin(phi) * torch.cos(theta)
        J_inv[:, 0, 1] = torch.sin(phi) * torch.sin(theta)
        J_inv[:, 0, 2] = torch.cos(phi)
        # TODO: This jacobian is incompleted

        grad_means = torch.matmul(J_inv.transpose(-1, -2), grad_sph_means.unsqueeze(-1)).squeeze(-1)

        # Convert gradients for covars back to upper triangular format
        grad_covars_upper = torch.stack((
            grad_covars[:, 0, 0],  # xx
            grad_covars[:, 0, 1],  # xy
            grad_covars[:, 0, 2],  # xz
            grad_covars[:, 1, 1],  # yy
            grad_covars[:, 1, 2],  # yz
            grad_covars[:, 2, 2],  # zz
        ), dim=-1)

        return grad_means, grad_covars_upper

# Note this method has inaccurate spherical covariance when x, y --> 0. The is due to the first-order Tyler expension in J*Cov*J^T. 
# Possible solution:
# -- Set the upper bound of theta and phi covariance
# -- Higher-Order Approximation
# -- Define all Gaussian in Spherical space in the beginning
def _cartesian_to_spherical(means, covars):
    """
    Converts 3D Gaussians (means and upper triangular covariances) from Cartesian to spherical space.
    
    Parameters:
        means (torch.Tensor): Cartesian means, shape (N, 3).
        covars (torch.Tensor): Upper triangular covariance elements, shape (N, 6).
            Order of elements: [xx, xy, xz, yy, yz, zz].
    
    Returns:
        sph_means (torch.Tensor): Spherical means, shape (N, 3) [r, azimuth, inclination].
        sph_covars (torch.Tensor): Spherical covariances, shape (N, 6).
            Upper triangular elements of transformed covariances in spherical space.
    """
    x, y, z = means[:, 0], means[:, 1], means[:, 2]
    
    # Means transformation
    r = torch.sqrt(x**2 + y**2 + z**2)
    r_xy = torch.sqrt(x**2 + y**2)
    
    theta = torch.atan2(y, x)  # Azimuth
    phi = torch.atan2(z, r_xy.clamp(min=1e-7))  # Inclination (phi=0 when the point is on z plane)
    
    sph_means = torch.stack((r, theta, phi), dim=-1)
    
    J = torch.zeros(means.shape[0], 3, 3, device=means.device)
    
    # d(r)/d(x, y, z)
    J[:, 0, 0] = x / r
    J[:, 0, 1] = y / r
    J[:, 0, 2] = z / r
    
    # d(theta)/d(x, y, z)
    J[:, 1, 0] = -y / r_xy.clamp(min=1e-7)**2
    J[:, 1, 1] = x / r_xy.clamp(min=1e-7)**2
    J[:, 1, 2] = 0
    
    # d(phi)/d(x, y, z)
    J[:, 2, 0] = -x * z / (r_xy.clamp(min=1e-7) * r**2)
    J[:, 2, 1] = -y * z / (r_xy.clamp(min=1e-7) * r**2)
    J[:, 2, 2] = r_xy / r**2
    
    # Reconstruct full Cartesian covariance matrices from upper triangular elements
    cov_matrices = upper_triangular_to_matrices(covars)

    # Covariance transformation
    sph_cov_matrices = torch.matmul(J, torch.matmul(cov_matrices, J.transpose(-1, -2)))
    
    # # Differentiable Eigenvalue Decomposition
    # eigvals, eigvecs = torch.linalg.eigh(sph_cov_matrices)

    # # Selectively clamp eigenvalues for theta and phi (indices 1 and 2)
    # max_variance_theta_phi = torch.pi * 2
    # clamped_eigvals = eigvals.clone()
    # clamped_eigvals[:, 1:] = clamped_eigvals[:, 1:].clamp(max=max_variance_theta_phi)

    # # Reconstruct bounded covariance matrix
    # sph_cov_matrices_bounded = torch.matmul(
    #     eigvecs, torch.matmul(torch.diag_embed(clamped_eigvals), eigvecs.transpose(-1, -2))
    # )

    # Extract upper triangular elements from spherical covariance matrices
    sph_covars = matrices_to_upper_triangular(sph_cov_matrices)
    
    mask = sph_means[:,1] < 0
    sph_means[mask,1] = 2*torch.pi + sph_means[mask,1]
    
    return sph_means, sph_covars


# This is probably wrong
def _cartesian_to_spherical_2nd_order(means, covars):
    """
    Converts 3D Gaussians (means and upper triangular covariances) from Cartesian to spherical space
    with higher-order corrections.

    Parameters:
        means (torch.Tensor): Cartesian means, shape (N, 3).
        covars (torch.Tensor): Upper triangular covariance elements, shape (N, 6).
            Order of elements: [xx, xy, xz, yy, yz, zz].

    Returns:
        sph_means (torch.Tensor): Spherical means, shape (N, 3) [r, azimuth, inclination].
        sph_covars (torch.Tensor): Spherical covariances, shape (N, 6).
            Upper triangular elements of transformed covariances in spherical space.
    """
    x, y, z = means[:, 0], means[:, 1], means[:, 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    r_xy = torch.sqrt(x**2 + y**2)

    # Spherical means
    theta = torch.atan2(y, x)  # Azimuth
    phi = torch.atan2(z, r_xy.clamp(min=1e-7))  # Inclination
    sph_means = torch.stack((r, theta, phi), dim=-1)

    # Jacobian (first derivatives)
    J = torch.zeros(means.shape[0], 3, 3, device=means.device)
    # d(r)/d(x, y, z)
    J[:, 0, 0] = x / r
    J[:, 0, 1] = y / r
    J[:, 0, 2] = z / r
    # d(theta)/d(x, y, z)
    J[:, 1, 0] = -y / r_xy.clamp(min=1e-7)**2
    J[:, 1, 1] = x / r_xy.clamp(min=1e-7)**2
    J[:, 1, 2] = 0
    # d(phi)/d(x, y, z)
    J[:, 2, 0] = -x * z / (r_xy.clamp(min=1e-7) * r**2)
    J[:, 2, 1] = -y * z / (r_xy.clamp(min=1e-7) * r**2)
    J[:, 2, 2] = r_xy / r**2

    # Reconstruct full Cartesian covariance matrices from upper triangular elements
    cov_matrices = upper_triangular_to_matrices(covars)

    # First-order covariance transformation
    first_order_cov = torch.matmul(J, torch.matmul(cov_matrices, J.transpose(-1, -2)))

    # Hessians (second derivatives)
    H = torch.zeros(means.shape[0], 3, 3, 3, device=means.device)
    
    # Hessian for r
    H[:, 0, 0, 0] = (y**2 + z**2) / r**3
    H[:, 0, 1, 1] = (x**2 + z**2) / r**3
    H[:, 0, 2, 2] = (x**2 + y**2) / r**3
    H[:, 0, 0, 1] = -x * y / r**3
    H[:, 0, 1, 0] = H[:, 0, 0, 1]
    H[:, 0, 0, 2] = -x * z / r**3
    H[:, 0, 2, 0] = H[:, 0, 0, 2]
    H[:, 0, 1, 2] = -y * z / r**3
    H[:, 0, 2, 1] = H[:, 0, 1, 2]

    # Hessian for theta
    r_xy_sq = r_xy.clamp(min=1e-7)**2
    H[:, 1, 0, 0] = (2 * x * y) / r_xy_sq**2
    H[:, 1, 1, 1] = -(2 * x * y) / r_xy_sq**2
    H[:, 1, 0, 1] = (y**2 - x**2) / r_xy_sq**2
    H[:, 1, 1, 0] = H[:, 1, 0, 1]

    # Hessian for phi
    H[:, 2, 0, 0] = (2 * x * z**2) / (r**3 * r_xy.clamp(min=1e-7))
    H[:, 2, 1, 1] = (2 * y * z**2) / (r**3 * r_xy.clamp(min=1e-7))
    H[:, 2, 2, 2] = -(r_xy / r**3)
    H[:, 2, 0, 1] = (z * (x * y)) / (r**3 * r_xy.clamp(min=1e-7))
    H[:, 2, 1, 0] = H[:, 2, 0, 1]

    # Second-order covariance correction
    second_order_cov = torch.zeros_like(first_order_cov)
    for i in range(3):  # r, theta, phi
        for j in range(3):  # Diagonal elements of covariance
            second_order_cov[:, i, j] += 0.5 * torch.einsum('bi,bij,bj->b', H[:, i, :, j], cov_matrices, H[:, i, :, j])

    # Final covariance with second-order correction
    sph_cov_matrices = first_order_cov + second_order_cov

    # Extract upper triangular elements from spherical covariance matrices
    sph_covars = matrices_to_upper_triangular(sph_cov_matrices)

    return sph_means, sph_covars


if __name__ == "__main__":
    print("Unit Test for _cartesian_to_spherical")
    
     # Define input means and covariances
    means = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0]])
    
    covars = torch.tensor([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],  # Diagonal covariance matrix
        [1.0, 0.1, 0.0, 1.0, 0.2, 1.0],  # Slightly off-diagonal
        [1.0, 0.0, 0.1, 1.0, 0.1, 1.0],  # Z-axis emphasis
        [1.0, 0.5, 0.5, 1.0, 0.5, 1.0]   # Mixed off-diagonal
    ])
    
    sph_means, sph_covars = _cartesian_to_spherical(means, covars)