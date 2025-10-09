import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d
import cv2
import imageio
from typing import Literal

# This code is borrow from RadarFields github
# https://github.com/princeton-computational-imaging/RadarFields/blob/ee76d76570f58b3d8539eafd7df0c188b58af333/utils/vis.py#L36
def polar_to_cart(fft_img, num_bins_to_show, bin_size, num_azims,
                          resolution=781, noise_floor=None, norm=False):
    """## Interpolate to render a single FFT frame in cartesian coordinates"""

    height, width = fft_img.shape
    assert(num_bins_to_show <= width)
    radians_per_azim = np.pi*2.0/num_azims

    # Setting up grid interpolator
    a_pts = np.arange(-1, num_azims+1)
    r_pts = np.arange(num_bins_to_show)
    fft_img_wrapped = np.zeros((height+2,width))
    fft_img_wrapped[1:height+1,:] = fft_img
    fft_img_wrapped[0,:] = fft_img[-1,:]
    fft_img_wrapped[-1,:] = fft_img[0,:]
    interp = RegularGridInterpolator((a_pts, r_pts), fft_img_wrapped[:,:num_bins_to_show],
                                     bounds_error=False, fill_value=0.0)

    render = np.zeros((resolution, resolution))
    white_mask = np.zeros_like(render)

    # Computing pixel pitch and range bounds for rendered img
    max_radial_distance = (num_bins_to_show)*bin_size - (bin_size/2.0)
    pitch = ((num_bins_to_show * 2.0) - 1) * bin_size / resolution

    # Array of image indices
    i, j = np.meshgrid(range(resolution), range(resolution-1,-1,-1), indexing='ij')
    img_coords = np.stack((i, j), axis=-1)*pitch-max_radial_distance #[resolution, resolution, 2]

    # Computing pixel world polar coords
    radial_distances = np.sqrt(img_coords[...,0]**2 + img_coords[...,1]**2)
    azimuths = np.arctan2(img_coords[...,1],img_coords[...,0])+(np.pi/2.0)
    azimuths[azimuths < 0.0] += 2*np.pi
    white_mask[radial_distances > max_radial_distance] = 1.0

    # Mapping to FFT pixel space
    rs = (radial_distances / bin_size) - 1
    azs = (azimuths / radians_per_azim) - 1
    render = interp(np.stack((azs, rs), axis=-1)) # [resolution, resolution]

    if norm:
        render = render - np.min(render)
        render = render / np.max(render)

    # Noise thresholding
    if noise_floor is not None:
        has_signal = render > noise_floor
        render = render * has_signal
    
    render[white_mask.astype(bool)] = np.nan # white
    
    return render

def cart_to_polar(
    cart_img,
    num_bins_to_show,
    bin_size,
    num_azims,
    resolution=781,
    noise_floor=None,
    norm=False
):
    """
    Convert from Cartesian image to polar/FFT domain in a fully vectorized way, with:
      - Azimuth = 0 along the positive x-axis (pointing right).
      - Azimuth increases clockwise.
    
    Args:
        cart_img (np.ndarray): The Cartesian domain image, shape [resolution, resolution].
        num_bins_to_show (int): Number of range bins (r) in the polar domain.
        bin_size (float): Range-bin spacing (in x/y units).
        num_azims (int): Number of azimuth bins (theta) in the polar domain.
        resolution (int): Size of the cart_img array in x and y.
        noise_floor (float, optional): If set, zero out values below this floor.
        norm (bool, optional): If True, min-max normalize the result.

    Returns:
        np.ndarray: Reconstructed polar/FFT image of shape [num_azims, num_bins_to_show].
    """

    # ---- 1) Reproduce the forward transform's radial extent ----
    # If your forward transform uses:
    #   max_radial_distance = num_bins_to_show*bin_size - bin_size/2
    # then do the same here to match.
    max_radial_distance = num_bins_to_show * bin_size - (bin_size / 2.0)

    # ---- 2) Build the y and x coordinate arrays for the Cartesian image ----
    # In many imaging conventions, row=0 is the top, so y should go from +max to -max.
    y_coords = np.linspace(max_radial_distance, -max_radial_distance, resolution)
    x_coords = np.linspace(-max_radial_distance, max_radial_distance, resolution)

    # ---- 3) Create an interpolator for the Cartesian image: (y, x) order ----
    interp_cart = RegularGridInterpolator(
        (y_coords, x_coords),
        cart_img,
        bounds_error=False,
        fill_value=0.0
    )

    # ---- 4) Prepare a grid of (azimuth, range) for the polar image ----
    # We want azimuth in [0, 2*pi), with clockwise rotation:
    #   x = r * cos(theta)
    #   y = -r * sin(theta)
    radians_per_azim = 2.0 * np.pi / num_azims

    # Azimuth bins: shape [num_azims]
    azimuths = np.arange(num_azims) * radians_per_azim

    # Range bins: shape [num_bins_to_show], offset so bin 0 ~ r=bin_size
    radial_bins = (np.arange(num_bins_to_show) + 1) * bin_size

    # Create a 2D mesh of angles and radii: shape both => (num_azims, num_bins_to_show)
    # indexing='ij' => A[a,r], R[a,r]
    A, R = np.meshgrid(azimuths, radial_bins, indexing='ij')

    # ---- 5) Compute (x, y) for all polar coordinates in a vectorized manner ----
    X = R * np.cos(A)      # shape (num_azims, num_bins_to_show)
    Y = -R * np.sin(A)     # the minus sign makes the angle increase clockwise

    # ---- 6) Interpolate all points in a single call ----
    # Interpolator expects points in shape [N, 2], with (y, x) order
    query_points = np.stack([Y.ravel(), X.ravel()], axis=-1)  # shape (num_azims*num_bins_to_show, 2)
    polar_values = interp_cart(query_points)

    # ---- 7) Reshape back to [num_azims, num_bins_to_show] ----
    polar_img = polar_values.reshape(num_azims, num_bins_to_show)

    # ---- 8) Optional normalization ----
    if norm and polar_img.size > 0:
        min_val, max_val = polar_img.min(), polar_img.max()
        if max_val > min_val:
            polar_img = (polar_img - min_val) / (max_val - min_val)

    # ---- 9) Optional noise threshold ----
    if noise_floor is not None:
        polar_img[polar_img <= noise_floor] = 0.0

    return polar_img

import torch
from matplotlib.colors import Normalize
cm = plt.get_cmap("hot")
cm_rd = plt.get_cmap("viridis")

def visualize_in_polar_space(pixels, out_img, sensor_type, viz_type=Literal["plt","rgb"]):
    pixels = pixels.squeeze()
    out_img = out_img.squeeze()
    if sensor_type=="sonar": # TODO: use self.parser.azimuth_coverage
        # For sonar, we only use 130 deg for training
        out_img_130 = out_img.clone()
        out_img_130 = out_img_130[:pixels.shape[0],:]
    
    if viz_type == "plt":
        canvas = torch.cat([pixels, out_img], dim=0).detach().cpu().numpy()
        canvas = np.expand_dims(canvas, axis=0)
        if sensor_type=="sonar": # TODO: use self.parser.azimuth_coverage
            img = torch.cat((out_img_130, pixels),dim=0)
        else:
            img = torch.cat((out_img, pixels),dim=0)
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        ax.imshow(img.detach().cpu().numpy(), vmin=0.0, vmax=1.0)
        ax.axis('off')
        return fig, ax
    if viz_type == "rgb":
        canvas_list = [out_img.unsqueeze(-1), pixels.unsqueeze(-1)]
        canvas = torch.cat(canvas_list, dim=0).cpu().numpy()
        canvas = (canvas * 255).astype(np.uint8)
        canvas = canvas.repeat(3,2)
        return canvas

def visualize_in_cart_space(pixels, out_img, out_occ, sensor_type, range_resolution, azimuth_resolution, viz_type=Literal["plt","rgb"]):
    pixels = pixels.squeeze()
    out_img = out_img.squeeze()
    out_occ = out_occ.squeeze()

    if sensor_type=="sonar": # TODO: use self.parser.azimuth_coverage
        # For sonar, we pad 130 deg polar image to 360 deg 
        pixels_360 = torch.zeros_like(out_img)
        pixels_360[:pixels.shape[0],:] = pixels
    else:
        pixels_360 = pixels
    
    num_bins_to_show = pixels.shape[1]
    bin_size = range_resolution
    num_azims = int(360/azimuth_resolution)
    pixels_cart = polar_to_cart(pixels_360.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    out_img_cart = polar_to_cart(out_img.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    out_occ_cart = polar_to_cart(out_occ.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    
    pixels_cart[np.isnan(pixels_cart)] = 1.0
    out_img_cart[np.isnan(out_img_cart)] = 1.0
    out_occ_cart[np.isnan(out_occ_cart)] = 1.0

    if viz_type == "plt":
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        cart_img = np.hstack((out_img_cart, pixels_cart))
        ax.imshow(cart_img, vmin=0.0, vmax=1.0)
        ax.axis('off')
        return fig, ax

    if viz_type == "rgb":
        canvas_cart_list = [out_img_cart, pixels_cart]
        canvas_cart = np.expand_dims(np.hstack(canvas_cart_list), axis=-1)
        canvas_cart = (canvas_cart * 255).astype(np.uint8)
        canvas_cart = canvas_cart.repeat(3,2)
        return canvas_cart

def visualize_with_lidar(pixels, out_img, out_occ, points, sensor_type, range_resolution, azimuth_resolution, max_range):
    pixels = pixels.squeeze()
    out_img = out_img.squeeze()
    if sensor_type=="sonar": # TODO: use self.parser.azimuth_coverage
        # For sonar, we pad 130 deg polar image to 360 deg 
        pixels_360 = torch.zeros_like(out_img)
        pixels_360[:pixels.shape[0],:] = pixels
    else:
        pixels_360 = pixels
    
    num_bins_to_show = pixels.shape[1]
    bin_size = range_resolution
    num_azims = int(360/azimuth_resolution)
    pixels_cart = polar_to_cart(pixels_360.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    out_img_cart = polar_to_cart(out_img.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    
    pixels_cart[np.isnan(pixels_cart)] = 1.0
    out_img_cart[np.isnan(out_img_cart)] = 1.0
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 8), dpi=150)
    canvas = np.zeros_like(pixels_cart)
    ax[0].imshow(canvas)
    width = pixels_cart.shape[0]
    range_resolution = max_range*2/width
    range_mask = np.linalg.norm(points,axis=1) < max_range
    points = points[range_mask]
    ax[0].scatter(points[:, 0]/range_resolution + width/2, points[:, 1]/range_resolution + width/2, s=0.01, c=points[:, 2], vmin=-5, vmax=5, cmap='jet')
    ax[0].axis('off')
    r = np.linalg.norm(points[:,:2],axis=1)
    z = points[:,2]
    phi = np.arctan2(z,r)
    phi = np.abs(phi)
    norm = Normalize(vmin=0, vmax=1.8/180*np.pi)
    normalized_phi = norm(phi)
    colors = plt.cm.viridis.reversed()(normalized_phi)[:,:3]  # Normalize and apply a colormap (e.g., jet)
    colors[phi > 1.8/180*np.pi] = (1.0, 0, 0)
    show_mask = phi < 1.8*2/180*np.pi
    points = points[show_mask]
    floor_thres = points[:,2] < 1.
    points = points[floor_thres]
    ax[1].imshow(canvas)
    ax[1].scatter(points[:, 0]/range_resolution + width/2, points[:, 1]/range_resolution + width/2, s=0.01, c=points[:, 2], vmin=-5, vmax=5, cmap='jet')
    ax[1].axis('off')
    ax[2].imshow(out_img_cart)
    ax[2].axis('off')
    return fig, ax


def visualize_signal_refl(out_occ_refl, sensor_type, range_resolution, azimuth_resolution):
    out_occ_refl = out_occ_refl.squeeze()

    num_bins_to_show = out_occ_refl.shape[1]
    bin_size = range_resolution
    num_azims = int(360/azimuth_resolution)
    out_occ_refl_cart = polar_to_cart(out_occ_refl.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    nan_mask = np.isnan(out_occ_refl_cart)

    cm_viridis = plt.get_cmap("viridis")
    out_occ_refl_cart = cm_viridis(out_occ_refl_cart)
    out_occ_refl_cart[nan_mask] = 1.0
    out_occ_refl_cart = Image.fromarray((out_occ_refl_cart[:,:,:3]*255).astype(np.uint8))
    return out_occ_refl_cart

def visualize_signal_decomposition(out_img, out_occ, out_noise, multipath_bg, sensor_type, range_resolution, azimuth_resolution):
    out_img = out_img.squeeze()
    out_occ = out_occ.squeeze()
    out_noise = out_noise.squeeze()
    multipath_bg = multipath_bg.squeeze()

    num_bins_to_show = out_img.shape[1]
    bin_size = range_resolution
    num_azims = int(360/azimuth_resolution)
    out_img_cart = polar_to_cart(out_img.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    out_occ_cart = polar_to_cart(out_occ.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    out_noise_cart = polar_to_cart(out_noise.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    multipath_bg_cart = polar_to_cart(multipath_bg.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    
    cm_gray = plt.get_cmap("gray")
    out_img_cart[np.isnan(out_img_cart)] = 1.0
    out_img_cart = cm_gray(out_img_cart)
    out_img_cart = Image.fromarray((out_img_cart[:,:,:3]*255).astype(np.uint8))
    out_occ_cart[np.isnan(out_occ_cart)] = 1.0
    out_occ_cart = cm_gray(out_occ_cart)
    out_occ_cart = Image.fromarray((out_occ_cart[:,:,:3]*255).astype(np.uint8))
    out_noise_cart[np.isnan(out_noise_cart)] = 1.0
    out_noise_cart = cm_gray(out_noise_cart)
    out_noise_cart = Image.fromarray((out_noise_cart[:,:,:3]*255).astype(np.uint8))
    multipath_bg_cart[np.isnan(multipath_bg_cart)] = 1.0
    multipath_bg_cart = cm_gray(multipath_bg_cart)
    multipath_bg_cart = Image.fromarray((multipath_bg_cart[:,:,:3]*255).astype(np.uint8))

    return out_img_cart, out_occ_cart, out_noise_cart, multipath_bg_cart
    


def visualize_in_cart_space_separated(pixels, pixels_occ, out_img, out_occ, cart_render_occ, points, sensor_type, range_resolution, azimuth_resolution, max_range, viz_type=Literal["plt","rgb"], viz_lidar=False):
    pixels = pixels.squeeze()
    pixels_occ = pixels_occ.squeeze()
    out_img = out_img.squeeze()
    out_occ = out_occ.squeeze()
    cart_render_occ = cart_render_occ.squeeze()
    if sensor_type=="sonar": # TODO: use self.parser.azimuth_coverage
        # For sonar, we pad 130 deg polar image to 360 deg 
        pixels_360 = torch.zeros_like(out_img)
        pixels_360[:pixels.shape[0],:] = pixels
        pixels_occ_360 = torch.zeros_like(out_img)
        pixels_occ_360[:pixels_occ.shape[0],:] = pixels_occ
    else:
        pixels_360 = pixels
        pixels_occ_360 = pixels_occ
    
    num_bins_to_show = pixels.shape[1]
    bin_size = range_resolution
    num_azims = int(360/azimuth_resolution)
    pixels_cart = polar_to_cart(pixels_360.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    pixels_occ_cart = polar_to_cart(pixels_occ_360.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    out_img_cart = polar_to_cart(out_img.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    out_occ_cart = polar_to_cart(out_occ.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
            resolution=1000, noise_floor=None, norm=False)
    cart_render_occ = cart_render_occ.detach().cpu().numpy()

    pixels_cart_zero_bg = pixels_cart.copy()
    pixels_cart_zero_bg[np.isnan(pixels_cart)] = 0.0
    pixels_occ_cart_zero_bg = pixels_occ_cart.copy()
    pixels_occ_cart_zero_bg[np.isnan(pixels_occ_cart)] = 0.0
    out_img_cart_zero_bg = out_img_cart.copy()
    out_img_cart_zero_bg[np.isnan(out_img_cart)] = 0.0
    out_occ_cart_zero_bg = out_occ_cart.copy()
    out_occ_cart_zero_bg[np.isnan(out_occ_cart)] = 0.0
    cart_render_occ_zero_bg = cart_render_occ.copy()
    cart_render_occ_zero_bg[np.isnan(out_occ_cart)]=0.0
    
    pixels_cart[np.isnan(pixels_cart)] = 1.0
    pixels_occ_cart[np.isnan(pixels_occ_cart)] = 1.0
    out_img_cart[np.isnan(out_img_cart)] = 1.0
    cart_render_occ[np.isnan(out_occ_cart)]=1.0 # reuse out_occ_cart nan mask here
    out_occ_cart[np.isnan(out_occ_cart)] = 1.0

    if viz_lidar:
        fig_lidar, ax_lidar = plt.subplots(figsize=(15, 15), dpi=150)
        canvas = np.zeros_like(pixels_cart)
        ax_lidar.imshow(canvas, cmap='binary')
        width = pixels_cart.shape[0]
        range_resolution = max_range*2/width
        # Range filter
        range_mask = np.linalg.norm(points,axis=1) < max_range
        points = points[range_mask]
        ax_lidar.scatter(points[:, 0]/range_resolution + width/2, points[:, 1]/range_resolution + width/2, s=1, c='tab:blue') #, c=points[:, 2], vmin=-2, vmax=2, cmap='jet_r')
        ax_lidar.axis('off')

        # Angle filter
        r = np.linalg.norm(points[:,:2],axis=1)
        z = points[:,2]
        phi = np.arctan2(z,r)
        # phi = np.abs(phi)
        show_mask1 = phi < 1.8*2/180*np.pi
        show_mask2 = phi > -1.8/180*np.pi
        
        points = points[show_mask1*show_mask2]
        # Floor filter
        floor_thres = points[:,2] < 1.5
        points = points[floor_thres]
        
        fig_lidar_fov, ax_lidar_fov = plt.subplots(figsize=(15, 15), dpi=150)
        ax_lidar_fov.imshow(canvas, cmap='binary')
        ax_lidar_fov.scatter(points[:, 0]/range_resolution + width/2, points[:, 1]/range_resolution + width/2, s=20, c='tab:blue')
        ax_lidar_fov.axis('off')

        radar_visible_points = radar_visible_lidar_points_filter(points, pixels_cart_zero_bg, resolution=(max_range*2/pixels_cart.shape[0]), power_thres=0.15, origin_x=0, origin_y=0)
        
        # # o3d debug
        # pcd = bev_to_pointcloud(pixels_cart_zero_bg, resolution=(max_range*2/pixels_cart.shape[0]), power_thres=0.15, origin_x=0, origin_y=0)
        # pcd.paint_uniform_color([0,0,1])
        # pred_pcd = bev_to_pointcloud(out_img_cart_zero_bg, resolution=(max_range*2/pixels_cart.shape[0]), power_thres=0.15, origin_x=0, origin_y=0)
        # pred_pcd.paint_uniform_color([1,0,1])
        # pcd_lidar = o3d.geometry.PointCloud()
        # pcd_lidar.points = o3d.utility.Vector3dVector(points)
        # pcd_lidar.paint_uniform_color([0,1,0])
        # pcd_radar_visible_lidar = o3d.geometry.PointCloud()
        # pcd_radar_visible_lidar.points = o3d.utility.Vector3dVector(radar_visible_points)
        # pcd_radar_visible_lidar.paint_uniform_color([1,0,0])
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # occ_pcd = bev_to_pointcloud(out_occ_cart_zero_bg, resolution=(max_range*2/pixels_cart.shape[0]), power_thres=0.5)
        # occ_pcd.paint_uniform_color([0,1,1])
        # o3d.visualization.draw_geometries([pcd, pred_pcd, occ_pcd, pcd_lidar, axis])
        
        fig_lidar_fov_visiable = None
        fig_lidar_fov_visiable, ax_lidar_fov_visiable = plt.subplots(figsize=(15, 15), dpi=150)
        ax_lidar_fov_visiable.imshow(canvas, cmap='binary')
        ax_lidar_fov_visiable.scatter(radar_visible_points[:, 0]/range_resolution + width/2, radar_visible_points[:, 1]/range_resolution + width/2, s=20, c='tab:blue')
        ax_lidar_fov_visiable.axis('off')
    else:
        fig_lidar=None
        fig_lidar_fov=None
        fig_lidar_fov_visiable = None

    fig_view3d, ax_view3d = None, None

    pixels_cart = Image.fromarray((pixels_cart*255).astype(np.uint8))
    pixels_occ_cart = Image.fromarray((pixels_occ_cart*255).astype(np.uint8))
    out_img_cart = Image.fromarray((out_img_cart*255).astype(np.uint8))
    out_occ_cart = cm(out_occ_cart)
    out_occ_cart = Image.fromarray((out_occ_cart[:,:,:3]*255).astype(np.uint8))
    cart_render_occ = cm(cart_render_occ)
    cart_render_occ = Image.fromarray((cart_render_occ[:,:,:3]*255).astype(np.uint8))

    return pixels_cart, pixels_occ_cart, out_img_cart, out_occ_cart, cart_render_occ, fig_lidar, fig_lidar_fov, fig_lidar_fov_visiable, fig_view3d, ax_view3d

def bev_to_pointcloud(radar_img, resolution=0.1, power_thres=0.5, origin_x=0, origin_y=0):
    """
    Converts a BEV occupancy image (2D binary grid) into a 3D point cloud.
    
    Args:
        radar_img (numpy.ndarray): 2D array with value between 0 to 1.
        resolution (float): Size of each pixel in meters.
        origin_x (float): X-coordinate of the image's bottom-left corner.
        origin_y (float): Y-coordinate of the image's bottom-left corner.

    Returns:
        o3d.geometry.PointCloud: Point cloud representation.
    """
    occupied_indices = np.argwhere(radar_img > power_thres)

    # Convert pixel coordinates to real-world coordinates
    H, W = radar_img.shape
    points = []
    colors = []

    for px, py in occupied_indices:
        world_y = (px-H/2) * resolution + origin_x 
        world_x = (py-W/2) * resolution + origin_y
        world_z = 0  # BEV is at Z=0
        points.append([world_x, world_y, world_z])
        colors.append([0, 1, 0])

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    return pcd

from scipy.spatial import cKDTree
def radar_visible_lidar_points_filter(lidar_points, radar_img, resolution=0.1, power_thres=0.15, origin_x=0, origin_y=0):
    
    occupied_indices = np.argwhere(radar_img > power_thres)

    # Convert pixel coordinates to real-world coordinates
    H, W = radar_img.shape
    radar_points = []

    for px, py in occupied_indices:
        world_y = (px-H/2) * resolution + origin_x 
        world_x = (py-W/2) * resolution + origin_y
        world_z = 0  # BEV is at Z=0
        radar_points.append([world_x, world_y, world_z])

    # Flatten LiDAR points
    lidar_points_flat = lidar_points.copy()
    lidar_points_flat[:,2]=0

    # Use SciPy KDTree for fast nearest neighbor search (avoids for loop)
    tree = cKDTree(radar_points)
    distances, _ = tree.query(lidar_points_flat, k=1)  # Finds nearest neighbor for all points in pcd1
    mask = (distances < resolution)
    return lidar_points[mask]

import torch.nn.functional as F
def get_multipath_model(multipath_sources, x_shift, y_shift, out_width, range_resolution, azimuth_resolution, angle_diff_thres_deg):
    # multipath_sources_pose, current_pose

    device = 'cuda'
    # Load multipath params
    amplitudes_positive = multipath_sources['amplitudes_positive'].to(device).clone().detach()
    freq_positive = multipath_sources['freq_positive'].to(device).clone().detach()
    phases_positive = multipath_sources['phases_positive'].to(device).clone().detach()
    estimated_A = multipath_sources['estimated_A'].to(device).clone().detach()
    estimated_alpha = multipath_sources['estimated_alpha'].to(device).clone().detach()

    # Compute local pose
    # current_pose_local = multipath_sources_pose.inverse() @ current_pose

    # Obtain reconstructed multipath signal
    azi_id_list = multipath_sources["azi_id_list"].squeeze().clone().detach()
    range_id_list = multipath_sources["range_id_list"].squeeze().clone().detach()

    # List of all distances
    dist_values = torch.arange(out_width, requires_grad=False).to(device).clone().detach().double()*range_resolution
    # List of all angles
    angle_values = torch.linspace(0, 2 * torch.pi, 400+1, requires_grad=False)[:-1].to(device).clone().detach().double() # Assume uniform azimuth angle
    angle_values += np.deg2rad(azimuth_resolution/2) # add helf azimuth resolution offset
    angle_values[angle_values>2 * torch.pi] -= 2 * torch.pi

    # Get theta and d of multipath sources
    source_theta = angle_values[azi_id_list].reshape(-1)

    # source_d = range_id_list * range_resolution
    source_d = 1./freq_positive.squeeze() # distance is more accurate in freq_positive

    # Convert polar to Cartesian coordinates
    source_x = source_d * torch.cos(source_theta)
    source_y = source_d * torch.sin(source_theta)

    # Compute original view angle
    source_xy = torch.stack([source_x, source_y], dim=0).to(device).clone().detach()
    viewdir_xy = source_xy

    # Compute current view angle
    # current_pose_local_xy = current_pose_local[:2,3]
    current_pose_local_xy = torch.tensor([x_shift, y_shift]).to(device)
    current_viewdir_xy = source_xy - current_pose_local_xy.reshape(-1,1)
    current_source_x = current_viewdir_xy[0,:]
    current_source_y = current_viewdir_xy[1,:]

    # Compute view angle diffference in deg
    cos_sim = F.cosine_similarity(viewdir_xy.t(), current_viewdir_xy.t(), dim=1)
    angle_diff_radians = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))
    angle_diff_degrees = torch.rad2deg(angle_diff_radians)

    # Compute mask
    activate_mask = angle_diff_degrees < angle_diff_thres_deg

    # Mask out unactivate multipath source
    current_viewdir_xy = current_viewdir_xy[:, activate_mask]
    amplitudes_positive = amplitudes_positive[:, activate_mask]
    phases_positive = phases_positive[:, activate_mask]
    estimated_A = estimated_A[:, activate_mask]
    estimated_alpha = estimated_alpha[:, activate_mask]

    if current_viewdir_xy.shape[1]==0:
        reconstructed_multipath = None
        current_azi_id_list = None
        current_range_id_list = None
    else:
        # Compute new theta and ditance for multipath signal generation
        current_theta = torch.atan2(current_viewdir_xy[1,:], current_viewdir_xy[0,:])
        current_theta[current_theta<0] = current_theta[current_theta<0]+2*torch.pi
        current_d = torch.norm(current_viewdir_xy, dim=0)
        
        # Get azi and range id of source in current image
        current_azi_id_list = torch.argmin(torch.abs(angle_values.reshape(1,-1)-current_theta.reshape(-1,1)), dim=1)
        current_range_id_list = torch.argmin(torch.abs(dist_values.reshape(1,-1)-current_d.reshape(-1,1)), dim=1)

        # Reconstruct multipath signal
        current_freq = freq_positive # original signal
        # current_freq = 1./current_d
        # (Frank) phase should be the same 
        reconstructed_signal = amplitudes_positive.reshape(-1,1) * torch.cos(2 * torch.pi * current_freq.reshape(-1,1) * dist_values.reshape(1,-1) + phases_positive.reshape(-1,1))        
        reconstructed_signal = estimated_A.reshape(-1,1) * reconstructed_signal
        reconstructed_signal = torch.exp(-estimated_alpha.reshape(-1,1) @ dist_values.reshape(1,-1)) * reconstructed_signal

        # Mask out main target and negative signal
        reconstructed_multipath = reconstructed_signal.clone()
        reconstructed_multipath[reconstructed_multipath<0]=0
        reconstructed_multipath[dist_values.reshape(1,-1) < 1.5*current_d.reshape(-1,1)]=0
    return reconstructed_multipath , current_azi_id_list, current_range_id_list, current_source_x, current_source_y


def compute_view_consistency(gaussians_sh: torch.Tensor, L: int = 10, lambda_weight: float = 1.0):
    """
    Compute view-consistency score for each Gaussian using PyTorch.

    Args:
        gaussians_sh (torch.Tensor): Shape (N, 121), SH coefficients for N Gaussians.
        L (int): Maximum SH order (default is 10).
        lambda_weight (float): Weighting factor for exponential decay in higher orders.

    Returns:
        torch.Tensor: Consistency scores for each Gaussian (higher = more view-consistent).
    """
    N = gaussians_sh.shape[0]  # Number of Gaussians
    scores = torch.zeros(N, device=gaussians_sh.device)

    for i in range(N):
        sh_coeffs = gaussians_sh[i]
        index = 0
        total_energy = torch.tensor(0.0, device=gaussians_sh.device)
        low_order_energy = torch.tensor(0.0, device=gaussians_sh.device)
        weighted_energy = torch.tensor(0.0, device=gaussians_sh.device)

        for l in range(L + 1):
            num_coeffs = 2 * l + 1
            E_l = torch.sum(sh_coeffs[index : index + num_coeffs] ** 2)  # Compute energy
            total_energy += E_l
            
            if l <= 1:
                low_order_energy += E_l  # Sum for consistency ratio
            
            weighted_energy += torch.exp(-torch.tensor(lambda_weight) * l) * E_l  # Weighted score
            
            index += num_coeffs

        # Method 1: Ratio of Low-Order Energy to Total Energy
        score_ratio = low_order_energy / total_energy if total_energy > 0 else torch.tensor(0.0, device=gaussians_sh.device)

        # Method 2: Weighted Energy (higher weight for low-order terms)
        # score_weighted = weighted_energy / total_energy if total_energy > 0 else torch.tensor(0.0, device=gaussians_sh.device)

        # scores[i] = (score_ratio + score_weighted) / 2  # Combine both scores
        
        scores[i] = score_ratio

    return scores


def compute_view_inconsistency(gaussians_sh: torch.Tensor, L: int = 10, lambda_weight: float = 1.0):
    """
    Compute view-inconsistency score for each Gaussian using PyTorch.

    Args:
        gaussians_sh (torch.Tensor): Shape (N, 121), SH coefficients for N Gaussians.
        L (int): Maximum SH order (default is 10).
        lambda_weight (float): Weighting factor for exponential decay in higher orders.

    Returns:
        torch.Tensor: Consistency scores for each Gaussian (higher = more view-consistent).
    """
    N = gaussians_sh.shape[0]  # Number of Gaussians
    scores = torch.zeros(N, device=gaussians_sh.device)

    for i in range(N):
        sh_coeffs = gaussians_sh[i]
        index = 0
        total_energy = torch.tensor(0.0, device=gaussians_sh.device)
        high_order_energy = torch.tensor(0.0, device=gaussians_sh.device)
        weighted_energy = torch.tensor(0.0, device=gaussians_sh.device)

        for l in range(L + 1):
            num_coeffs = 2 * l + 1
            E_l = torch.sum(sh_coeffs[index : index + num_coeffs] ** 2)  # Compute energy
            total_energy += E_l
            
            if l >= L-1:
                high_order_energy += E_l  # Sum for consistency ratio
            
            weighted_energy += torch.exp(-torch.tensor(lambda_weight) * l) * E_l  # Weighted score
            
            index += num_coeffs

        # Ratio of High-Order Energy to Total Energy
        score_ratio = high_order_energy / total_energy if total_energy > 0 else torch.tensor(0.0, device=gaussians_sh.device)

        scores[i] = score_ratio

    return scores