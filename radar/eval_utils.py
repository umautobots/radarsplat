import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d
import imageio
from typing import Literal
from radar.utils import *
import torch

def eval_geometry(pixels, pixels_occ, out_img, out_occ, points, sensor_type, range_resolution, azimuth_resolution, max_range, tau=0.1):
    pixels = pixels.squeeze()
    pixels_occ = pixels_occ.squeeze()
    out_img = out_img.squeeze()
    out_occ = out_occ.squeeze()
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
    
    pixels_cart[np.isnan(pixels_cart)] = 0.0
    pixels_occ_cart[np.isnan(pixels_occ_cart)] = 0.0
    out_img_cart[np.isnan(out_img_cart)] = 0.0
    out_occ_cart[np.isnan(out_occ_cart)] = 0.0

    width = pixels_cart.shape[0]
    range_resolution = max_range*2/width
    # Range filter
    range_mask = np.linalg.norm(points,axis=1) < max_range
    points = points[range_mask]

    # Angle filter
    r = np.linalg.norm(points[:,:2],axis=1)
    z = points[:,2]
    phi = np.arctan2(z,r)
    
    # Mask type I
    # show_mask = np.abs(phi) < 1.8/180*np.pi

    # # Mask type II
    show_mask1 = phi < 1.8*2/180*np.pi
    show_mask2 = phi > -1.8/180*np.pi
    show_mask = show_mask1*show_mask2
    
    points = points[show_mask]
    # Floor filter
    floor_thres = points[:,2] < 1.5
    points = points[floor_thres]
    
    # Visible filter
    radar_visible_points = radar_visible_lidar_points_filter(points, pixels_cart, resolution=(max_range*2/pixels_cart.shape[0]), power_thres=0.15, origin_x=0, origin_y=0)
    
    lidar_gt = radar_visible_points
    lidar_gt[:,2]=0
    
    occ_pc = bev_to_pointcloud(out_occ_cart, resolution=0.1, power_thres=0.5) # 0.5
    
    occ_pc.paint_uniform_color([0,1,0])
    pcd_radar_visible_lidar = o3d.geometry.PointCloud()
    pcd_radar_visible_lidar.points = o3d.utility.Vector3dVector(radar_visible_points)
    pcd_radar_visible_lidar.paint_uniform_color([1,0,0])
    # o3d.visualization.draw_geometries([occ_pc, pcd_radar_visible_lidar])

    ### Eval pointcloud metrics ###
    gt_points = np.asarray(pcd_radar_visible_lidar.points)
    recons_points = np.asarray(occ_pc.points)

    # Compute metrics
    rmse = compute_rmse(gt_points, recons_points)
    cd = chamfer_distance(gt_points, recons_points)
    rcd = relative_chamfer_distance(cd, gt_points)
    precision, recall, accuracy = precision_recall_accuracy(gt_points, recons_points, tau=tau)

    return rmse, cd, rcd, precision, recall, accuracy



import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def compute_rmse(gt_points, recons_points):
    """Compute the RMSE between two point clouds."""
    tree_gt = cKDTree(gt_points)
    dist_recons_to_gt, _ = tree_gt.query(recons_points)

    # RMSE calculation
    rmse = np.sqrt(np.mean(dist_recons_to_gt**2))
    return rmse

def chamfer_distance(gt_points, recons_points):
    """Compute the Chamfer Distance (CD) between two point clouds."""
    tree_gt = cKDTree(gt_points)
    tree_recons = cKDTree(recons_points)

    # Compute nearest distances
    dist_gt_to_recons, _ = tree_gt.query(recons_points)
    dist_recons_to_gt, _ = tree_recons.query(gt_points)

    # Chamfer Distance
    cd = np.mean(dist_gt_to_recons**2) + np.mean(dist_recons_to_gt**2)
    return cd

def relative_chamfer_distance(cd, gt_points):
    """Compute the Relative Chamfer Distance (RCD)."""
    min_bounds = np.min(gt_points, axis=0)
    max_bounds = np.max(gt_points, axis=0)
    diag_length = np.linalg.norm(max_bounds - min_bounds)
    return cd / diag_length

def precision_recall_accuracy(gt_points, recons_points, tau=0.01):
    """Compute Precision, Recall, and Accuracy for point cloud comparison."""
    tree_gt = cKDTree(gt_points)
    tree_recons = cKDTree(recons_points)

    # Find nearest distances
    dist_gt_to_recons, _ = tree_gt.query(recons_points)
    dist_recons_to_gt, _ = tree_recons.query(gt_points)

    # Define matches based on threshold tau
    matched_recons = np.sum(dist_gt_to_recons < tau)
    matched_gt = np.sum(dist_recons_to_gt < tau)

    precision = matched_recons / len(recons_points)
    recall = matched_gt / len(gt_points)
    accuracy = (matched_recons + matched_gt) / (len(gt_points) + len(recons_points))

    return precision, recall, accuracy
