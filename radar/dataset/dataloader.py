import os
import json
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager
from scipy.spatial.transform import Rotation as R

from typing import Literal
from PIL import Image
import yaml
import open3d as o3d

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("TkAgg")  # For local GUI environments

# from .normalize import (
#     align_principle_axes,
#     similarity_from_cameras,
#     transform_cameras,
#     transform_points,
# )

def load_tum_poses(file_path):
    """Load poses from a TUM format file and return a list of transformation matrices."""
    poses = []
    timestamps = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                values = list(map(float, line.split()))
                timestamp, tx, ty, tz, qx, qy, qz, qw = values

                # Use scipy to convert quaternion to rotation matrix
                rotation = R.from_quat([qx, qy, qz, qw])
                R_matrix = rotation.as_matrix()  # 3x3 rotation matrix

                # Construct the 4x4 transformation matrix
                T = np.eye(4)
                T[:3, :3] = R_matrix
                T[:3, 3] = [tx, ty, tz]
                poses.append(T)
                timestamps.append(timestamp)
    return poses

def load_images_from_folder(folder_path):
    """Load all images from a folder and return a list of PIL.Image objects."""
    images = []
    image_names = []
    filenames = os.listdir(folder_path)
    filenames.sort()
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(image_path)
                images.append(img)
                image_names.append(os.path.splitext(filename)[0])
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return images, image_names

def load_multipath_from_folder(multipath_dir):
    """Load all multipath data from a folder and return a list of Dicts."""
    multipath_data = []
    filenames = os.listdir(multipath_dir)
    filenames.sort()
    for filename in filenames:
        if filename.lower().endswith(('.npy')):
            multipath_path = os.path.join(multipath_dir, filename)
            try:
                multipath = np.load(multipath_path, allow_pickle=True)
                multipath_data.append(multipath)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return multipath_data

def compute_spherical_grid_noise_threshold(fft_img, min_range, max_range):
    '''
    ## Dynamically threshold all range-azimuth bins with a grid\
    of range-wise and azimuth-wise medians
    '''
    fft = fft_img.clone()

    # Compute per-bin threshold grid
    medians_range, _ = torch.median(fft[:, min_range:max_range+1], axis=1) # [400]
    medians_azimuth, _ = torch.median(fft,axis=0)
    grid = torch.maximum(medians_azimuth.unsqueeze(0), medians_range.unsqueeze(1))
    has_signal = fft > 1.5*grid # [400, 7536]

    return fft * has_signal
        
class WaveSensorDataParser:
    """Wave-based sensor data parser."""

    def __init__(
        self,
        data_dir: str,
        synced_lidar_map_name: str,
        radar_average_map_name: str,
        factor: int = 1, # not used
        normalize: bool = False, # not used
        test_every: int = 8,
        intermediate_azimuth_resolution: float = 0.1, # Only used when using scanning radar & use_polar=True
        max_range: float = None, # Only used when using scanning radar & use_polar=True
        frame_selection: Optional[List[int]] = None # Select frames for training
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
                
        self.synced_lidar_map_name = synced_lidar_map_name
        self.radar_average_map_name = radar_average_map_name

        assert os.path.exists(
            data_dir
        ), f"Data directory {data_dir} does not exist."

        # Load poses data from the .tum file
        poses_file = os.path.join(data_dir, "radar_trajectory.tum")
        assert os.path.exists(poses_file), f"Poses file {poses_file} does not exist."
        poses = load_tum_poses(poses_file) # (num_images, 4, 4)

        # Load sensor intrinsic parameters from the .yaml file
        sensor_file = os.path.join(data_dir, "sensor.yaml")
        with open(sensor_file, 'r') as file:
            sensor = yaml.safe_load(file)
            
        # sensor intrinsics
        sensor_type = sensor['sensor_type']
        use_polar = sensor['use_polar']

        range_resolution = sensor['range_resolution']
        W_metadata = sensor['W_metadata'] if sensor['W_metadata'] is not None else 0
        azimuth_resolution = sensor['azimuth_resolution']
        azimuth_beamwidth = sensor['azimuth_beamwidth']
        azimuth_coverage = sensor['azimuth_coverage']

        # Load images.
        if factor==1:
            folder_surfix = ""
        else:
            folder_surfix = f"_{factor}"
        image_dir = os.path.join(data_dir, "images" + folder_surfix)
        
        filenames = os.listdir(image_dir)
        filenames.sort()
        image_paths = [os.path.join(image_dir, item) for item in filenames]

        imdata, image_names = load_images_from_folder(image_dir)

        multipath_dir = os.path.join(data_dir, "multipath_model/dist:50" + folder_surfix)
        multipath_data = load_multipath_from_folder(multipath_dir)
        
        # Apply frame_selection
        if frame_selection is not None:
            image_paths = image_paths[frame_selection[0]:frame_selection[1]]
            imdata = imdata[frame_selection[0]:frame_selection[1]]
            image_names = image_names[frame_selection[0]:frame_selection[1]]
            poses = poses[frame_selection[0]:frame_selection[1]]
        
        w2r_mats = []
        radar_ids = []
        Ks_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        for k, im in enumerate(imdata):
            w2r = poses[k]
            w2r_mats.append(w2r)

            # support different camera intrinsics
            radar_id = 0 # use the same camera intrinsics for all images
            radar_ids.append(radar_id)

            if sensor_type == 'scanning_radar' or sensor_type == 'sonar':
                if use_polar == True:
                    W = sensor['W'] - W_metadata if max_range==None else int(max_range / range_resolution)
                    H = int(360*(1/intermediate_azimuth_resolution))
                    K = np.array(
                    [
                        [1/(range_resolution*factor), 0, 0], # range resolution => 0.0596 m 
                        [0, 1/(intermediate_azimuth_resolution*(torch.pi/180)), 0], # azimuth resolution => 1/10 deg
                        [0, 0, 1],
                    ])
                else:
                    W, H = sensor['W'], sensor['H']
                    K = np.array(
                    [
                        [1/range_resolution, 0, W/2.], # cart. range resolution
                        [0, 1/range_resolution, H/2.],
                        [0, 0, 1],
                    ])
            
            Ks_dict[radar_id] = K
            assert sensor['H']==im.height and sensor['W']==im.width, "Image size does not match sensor size."
            imsize_dict[radar_id] = (im.width // factor, H) # apply factor to image width (range) only
            mask_dict[radar_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(radar_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in data folder.")

        radarposes = np.stack(w2r_mats, axis=0)

        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        radarposes = radarposes[inds]
        radar_ids = [radar_ids[i] for i in inds]

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0]) # TODO: not sure what is this bound for?
        # posefile = os.path.join(data_dir, "poses_bounds.npy")
        # if os.path.exists(posefile):
        #     self.bounds = np.load(posefile)[:, -2:]

        # # Normalize the world space.
        # if normalize:
        #     T1 = similarity_from_cameras(camtoworlds)
        #     camtoworlds = transform_cameras(T1, camtoworlds)
        #     points = transform_points(T1, points)

        #     T2 = align_principle_axes(points)
        #     camtoworlds = transform_cameras(T2, camtoworlds)
        #     points = transform_points(T2, points)

        #     transform = T2 @ T1
        # else:
        #     transform = np.eye(4)

        transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.radarposes = radarposes  # np.ndarray, (num_images, 4, 4)
        self.radar_ids = radar_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of radar_id -> K
        self.imsize_dict = imsize_dict  # Dict of radar_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of radar_id -> mask
        self.transform = transform  # np.ndarray, (4, 4)

        # size of the scene measured by cameras
        radar_locations = radarposes[:, :3, 3]
        scene_center = np.mean(radar_locations, axis=0)
        dists = np.linalg.norm(radar_locations - scene_center, axis=1)
        # Define the scene scale as the maximum distance from the center of the scene to the camera locations
        # plus the maximum sensing range of the sensor
        self.scene_scale = np.max(dists) + 2 * W * sensor['range_resolution']

        # For radar sensor 
        self.sensor_type = sensor_type
        self.use_polar = use_polar
        self.W = sensor['W']
        self.H = sensor['H']
        self.intermediate_azimuth_resolution = intermediate_azimuth_resolution
        self.W_metadata = W_metadata
        self.max_range = max_range
        self.range_resolution = range_resolution
        self.azimuth_resolution = azimuth_resolution
        self.azimuth_beamwidth = azimuth_beamwidth
        self.azimuth_coverage = azimuth_coverage
        self.scene_center = scene_center
        
        self.multipath_data = multipath_data

        # # Check cart_to_polar # TODO: move to unit test
        # from radar.utils import polar_to_cart, cart_to_polar
        # for index in range(len(self.image_paths)):
        #     image = imageio.imread(self.image_paths[index])
        #     image = image[:,self.W_metadata:]
        #     if self.max_range is not None:
        #         W = int(self.max_range / self.range_resolution)
        #         image = image[:,:W]
            
        #     image_np = image.copy()/255.
        #     image = torch.from_numpy(image).float()                  
            
        #     num_bins_to_show = image.shape[1]
        #     bin_size = self.range_resolution
        #     num_azims = int(360/self.azimuth_resolution)
        #     image_cart = polar_to_cart(image.detach().cpu().numpy()/255., num_bins_to_show, bin_size, num_azims,
        #             resolution=1000, noise_floor=None, norm=False)
            
        #     recovered_image_polar = cart_to_polar(image_cart, num_bins_to_show, bin_size, num_azims,
        #             resolution=1000, noise_floor=None, norm=False)
            
        #     fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))  # Create 1 row, 3 columns

        #     ax1.imshow(image_np, vmin=0.0, vmax=1.0)
        #     ax1.axis('off')
        #     ax2.imshow(image_cart, vmin=0.0, vmax=1.0)
        #     ax2.axis('off')
        #     ax3.imshow(recovered_image_polar, vmin=0.0, vmax=1.0)
        #     ax3.axis('off')
        #     ax4.imshow(np.abs(image_np-recovered_image_polar))
        #     ax4.axis('off')
        #     plt.show()

        
        # # Check RadarFields noise removal # TODO: move to unit test
        # import matplotlib.pyplot as plt
        # from radar.utils import polar_to_cart
        # for index in range(len(self.image_paths)):
        #     image = imageio.imread(self.image_paths[index])
        #     image = image[:,self.W_metadata:]
        #     if self.max_range is not None:
        #         W = int(self.max_range / self.range_resolution)
        #         image = image[:,:W]
                
        #     image = torch.from_numpy(image).float()
        #     image_thres = compute_spherical_grid_noise_threshold(image, 0, image.shape[1])
            
        #     img = torch.cat((image_thres, image),dim=0)/255.
        #     fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        #     ax.imshow(img.detach().cpu().numpy(), vmin=0.0, vmax=1.0)
        #     ax.axis('off') 
        #     plt.show()                   
            
        #     num_bins_to_show = image.shape[1]
        #     bin_size = self.range_resolution
        #     num_azims = int(360/self.azimuth_resolution)
        #     image_cart = polar_to_cart(image.detach().cpu().numpy()/255., num_bins_to_show, bin_size, num_azims,
        #             resolution=1000, noise_floor=None, norm=False)
        #     image_thres_cart = polar_to_cart(image_thres.detach().cpu().numpy()/255., num_bins_to_show, bin_size, num_azims,
        #             resolution=1000, noise_floor=None, norm=False)
            
        #     fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        #     cart_img = np.hstack((image_cart, image_thres_cart))
        #     ax.imshow(cart_img, vmin=0.0, vmax=1.0)
        #     ax.axis('off')
        #     plt.show()


class WaveSensorDataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: WaveSensorDataParser,
        split: Literal["train", "val", "all"] = "train",
    ):
        self.parser = parser
        self.split = split
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
            if len(indices)==1: # Single image. Set the image as train set.
                self.indices = indices
        elif split == "val":
            self.indices = indices[indices % self.parser.test_every == 0]
        elif split == "all":
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])
        
        # Load LiDAR
        synced_lidar_path = self.parser.image_paths[index].replace("images", "synced_lidar").replace("png", "pcd")
        pcd = o3d.io.read_point_cloud(synced_lidar_path)

        # Load LiDAR Map
        # synced_lidar_map_path = self.parser.image_paths[index].replace("images", "synced_lidar_map_win5").replace("png", "pcd")
        synced_lidar_map_path = self.parser.image_paths[index].replace("images", f'{self.parser.synced_lidar_map_name}').replace("png", "pcd")
        
        map_pcd = o3d.io.read_point_cloud(synced_lidar_map_path)

        # Load Radar Map # TODO load from args
        # radar_map_path = self.parser.image_paths[index].replace("images", "radar_average_map_polar/res:0.0596_dist:50_win_size:5_CR_thres:0.21")
        # radar_map_path = self.parser.image_paths[index].replace("images", "baseline_polar_occ_filtered_polar/res:0.0596_dist:50_win_size:0")
        
        # radar_map_path = self.parser.image_paths[index].replace("images", "radar_average_map_polar/res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:3.0")
        
        radar_map_path = self.parser.image_paths[index].replace("images", f'{self.parser.radar_average_map_name}')

        radar_map_image_polar = imageio.imread(radar_map_path)
        radar_map_image_polar = torch.from_numpy(radar_map_image_polar).float()/255.

        # Load Multipath Source
        multipath_sources_path = self.parser.image_paths[index].replace("images", "multipath_model/dist:50").replace("png", "npy")
        multipath_sources = np.load(multipath_sources_path, allow_pickle=True).item()
        
        if image.ndim==3 and image.shape[-1]==3:
            # The data is in rgb. Convert to gray-scaled
            image = image[:,:,0]

        metadata = image[:,:self.parser.W_metadata]
        image = image[:,self.parser.W_metadata:]
        
        if self.parser.max_range is not None:
            W = int(self.parser.max_range / self.parser.range_resolution)
            image = image[:,:W]
        
        image = torch.from_numpy(image).float()
        
        # Apply dynamic threshold
        image_thres = compute_spherical_grid_noise_threshold(image, 0, image.shape[1])

        radar_id = self.parser.radar_ids[index]
        K = self.parser.Ks_dict[radar_id].copy()  # undistorted K
        radarposes = self.parser.radarposes[index]
        mask = self.parser.mask_dict[radar_id]

        data = {
            "K": torch.from_numpy(K).float(),
            "radarpose": torch.from_numpy(radarposes).float(),
            "image": image,
            "image_thres": image_thres,
            "image_id": item,  # the index of the image in the dataset
            "synced_lidar": np.asarray(pcd.points),
            "synced_lidar_map": np.asarray(map_pcd.points),
            "preprocess_radar_map_polar": radar_map_image_polar,
            "multipath_sources": multipath_sources,
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/radar/boreas_test")
    parser.add_argument("--factor", type=int, default=1)
    args = parser.parse_args()

    # Parse data.
    parser = WaveSensorDataParser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = WaveSensorDataset(parser, split="train")
    print(f"Dataset: {len(dataset)} images.")