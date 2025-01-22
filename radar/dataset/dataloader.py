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

from PIL import Image
import yaml

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
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(image_path)
                images.append(img)
                image_names.append(os.path.splitext(filename)[0])
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return images, image_names

class WaveSensorDataParser:
    """Wave-based sensor data parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1, # not used
        normalize: bool = False, # not used
        test_every: int = 8,
        intermediate_azimuth_resolution: float = 0.1, # Only used when using scanning radar & use_polar=True
        max_range: float = None, # Only used when using scanning radar & use_polar=True
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        assert os.path.exists(
            data_dir
        ), f"Data directory {data_dir} does not exist."

        # Load poses data from the .tum file
        poses_file = os.path.join(data_dir, "traj.tum")
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
        image_paths = [os.path.join(image_dir, item) for item in os.listdir(image_dir)]

        # Extract extrinsic matrices in world-to-camera format.
        imdata, image_names = load_images_from_folder(image_dir)
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        for k, im in enumerate(imdata):
            w2c = poses[k]
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = 0 # use the same camera intrinsics for all images
            camera_ids.append(camera_id)

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
            
            Ks_dict[camera_id] = K
            assert sensor['H']==im.height and sensor['W']==im.width, "Image size does not match sensor size."
            imsize_dict[camera_id] = (im.width // factor, H) # apply factor to image width (range) only
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in data folder.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

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
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.transform = transform  # np.ndarray, (4, 4)

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        # Define the scene scale as the maximum distance from the center of the scene to the camera locations
        # plus the maximum sensing range of the sensor
        self.scene_scale = np.max(dists) + 2 * W * sensor['range_resolution']

        # For radar sensor 
        self.sensor_type = sensor_type
        self.use_polar = use_polar
        self.intermediate_azimuth_resolution = intermediate_azimuth_resolution
        self.W_metadata = W_metadata
        self.max_range = max_range
        self.range_resolution = range_resolution
        self.azimuth_resolution = azimuth_resolution
        self.azimuth_beamwidth = azimuth_beamwidth
        self.azimuth_coverage = azimuth_coverage



class WaveSensorDataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: WaveSensorDataParser,
        split: str = "train",
    ):
        self.parser = parser
        self.split = split
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
            if len(indices)==1: # Single image. Set the image as train set.
                self.indices = indices
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])
        
        if image.ndim==3 and image.shape[-1]==3:
            # The data is in rgb. Convert to gray-scaled
            image = image[:,:,0]

        metadata = image[:,:self.parser.W_metadata]
        image = image[:,self.parser.W_metadata:]
        
        if self.parser.max_range is not None:
            W = int(self.parser.max_range / self.parser.range_resolution)
            image = image[:,:W]

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
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