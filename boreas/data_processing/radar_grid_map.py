import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.transforms as mtf
from matplotlib.colors import Normalize
import numpy as np

from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train
from pyboreas.utils.utils import get_inverse_tf

import os
import cv2
from pyboreas.utils.utils import get_T_bev_metric
import open3d as o3d
import copy
from tqdm import tqdm

import argparse

from play_radar_signal import FFT, apply_saturation_mask, radarfields_occupancy, multipath_modeling

parser = argparse.ArgumentParser(description="Boreas Data Visualization")
parser.add_argument("--data_root", default='/mnt/ws-frb/projects/radar_splat/data/boreas/', help="Boreas data root")
parser.add_argument("--seq_name", default='boreas-2021-09-02-11-42', help="Boreas seq name")
parser.add_argument("--save_radar_grid_map", action="store_true", help="whether to save grid map")
parser.add_argument("--radarfields", action="store_true", help="whether to save radarfields map")
parser.add_argument("--start_index", type=int, default=0, help="index to start viz/save")
parser.add_argument("--end_index", type=int, default=99999999, help="Only save first N frames")

parser.add_argument("--resolution", type=float, default=0.0596, help="resolution")

parser.add_argument("--window_size", type=int, default=0, help="lidar map window size")

parser.add_argument("--viz", action="store_true", help="viz")

args = parser.parse_args()

# split=[['boreas-objects-v1']]
# split = [['boreas-2021-03-09-14-23']] # Testing Slice. No sensor pose.
# split = [['boreas-2021-09-02-11-42']]
split = [[args.seq_name]]

# bd = BoreasDataset(args.data_root, split=obj_train, verbose=True)
bd = BoreasDataset(args.data_root, split=split, verbose=True, labelFolder='labels_detection')

seq = bd.sequences[0] # 36  ## ** Try different sequences!
if split[0][0] == "boreas-objects-v1":
    seq.filter_frames_gt()

sync_sensor = 'radar'
current_index = args.start_index

print(f'<< Filter_frames_gt & sync with {sync_sensor} >>')
seq.synchronize_frames(sync_sensor)
print(len(seq.camera_frames))
print(len(seq.lidar_frames))
print(len(seq.radar_frames))

# Define the folder containing images
image_folder = f"/mnt/ws-frb/projects/radar_splat/data/boreas/{split[0][0]}/camera"
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

data_len = len(seq.radar_frames)

# Data configs
range_resolution = args.resolution

resolution = args.resolution #0.0596 #0.25
max_dist = 50 #75

# Saturation detection
constant_ratio_thres = 0.21 #0.23 #0.21

# Multipath detection
skip_first_n_freq = 3
magnitude_thres = 30
constant_ratio_thres_=0.2

# Noise revomal
smoothing_sigma = 3.0

width = int(max_dist*2 / resolution)
range_pixels = int(max_dist/resolution)

class RadarIntensityMap:
    def __init__(self, img_size=(600, 600), max_range=75.0, resolution=0.25):
        """
        Initialize the radar map parameters.
        :param img_size: (height, width) of the radar BEV images.
        :param max_range: Maximum sensing distance in meters.
        :param resolution: Grid map resolution in meters per pixel.
        """
        self.img_size = img_size
        self.max_range = max_range
        self.resolution = resolution
        
        # Compute world coordinate limits
        self.grid_size = int(2 * max_range / resolution)
        self.global_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.global_count = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        self.global_grid_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * 0.5
    
    def _image_to_world(self, u, v):
        """ Convert image pixel (u, v) to world coordinates (x, y). """
        cx, cy = self.img_size[1] // 2, self.img_size[0] // 2  # Image center
        x = (u - cx) * self.resolution
        y = (cy - v) * self.resolution  # Flipping y-axis
        return x, y

    def _world_to_grid(self, x, y):
        """ Convert world coordinates (x, y) to grid indices. """
        gx = ((x + self.max_range) / self.resolution).astype(int)
        gy = ((self.max_range - y) / self.resolution).astype(int)  # Flip y-axis for grid indexing
        return gx, gy

    def add_radar_image(self, radar_img, pose):
        """
        Add a radar BEV image to the global intensity map.
        :param radar_img: 2D numpy array (600x600) with intensity values.
        :param pose: (x, y, theta) in meters and radians.
        """
        x_pose_, y_pose_, theta_ = pose  # Radar position and orientation
        
        x_pose, y_pose, theta = y_pose_, x_pose_, -theta_ 

        h, w = radar_img.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x_world, y_world = self._image_to_world(u, v)

        # Apply transformation: rotate and translate
        x_trans = x_world * np.cos(theta) - y_world * np.sin(theta) + x_pose
        y_trans = x_world * np.sin(theta) + y_world * np.cos(theta) + y_pose

        # Convert to global grid indices
        valid_mask = (x_trans >= -self.max_range) & (x_trans < self.max_range) & \
                     (y_trans >= -self.max_range) & (y_trans < self.max_range)

        x_trans, y_trans, intensities = x_trans[valid_mask], y_trans[valid_mask], radar_img[valid_mask]
        gx, gy = self._world_to_grid(x_trans, y_trans)

        # Filter valid grid indices
        valid_grid_mask = (gx >= 0) & (gx < self.grid_size) & (gy >= 0) & (gy < self.grid_size)
        gx, gy, intensities = gx[valid_grid_mask], gy[valid_grid_mask], intensities[valid_grid_mask]

        np.add.at(self.global_map, (gy, gx), intensities)
        np.add.at(self.global_count, (gy, gx), 1)
    
    def add_occ_image(self, occ_img, pose):
        """
        Add a radar occ image to the global intensity map.
        :param occ_img: 2D numpy array (600x600) with occupancy values.
        :param pose: (x, y, theta) in meters and radians.
        """
        x_pose_, y_pose_, theta_ = pose  # Radar position and orientation
        
        x_pose, y_pose, theta = y_pose_, x_pose_, -theta_ 

        h, w = occ_img.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x_world, y_world = self._image_to_world(u, v)

        # Apply transformation: rotate and translate
        x_trans = x_world * np.cos(theta) - y_world * np.sin(theta) + x_pose
        y_trans = x_world * np.sin(theta) + y_world * np.cos(theta) + y_pose

        # Convert to global grid indices
        valid_mask = (x_trans >= -self.max_range) & (x_trans < self.max_range) & \
                     (y_trans >= -self.max_range) & (y_trans < self.max_range)

        x_trans, y_trans, intensities = x_trans[valid_mask], y_trans[valid_mask], occ_img[valid_mask]
        gx, gy = self._world_to_grid(x_trans, y_trans)

        # Filter valid grid indices
        valid_grid_mask = (gx >= 0) & (gx < self.grid_size) & (gy >= 0) & (gy < self.grid_size)
        gx, gy, intensities = gx[valid_grid_mask], gy[valid_grid_mask], intensities[valid_grid_mask]
        
        current_probs = self.global_grid_map[gy, gx]
        updated_probs = (intensities * current_probs) / ((intensities * current_probs) + (1 - intensities) * (1 - current_probs)+1e-7)
        self.global_grid_map[gy, gx] = updated_probs

    def get_average_map(self):
        """ Compute the final averaged intensity map. """
        avg_map = np.divide(self.global_map, self.global_count, where=self.global_count > 0)
        avg_map[self.global_count == 0] = np.nan  # Mark empty areas as NaN
        return avg_map
    
    def get_occ_grid_map(self):
        return self.global_grid_map

    def plot_map(self):
        """ Plot the final averaged intensity map. """
        avg_map = self.get_average_map()
        plt.figure(figsize=(8, 8))
        plt.imshow(avg_map, extent=[-self.max_range, self.max_range, -self.max_range, self.max_range], cmap='jet')
        plt.colorbar(label="Intensity")
        plt.xlim(-75,75)
        plt.ylim(-75,75)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Averaged Radar Intensity Map")
        plt.show(block=True)

def transform_matrix_to_2d_pose(T):
    """
    Convert a 4x4 transformation matrix to (x, y, theta).
    :param T: 4x4 transformation matrix (NumPy array)
    :return: (x, y, theta) where x, y are in meters and theta is in radians
    """
    x = T[0, 3]  # Extract X translation
    y = T[1, 3]  # Extract Y translation
    theta = np.arctan2(T[1, 0], T[0, 0])  # Extract yaw angle

    return x, y, theta

def radar_grid_map(index, save=False, viz=True, radarfields=False):
    """Display the image at the given index."""
    cam = seq.get_camera(index)
    lid = seq.get_lidar(index)
    rad = seq.get_radar(index)
    T_enu_lidar = lid.pose
    T_enu_radar = rad.pose
    T_radar_lidar = np.matmul(get_inverse_tf(T_enu_radar), T_enu_lidar)
    
    # Initialize the intensity map with default settings
    radar_map = RadarIntensityMap(img_size=(width, width), max_range=max_dist, resolution=resolution)
    
    baseline_radar_map = RadarIntensityMap(img_size=(width, width), max_range=max_dist, resolution=resolution)
    
    origin = T_enu_radar

    min_index = max(index-args.window_size, 0)
    max_index = min(index+args.window_size+1, data_len)
    for local_index in range(min_index, max_index):        
        print('index: ', local_index)
        rad_ = seq.get_radar(local_index)
        T_enu_radar_ = rad_.pose
        T_local = np.linalg.inv(origin) @ T_enu_radar_
        x, y, theta = transform_matrix_to_2d_pose(T_local)      
        
        # Saturation azimuth detection
        polar_fft_log, polar_fft, fft_result, fft_freq = FFT(rad_.polar[:,:range_pixels], range_resolution=range_resolution)
        constant_ratio = polar_fft[:,:polar_fft.shape[1]//2][:,0]/polar_fft[:,:polar_fft.shape[1]//2].sum(axis=1)
        saturate_azi_mask = constant_ratio>constant_ratio_thres

        # Multipath azimuth detection        
        # fft_magnitude are symetric. use left-half to find maximum index
        fft_magnitude_half = polar_fft[:,skip_first_n_freq:polar_fft.shape[1]//2]
        # use magnitude threshold and constant ratio threshold to find azimuth with multipath
        magnitude_thres_azi_mask = np.max(fft_magnitude_half, axis=1) > magnitude_thres
        constant_ratio_thres_azi_mask = constant_ratio > constant_ratio_thres_
        # Get multipath azimuth mask. (Mask of selected azimuth that include multipath)
        multipath_azi_mask = (magnitude_thres_azi_mask * constant_ratio_thres_azi_mask)

        noise_azi_mask = saturate_azi_mask + multipath_azi_mask
        polar_image_filtered, polar_occ_filtered = apply_saturation_mask(rad_.polar[:,:range_pixels], noise_azi_mask, smoothing_sigma=smoothing_sigma, occ_thres=0.15)
       
        if radarfields:
            baseline_polar_occ_filtered = radarfields_occupancy(rad_.polar[:,:range_pixels])
        
        # Filtered Cart Image
        cart_polar_image_filtered = rad.polar_to_cart(
            cart_resolution=resolution,
            cart_pixel_width=width,
            polar=polar_image_filtered, #polar_image_saturation_detection,
            in_place=False,
        )
        # Filtered Occ Cart Image
        cart_occ_filtered = rad.polar_to_cart(
            cart_resolution=resolution,
            cart_pixel_width=width,
            polar=polar_occ_filtered,
            in_place=False,
        )
        if radarfields:
            # RF Filtered Occ Cart Image
            baseline_cart_occ_filtered = rad.polar_to_cart(
                cart_resolution=resolution,
                cart_pixel_width=width,
                polar=baseline_polar_occ_filtered,
                in_place=False,
            )
        pose = (x, y, theta)  # (x, y, theta)
        
        radar_map.add_radar_image(cart_polar_image_filtered, pose)
        radar_map.add_occ_image(cart_occ_filtered, pose)
        if radarfields:
            baseline_radar_map.add_occ_image(baseline_cart_occ_filtered, pose)

        # radar_map.plot_map()
        del rad_

    ### Target frame
    # Raw Cart image
    cart = rad.polar_to_cart(
        cart_resolution=resolution,
        cart_pixel_width=width,
        in_place=False)

    # Aggregate map
    avg_map = radar_map.get_average_map()
    process_image = avg_map
    occ_grid_map = radar_map.get_occ_grid_map()
    if radarfields:
        baseline_occ_grid_map = baseline_radar_map.get_occ_grid_map()

    # Saturation azimuth detection
    polar_fft_log, polar_fft, fft_result, fft_freq = FFT(rad.polar[:,:range_pixels], range_resolution=range_resolution)
    constant_ratio = polar_fft[:,:polar_fft.shape[1]//2][:,0]/polar_fft[:,:polar_fft.shape[1]//2].sum(axis=1)
    saturate_azi_mask = constant_ratio>constant_ratio_thres

    # Multipath azimuth detection        
    # fft_magnitude are symetric. use left-half to find maximum index
    fft_magnitude_half = polar_fft[:,skip_first_n_freq:polar_fft.shape[1]//2]
    # use magnitude threshold and constant ratio threshold to find azimuth with multipath
    magnitude_thres_azi_mask = np.max(fft_magnitude_half, axis=1) > magnitude_thres
    constant_ratio_thres_azi_mask = constant_ratio > constant_ratio_thres_
    # Get multipath azimuth mask. (Mask of selected azimuth that include multipath)
    multipath_azi_mask = (magnitude_thres_azi_mask * constant_ratio_thres_azi_mask)
    
    # Multipath modeling
    azi_id_list, range_id_list, \
    freq_positive, amplitudes_positive, phases_positive, \
    estimated_A, estimated_alpha, \
    reconstructed_signal, ifft_recovered_signal = multipath_modeling(polar_fft, fft_result, fft_freq, multipath_azi_mask, rad.polar, skip_first_n_freq, range_resolution, range_pixels)
    # Get theta and d of multipath sources
    source_theta = rad.azimuths[azi_id_list].reshape(-1)
    source_d = range_id_list * range_resolution
    # Convert polar to Cartesian coordinates
    source_x = source_d * np.cos(source_theta)
    source_y = source_d * np.sin(source_theta)

    # Save
    if save:

        save_path = f"{args.data_root}/{split[0][0]}/radar_raw/res:{resolution}_dist:{max_dist}"
        os.makedirs(save_path, exist_ok=True)
        image_uint8 = (cart * 255).astype(np.uint8)
        cv2.imwrite(save_path+f"/{rad.timestamp_micro}.png", image_uint8)

        save_path = f"{args.data_root}/{split[0][0]}/radar_average_map/res:{resolution}_dist:{max_dist}_win_size:{args.window_size}_CR_thres:{constant_ratio_thres:.2f}_smooth:{smoothing_sigma}"
        os.makedirs(save_path, exist_ok=True)
        image_uint8 = (avg_map * 255).astype(np.uint8)
        cv2.imwrite(save_path+f"/{rad.timestamp_micro}.png", image_uint8)

        save_path = f"{args.data_root}/{split[0][0]}/radar_occ_map/res:{resolution}_dist:{max_dist}_win_size:{args.window_size}_CR_thres:{constant_ratio_thres:.2f}_smooth:{smoothing_sigma}"
        os.makedirs(save_path, exist_ok=True)
        image_uint8 = (occ_grid_map * 255).astype(np.uint8)
        cv2.imwrite(save_path+f"/{rad.timestamp_micro}.png", image_uint8)

        save_path = f"{args.data_root}/{split[0][0]}/multipath_model/dist:{max_dist}"
        os.makedirs(save_path, exist_ok=True)
        multipath_dict = {
            'azi_id_list': azi_id_list,
            'range_id_list': range_id_list,
            'freq_positive': freq_positive,
            'amplitudes_positive': amplitudes_positive,
            'phases_positive': phases_positive,
            'reconstructed_signal': reconstructed_signal,
            'ifft_recovered_signal': ifft_recovered_signal,
            'estimated_A': estimated_A,
            'estimated_alpha': estimated_alpha,
        }
        np.save(save_path+f"/{rad.timestamp_micro}.npy", multipath_dict)

        if radarfields:
            save_path = f"{args.data_root}/{split[0][0]}/baseline_radar_occ_map/res:{resolution}_dist:{max_dist}_win_size:{args.window_size}_delta:10"
            os.makedirs(save_path, exist_ok=True)
            image_uint8 = (baseline_occ_grid_map * 255).astype(np.uint8)
            cv2.imwrite(save_path+f"/{rad.timestamp_micro}.png", image_uint8)

            save_path = f"{args.data_root}/{split[0][0]}/baseline_polar_occ_filtered_polar/res:{resolution}_dist:{max_dist}_win_size:{args.window_size}_delta:10"
            os.makedirs(save_path, exist_ok=True)
            image_uint8 = (baseline_polar_occ_filtered * 255).astype(np.uint8)
            cv2.imwrite(save_path+f"/{rad.timestamp_micro}.png", image_uint8)
    
    if viz:
        cart = rad.polar_to_cart(
            cart_resolution=resolution,
            cart_pixel_width=width,
            in_place=False,
        )
        # Radar
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(cart, cmap="gray")
        ax1.axis("off")  # Remove axes for cleaner display
        ax1.set_title(f"Radar {index + 1}/{data_len}: {rad.timestamp}")
        # Show multipath sources
        ax1.scatter(source_y/resolution + width/2, -source_x/resolution + width/2, s=5, c='lime')
        
        # Radar + LiDAR
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(cart, cmap="gray")
        ax2.axis("off")  # Remove axes for cleaner display
        # ax2.set_title(f"Image {index + 1}/{data_len}: {rad.timestamp}")

        lid.transform(T_radar_lidar)
        bounds = [-(max_dist-1), (max_dist-1), -(max_dist-1), (max_dist-1), -5, 10] # xmin, xmax, ymin, ymax, zmin, zmax
        lid.passthrough(bounds)
        # ax2.scatter(lid.points[:, 1]/resolution + width/2, -lid.points[:, 0]/resolution + width/2, s=0.01, c=lid.points[:, 2], vmin=-5, vmax=10, cmap='jet')
        
        pointcloud_undistorted = copy.deepcopy(lid.remove_motion(lid.body_rate))[:,:3]
        # pcd_undis = o3d.geometry.PointCloud()
        # pcd_undis.points = o3d.utility.Vector3dVector(pointcloud_undistorted)
        # pcd_undis.transform(T_radar_lidar)
        ax2.scatter(pointcloud_undistorted[:, 1]/resolution + width/2, -pointcloud_undistorted[:, 0]/resolution + width/2, s=0.01, c=pointcloud_undistorted[:, 2], vmin=-5, vmax=10, cmap='jet')

        # Radar Grid Map
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(process_image, extent=[-max_dist, max_dist, -max_dist, max_dist], cmap='jet')
        # ax3.colorbar(label="Intensity")
        ax3.set_xlim(-max_dist,max_dist)
        ax3.set_ylim(-max_dist,max_dist)
        ax3.set_xlabel("X (m)")
        ax3.set_xlabel("Y (m)")
        ax3.set_title("Averaged Radar Intensity Map")
        ax3.axis("off")
        
        # Thresholded Radar Grid Map
        ax4 = plt.subplot(2, 3, 4)
        cart_occ_map = process_image
        cart_occ_map[cart_occ_map>=0.15]=1
        cart_occ_map[cart_occ_map<0.15]=0
        ax4.imshow(cart_occ_map, extent=[-max_dist, max_dist, -max_dist, max_dist], cmap='gray')
        ax4.set_xlim(-max_dist,max_dist)
        ax4.set_ylim(-max_dist,max_dist)
        ax4.set_xlabel("X (m)")
        ax4.set_xlabel("Y (m)")
        ax4.set_title("Thres Averaged Map")
        ax4.axis("off")
        
        pointcloud_wrt_radar = lid.points
        r = np.linalg.norm(pointcloud_wrt_radar[:,:2],axis=1)
        z = pointcloud_wrt_radar[:,2]
        phi = np.arctan2(z,r)
        # phi = np.abs(phi)
        show_mask1 = phi < 1.8*2/180*np.pi
        show_mask2 = phi > -1.8/180*np.pi
        pointcloud_wrt_radar = pointcloud_wrt_radar[show_mask1*show_mask2]
        floor_mask = pointcloud_wrt_radar[:,2]<1.0
        pointcloud_wrt_radar = pointcloud_wrt_radar[floor_mask]
        ax4.scatter(pointcloud_wrt_radar[:, 1], pointcloud_wrt_radar[:, 0], s=0.1, c='r') # , vmin=-5, vmax=10, cmap='jet'
        
        ax5 = plt.subplot(2, 3, 5)
        ax5.imshow(occ_grid_map, extent=[-max_dist, max_dist, -max_dist, max_dist], cmap='gray')
        ax5.set_xlim(-max_dist,max_dist)
        ax5.set_ylim(-max_dist,max_dist)
        ax5.set_xlabel("X (m)")
        ax5.set_xlabel("Y (m)")
        ax5.set_title("Occ Grid Map")
        ax5.axis("off")
        ax5.scatter(pointcloud_wrt_radar[:, 1], pointcloud_wrt_radar[:, 0], s=0.1, c='r')
        if radarfields:
            ax6 = plt.subplot(2, 3, 6)
            ax6.imshow(baseline_occ_grid_map, extent=[-max_dist, max_dist, -max_dist, max_dist], cmap='gray')
            ax6.set_xlim(-max_dist,max_dist)
            ax6.set_ylim(-max_dist,max_dist)
            ax6.set_xlabel("X (m)")
            ax6.set_xlabel("Y (m)")
            ax6.set_title("RF Occ Grid Map")
            ax6.axis("off")
 
        plt.tight_layout()
        plt.show(block=True)

    del rad
    del radar_map
    del baseline_radar_map

def on_key_radar(event):
    """Handle key press events."""
    global current_index
    data_len = len(seq.lidar_frames)
    if event.key == "right":  # Next image
        current_index = (current_index + 1) % data_len
    elif event.key == "left":  # Previous image
        current_index = (current_index - 1) % data_len
    elif event.key == "escape":  # Exit interaction
        print("Exiting...")
        return
    plt.clf()  # Clear the current figure
    radar_grid_map(current_index)

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.save_radar_grid_map:
        for i in tqdm(range(data_len)[:args.end_index][args.start_index:]):
            radar_grid_map(i, save=True, viz=args.viz, radarfields=args.radarfields)
    else:
        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', on_key_radar)
        radar_grid_map(current_index, viz=args.viz, radarfields=args.radarfields)
    