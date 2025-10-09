import imageio.v2 as imageio
import numpy as np
from radar.utils import polar_to_cart, cart_to_polar
import glob
import os
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default='/mnt/ws-frb/projects/radar_splat/data/wave_gs/', help="Boreas data root")
parser.add_argument("--seq_name", default='boreas-2021-09-02-11-42', help="Boreas seq name")
parser.add_argument("--radar_avg_map", default='res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:3.0', help="Selected radar avg map")
parser.add_argument("--resolution", type=float, default=0.0596, help="Resolution")

args = parser.parse_args()

radar_map_path = args.data_root + '/' + args.seq_name + '/radar_average_map/' + args.radar_avg_map

range_resolution = args.resolution
max_dist = 50
num_azims = 400

num_bins_to_show = int(max_dist/range_resolution)

image_files = sorted(glob.glob(os.path.join(radar_map_path, "*")))

radar_map_polar_path = radar_map_path.replace("radar_average_map","radar_average_map_polar")
os.makedirs(radar_map_polar_path, exist_ok=True)

for i, image_file in tqdm(enumerate(image_files)):
    # if i < 380:
    #     continue
    radar_map_image = imageio.imread(image_file)
    radar_map_image_ = np.rot90(radar_map_image, k=3)
    radar_map_image_polar = cart_to_polar(radar_map_image_, num_bins_to_show=num_bins_to_show, bin_size=range_resolution, num_azims=num_azims,
                                                resolution=radar_map_image_.shape[0], noise_floor=None, norm=False)
    radar_map_image_polar = radar_map_image_polar.astype(np.uint8)
    # print(radar_map_image_polar.shape, radar_map_image_polar.dtype)
    polar_image_file = image_file.replace("radar_average_map","radar_average_map_polar")
    print(polar_image_file)
    imageio.imwrite(polar_image_file, radar_map_image_polar)
