'''
Visualize radar and lidar BEV point could in interactive mode.
Note this visualization only neither use extrinsic nor consider temporal sync. 
'''

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
print(matplotlib.get_backend())
import matplotlib.transforms as mtf
import numpy as np

from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train
from pyboreas.utils.utils import get_inverse_tf

import os
import cv2
from pyboreas.utils.utils import get_T_bev_metric


root = '/mnt/ws-frb/projects/radar_splat/data/boreas/'
split=[['boreas-objects-v1']]
# bd = BoreasDataset(root, split=obj_train, verbose=True)

bd = BoreasDataset(root, split=split, verbose=True, labelFolder='labels_detection')

seq = bd.sequences[0] # 36  ## ** Try different sequences!
seq.filter_frames_gt()
seq.synchronize_frames('lidar')

print('--filter_frames_gt & sync with lidar--')
print(len(seq.camera_frames))
print(len(seq.lidar_frames))
print(len(seq.radar_frames))

# Define the folder containing images
image_folder = "/mnt/ws-frb/projects/radar_splat/data/boreas/boreas-objects-v1/camera"
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# data_len = len(image_files)
data_len = len(seq.lidar_frames)
    
# Initialize index to keep track of the current image
current_index = 0

def display_image(index):
    """Display the image at the given index."""
    img_path = os.path.join(image_folder, image_files[index])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image {index + 1}/{len(image_files)}: {image_files[index]}")
    plt.show()

def display_radar(index):
    """Display the image at the given index."""
    print('index: ', index)
    cam = seq.get_camera(index)
    lid = seq.get_lidar(index)
    rad = seq.get_radar(index)
    
    bounds = [-75, 75, -75, 75, -5, 10] # xmin, xmax, ymin, ymax, zmin, zmax
    # bbs = lid.get_bounding_boxes()
    # bbs.filter_empty()
    # bbs.passthrough(bounds)

    T_enu_camera = cam.pose
    T_enu_lidar = lid.pose
    T_enu_radar = rad.pose

    T_camera_lidar = np.matmul(get_inverse_tf(T_enu_camera), T_enu_lidar)
    T_radar_lidar = np.matmul(get_inverse_tf(T_enu_radar), T_enu_lidar)

    # bbs = rad.get_bounding_boxes(seq.labelFiles, seq.labelTimes, seq.labelPoses)
    # bbs.passthrough(bounds)
    # T_bev_metric = get_T_bev_metric(resolution, width)
    # bbs.transform(T_bev_metric)
    
    resolution = 0.25 # 0.2384
    max_dist = 75
    width = int(max_dist*2 / resolution) # 640
    cart = rad.polar_to_cart(
        cart_resolution=resolution,
        cart_pixel_width=width,
        in_place=False,
    )
    
    # Radar
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(cart, cmap="gray")
    ax1.axis("off")  # Remove axes for cleaner display
    ax1.set_title(f"Radar {index + 1}/{data_len}: {rad.timestamp}")
    
    # Radar + LiDAR
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(cart, cmap="gray")
    ax2.axis("off")  # Remove axes for cleaner display
#     ax2.set_title(f"Image {index + 1}/{data_len}: {rad.timestamp}")
    bounds = [-(max_dist-1), (max_dist-1), -(max_dist-1), (max_dist-1), 0, 10] # xmin, xmax, ymin, ymax, zmin, zmax
    lid.passthrough(bounds)
    ax2.scatter(-lid.points[:, 1]/resolution + width/2, -lid.points[:, 0]/resolution + width/2, s=0.01, c=lid.points[:, 2], vmin=-5, vmax=10, cmap='jet')
    
    # Camera + LiDAR
    cam = seq.get_camera(index)
    lid = seq.get_lidar(index)
    lid.transform(T_camera_lidar)
    lid.passthrough([-max_dist, max_dist, -20, 10, 0, max_dist])
    uv, colors, _ = lid.project_onto_image(seq.calib.P0)
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(cam.img)
    ax3.set_xlim(0, 2448)
    ax3.set_ylim(2048, 0)
#     ax3.scatter(uv[:, 0], uv[:, 1], c=colors, marker=',', s=0.1, edgecolors='none', alpha=0.7, cmap='jet')
    ax3.set_axis_off()

#     ax_radar = rad.visualize(show=False, cart_resolution=resolution, cart_pixel_width=width)
#     ax_radar.scatter(-lid.points[:, 1]/resolution + width/2, -lid.points[:, 0]/resolution + width/2, s=0.01, c=lid.points[:, 2], vmin=-5, vmax=10, cmap='jet')
#     ax_radar.set_xlim(0, width)
#     ax_radar.set_ylim(0, width)

    # bbs.render_2d(ax_radar)
    
    plt.tight_layout()
    plt.show()

def on_key(event):
    """Handle key press events."""
    global current_index
#     data_len = len(image_files)
    data_len = len(seq.lidar_frames)
    if event.key == "right":  # Next image
        current_index = (current_index + 1) % data_len
    elif event.key == "left":  # Previous image
        current_index = (current_index - 1) % data_len
    elif event.key == "escape":  # Exit interaction
        print("Exiting...")
        return

    plt.clf()  # Clear the current figure
#     display_image(current_index)
    display_radar(current_index)

fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', on_key)

# Display the first image
# display_image(current_index)
display_radar(current_index)

# Connect the event to the figure
# fig = plt.gcf()
# fig = plt.figure()
# fig.canvas.mpl_connect('key_press_event', on_key)

# plt.ion()  # Enable interactive mode
# plt.show(block=True)