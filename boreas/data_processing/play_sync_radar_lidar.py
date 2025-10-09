import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg') # when running from remote
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

import argparse

parser = argparse.ArgumentParser(description="Boreas Data Visualization")
parser.add_argument("--data_root", default='/mnt/ws-frb/projects/radar_splat/data/boreas/', help="Boreas data root")
parser.add_argument("--seq_name", default='boreas-2021-09-02-11-42', help="Boreas seq name")
parser.add_argument("--viz_plt_image", action="store_true", help="viz image")
parser.add_argument("--viz_plt_radar", action="store_true", help="viz radar")
parser.add_argument("--viz_o3d", action="store_true", help="viz pointcloud with o3d")
parser.add_argument("--save_sync_lidar", action="store_true", help="path to save synced lidar data")
parser.add_argument("--start_index", type=int, default=0, help="index to start viz/save")
parser.add_argument("--end_index", type=int, default=99999999, help="Only save first N frames")
parser.add_argument("--window_size", type=int, default=0, help="lidar map window size")

args = parser.parse_args()

# split=[['boreas-objects-v1']]
# split = [['boreas-2021-03-09-14-23']] # Testing Slice. No sensor pose.
# split = [['boreas-2021-09-02-11-42']]
split = [[args.seq_name]]

# bd = BoreasDataset(args.data_root, split=obj_train, verbose=True)
bd = BoreasDataset(args.data_root, split=split, verbose=True, labelFolder='labels_detection')

seq = bd.sequences[0] # 36  # Try different sequences!
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

# data_len = len(image_files)
data_len = len(seq.lidar_frames)

def vis_lidar_bbs(index):
    lid = seq.get_lidar(index)
    T = lid.pose
    
    o3d_vis=[]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
    o3d_vis.append(frame)
    
    bbs = lid.get_bounding_boxes()
    bbs.filter_empty()
    
    for bb in bbs.bbs:
        obb = o3d.geometry.OrientedBoundingBox(center=bb.pos.reshape(-1), R=bb.rot, extent=bb.extent.reshape(-1))
        obb.color = (0, 1, 0)
        o3d_vis.append(obb)
    
    pointcloud = copy.deepcopy(lid).points[:,:3]
    
    # pointcloud_undistorted = copy.deepcopy(lid.remove_motion(lid.body_rate))[:,:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.transform(get_inverse_tf(T))
    o3d_vis.append(pcd)
    
    o3d.visualization.draw_geometries(o3d_vis)

# TODO get_local_radar_map

def get_local_lidar_map(index, window_size=5):
    
    min_index = index-window_size if index-window_size>0 else 0
    max_index = index+window_size if index+window_size<len(seq.lidar_frames) else len(seq.lidar_frames)-1

    indice_list = range(min_index, max_index, 1)
    
    lid = seq.get_lidar(index)
    T_target = lid.pose
    
    o3d_vis=[]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
    o3d_vis.append(frame)
    
    for i in indice_list:
    
        lid = seq.get_lidar(i)
        bbs = lid.get_bounding_boxes()
        bbs.filter_empty()
        
        for bb in bbs.bbs:
            obb = o3d.geometry.OrientedBoundingBox(center=bb.pos.reshape(-1), R=bb.rot, extent=bb.extent.reshape(-1))
            obb.color = (0, 1, 0)
            o3d_vis.append(obb)
        
        # # Find all points inside any bounding box
        # inside_mask = np.zeros(len(pcd_points), dtype=bool)
        # for bb in bbs:
        #     inside_mask |= is_point_in_bbox(pcd_points, bb)

        T_current = lid.pose
        T_local = np.matmul(get_inverse_tf(T_target), T_current)
        
        # pointcloud = copy.deepcopy(lid).transform(T_local)[:,:3]
        pointcloud_undistorted = copy.deepcopy(lid.remove_motion(lid.body_rate))[:,:3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud_undistorted)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.transform(T_local)
        o3d_vis.append(pcd)
        
        o3d.visualization.draw_geometries(o3d_vis)
    

def image_to_pointcloud(image, resolution, vmax=0.4, vmin=0.1):
    # Generate the 3D point cloud
    height, width = image.shape
    x = np.arange(0, width) * resolution
    y = np.arange(0, height) * resolution
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)  # Assume all points lie on a flat plane (z = 0)

    # Flatten arrays
    points = np.vstack((yy.ravel(), xx.ravel(), zz.ravel())).T # radar frame: x -> front | y -> right
    points[:,0] = np.max(points[:,0]) - points[:,0]
    
    points[:,0] -= width/2. * resolution
    points[:,1] -= height/2. * resolution
    
    # Get show with power threshold
    show_mask = image.reshape(-1) > vmin  # Flatten powers
    
    # Normalize the image to the colormap bounds
    norm = Normalize(vmin=vmin, vmax=vmax)
    normalized_image = norm(image)

    # Map grayscale values to colors
    colors = plt.cm.gray(normalized_image)[:, :, :3]  # Normalize and apply a colormap (e.g., jet)
    colors = colors.reshape(-1, 3)  # Flatten colors

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[show_mask])
    pcd.colors = o3d.utility.Vector3dVector(colors[show_mask])
    return pcd


def o3d_viz_radar_lidar(index):
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
    T_camera_radar = np.matmul(get_inverse_tf(T_enu_camera), T_enu_radar)

    # bbs = rad.get_bounding_boxes(seq.labelFiles, seq.labelTimes, seq.labelPoses)
    # bbs.passthrough(bounds)
    # T_bev_metric = get_T_bev_metric(resolution, width)
    # bbs.transform(T_bev_metric)
    
    # Create coordinate frames for both transformations
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    frame_lidar = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    frame_radar = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])

    # Apply the transformations to the coordinate frames
    frame_cam = copy.deepcopy(frame)
    frame_lidar = frame_lidar.transform(T_camera_lidar)
    frame_radar = frame_radar.transform(T_camera_radar)
    
    # Compute colormap from height wrt lidar frame
    height = copy.deepcopy(lid).points[:,2]
    norm = Normalize(vmin=-3, vmax=5)
    normalized_height = norm(height)
    colors = plt.cm.magma(normalized_height)[:,:3]  # Normalize and apply a colormap (e.g., jet)
    
    pointcloud_wrt_radar = copy.deepcopy(lid).transform(T_radar_lidar)[:,:3]
    r = np.linalg.norm(pointcloud_wrt_radar[:,:2],axis=1)
    z = pointcloud_wrt_radar[:,2]
    phi = np.arctan2(z,r)
    phi = np.abs(phi)
    # # Compute colormap from abs(phi) wrt radar frame
    # norm = Normalize(vmin=0, vmax=1.8/180*np.pi)
    # normalized_phi = norm(phi)
    # colors = plt.cm.viridis.reversed()(normalized_phi)[:,:3]  # Normalize and apply a colormap (e.g., jet)
    
    # colors[phi > 1.8/180*np.pi] = (1.0, 0, 0)
    show_mask1 = phi < 1.8*2/180*np.pi
    show_mask2 = phi > -1.8/180*np.pi
    show_mask = show_mask1 * show_mask2
    # show_mask = np.ones_like(height).astype(bool)
    
    # LiDAR PC
    pointcloud = copy.deepcopy(lid).transform(T_camera_lidar)[:,:3]
    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pointcloud)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.points = o3d.utility.Vector3dVector(pointcloud[show_mask])
    pcd.colors = o3d.utility.Vector3dVector(colors[show_mask])
    
    # undistorted lidar
    pointcloud_undistorted = copy.deepcopy(lid.remove_motion(lid.body_rate))[:,:3]
    pcd_undis = o3d.geometry.PointCloud()
    pcd_undis.points = o3d.utility.Vector3dVector(pointcloud_undistorted)
    pcd_undis.transform(T_camera_lidar)
    
    resolution = 0.25 # 0.2384
    max_dist = 75
    width = int(max_dist*2 / resolution) # 640
    cart = rad.polar_to_cart(
        cart_resolution=resolution,
        cart_pixel_width=width,
        in_place=False,
    )
    
    radar_pcd = image_to_pointcloud(cart, resolution, vmax=0.7, vmin=0.0)
    radar_pcd.transform(T_camera_radar)
    
    # Visualize the frames in Open3D
    # o3d.visualization.draw_geometries([frame_cam, frame_lidar, frame_radar, pcd], window_name="Transformation Matrices Visualization")
    # o3d.visualization.draw_geometries([frame_cam, frame_lidar, pcd, frame_radar, radar_pcd], window_name="Transformation Matrices Visualization")
    # o3d.visualization.draw_geometries([pcd, pcd_undis], window_name="Transformation Matrices Visualization")
    
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add point cloud to the visualizer
    # vis.add_geometry(frame_cam)
    # vis.add_geometry(frame_lidar)
    vis.add_geometry(pcd)
    # vis.add_geometry(radar_pcd)
    vis.add_geometry(frame_radar)

    # Get render options and set background color (e.g., black)
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # RGB values (black)

    # Run the visualizer
    vis.run()
    vis.destroy_window()
    
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
    
    T_enu_camera = cam.pose
    T_enu_lidar = lid.pose
    T_enu_radar = rad.pose

    T_camera_lidar = np.matmul(get_inverse_tf(T_enu_camera), T_enu_lidar)
    T_radar_lidar = np.matmul(get_inverse_tf(T_enu_radar), T_enu_lidar)
    T_camera_radar = np.matmul(get_inverse_tf(T_enu_camera), T_enu_radar)
    
    resolution = 0.25 # 0.2384
    max_dist = 75
    width = int(max_dist*2 / resolution) # 600
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
    lid.transform(T_radar_lidar)
    bounds = [-(max_dist-1), (max_dist-1), -(max_dist-1), (max_dist-1), 0, 10] # xmin, xmax, ymin, ymax, zmin, zmax
    lid.passthrough(bounds)
    ax2.scatter(lid.points[:, 1]/resolution + width/2, -lid.points[:, 0]/resolution + width/2, s=0.01, c=lid.points[:, 2], vmin=-5, vmax=10, cmap='jet')
    
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

def save_sync_lidar(index, window_size=5):
    """Save synced lidar at the given index."""
    # print('index: ', index)
    lid = seq.get_lidar(index)
    rad = seq.get_radar(index)
    
    T_enu_lidar = lid.pose
    T_enu_radar = rad.pose
    T_radar_lidar = np.matmul(get_inverse_tf(T_enu_radar), T_enu_lidar)
    
    origin = T_enu_radar
    pcd_list=[]
    for local_index in range(max(index-window_size,0), min(index+window_size+1,data_len)): 
        lid_ = seq.get_lidar(local_index)
        rad_ = seq.get_radar(local_index)
        T_enu_radar_local = rad_.pose
        
        lid_.transform(T_radar_lidar)
        
        points = lid_.remove_motion(lid.body_rate)[:,:3]
        points = lid_.points[:,:3]
        
        # remove points from ego vehicle
        distances = np.linalg.norm(points, axis=1)
        points = points[distances > 0.5]
        
        # remove outside radar fov
        r = np.linalg.norm(points[:,:2],axis=1)
        z = points[:,2]
        phi = np.arctan2(z,r)
        phi = np.abs(phi)
        show_mask = phi < 1.8*2/180*np.pi
        points = points[show_mask]
        
        # transform pc
        T_local = np.linalg.inv(origin) @ T_enu_radar_local
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points) 
        pcd.transform(T_local)

        pcd_list.append(pcd)
        
        del rad_, lid_, pcd
    
    pcd_merge = sum(pcd_list, o3d.geometry.PointCloud())
    
    # resolution = 0.25 # 0.2384
    # max_dist = 75
    # width = int(max_dist*2 / resolution) # 640
    # cart = rad.polar_to_cart(
    #     cart_resolution=resolution,
    #     cart_pixel_width=width,
    #     in_place=False,
    # )
    
    # radar_pcd = image_to_pointcloud(cart, resolution, vmax=0.4, vmin=0.1)
    
    # origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # o3d.visualization.draw_geometries([origin_frame, pcd_merge, radar_pcd])
    if window_size==0:
        save_path = f"{args.data_root}/{split[0][0]}/synced_lidar"
    else:
        save_path = f"{args.data_root}/{split[0][0]}/synced_lidar_map_win{window_size}"
    os.makedirs(save_path, exist_ok=True)
    o3d.io.write_point_cloud(f"{save_path}/{rad.timestamp_micro}.pcd", pcd_merge)
    
    del rad, lid

def on_key_image(event):
    """Handle key press events."""
    global current_index
    # data_len = len(image_files)
    data_len = len(seq.lidar_frames)
    if event.key == "right":  # Next image
        current_index = (current_index + 1) % data_len
    elif event.key == "left":  # Previous image
        current_index = (current_index - 1) % data_len
    elif event.key == ".":  # Next image
        current_index = (current_index + 20) % data_len
    elif event.key == ",":  # Previous image
        current_index = (current_index - 20) % data_len
    elif event.key == "escape":  # Exit interaction
        print("Exiting...")
        return
    plt.clf()  # Clear the current figure
    display_image(current_index)

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
    display_radar(current_index)

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.viz_plt_image:
        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', on_key_image)
        display_image(current_index)
    if args.viz_plt_radar:
        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', on_key_radar)
        display_radar(current_index)
    
    if args.viz_o3d:
        # vis_lidar_bbs(index)
        # get_local_lidar_map(index, window_size=5)
        o3d_viz_radar_lidar(current_index)

    if args.save_sync_lidar:
        from tqdm import tqdm
        for index in tqdm(range(data_len)[:args.end_index][args.start_index:], desc="Saving synced lidar"):
            save_sync_lidar(index, window_size=args.window_size)
    