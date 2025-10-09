import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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

from scipy.spatial.transform import Rotation

import argparse

parser = argparse.ArgumentParser(description="Boreas Poses Saver")
parser.add_argument("--data_root", default='/mnt/ws-frb/projects/radar_splat/data/boreas/', help="Boreas data root")
parser.add_argument("--seq_name", default='boreas-2021-09-02-11-42', help="Boreas seq name")

args = parser.parse_args()


root = args.data_root
# split=[['boreas-objects-v1']]
# split = [['boreas-2021-03-09-14-23']] # Testing Slice. No sensor pose.
# split = [['boreas-2021-09-02-11-42']]
split = [[args.seq_name]]

# bd = BoreasDataset(root, split=obj_train, verbose=True)

bd = BoreasDataset(root, split=split, verbose=True, labelFolder='labels_detection')

seq = bd.sequences[0] # 36  ## ** Try different sequences!
if split[0][0] == "boreas-objects-v1": # TODO: here we only consider the first sequence
    seq.filter_frames_gt()

print('--radar frames--')
print(len(seq.radar_frames))
data_len = len(seq.radar_frames)

if __name__ == "__main__":
    args = parser.parse_args()

    # Show radar poses
    radar_origin=None
    vis_list=[]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    filename = seq.seq_root+"/radar_trajectory.tum"
    with open(filename, "w") as f:
        from tqdm import tqdm
        for index in tqdm(range(data_len), desc="Saving radar poses"):
            rad = seq.get_radar(index)
            T_enu_radar = rad.pose
            timestamp = rad.frame
            
            if index==0:
                radar_origin = T_enu_radar
                # frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                # vis_list.append(frame_origin)
                
            T_local = np.linalg.inv(radar_origin) @ T_enu_radar
                    
            filename = seq.seq_root+"/radar_trajectory.tum"
            
            # Extract translation
            tx, ty, tz = T_local[:3, 3]

            # Extract rotation matrix
            R = T_local[:3, :3]

            # Convert rotation matrix to quaternion (qx, qy, qz, qw)
            quat = Rotation.from_matrix(R).as_quat()
            qx, qy, qz, qw = quat  # scipy outputs (x, y, z, w) order

            # Save in TUM format
            f.write(f"{timestamp} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            
            # frame_current = copy.deepcopy(frame)
            # frame_current = frame_current.transform(T_local_radar)
            # vis_list.append(frame_current)
            
            # if index%100==0:
            #     o3d.visualization.draw_geometries(vis_list)