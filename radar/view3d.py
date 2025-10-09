import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import cv2

def bev_image_view3d(bev_image, max_height=2.):
    bev_image = bev_image*max_height
    # Create voxel grid
    H, W = bev_image.shape

    # Normalize height values to [0,1] for colormap
    norm = mcolors.Normalize(vmin=0, vmax=max_height)

    # Get colors from magma colormap
    colors = cm.magma_r(norm(bev_image))  # Apply colormap to height values

    # Convert heightmap to a 3D voxel boolean mask
    voxels = np.zeros((H, W, int(max_height * 10)), dtype=bool)  # Increase resolution
    voxel_colors = np.zeros(voxels.shape + (4,))
    for i in range(H):
        for j in range(W):
            height = int(bev_image[i, j] * 10)+1  # Scale height to voxel layers
            voxels[i, j, :height] = True  # Fill voxels up to the height value
            voxel_colors[i, j, :height, :] = colors[i, j]  # Assign color to filled voxels

    # Set up the figure
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Plot voxels
    ax.voxels(voxels, facecolors=voxel_colors, edgecolor=None)

    # Set labels and limits
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height (m)")
    ax.set_zlim(0, max_height * 100)

    # Set view angle
    # ax.view_init(elev=45, azim=0) # azim=-90 (back to front view)
    # ax.dist = 6
    
    ax.grid(False)
    ax.set_axis_off()

    # fig.show()

    return fig, ax

if __name__=="__main__":
    # Load BEV image from PNG file (grayscale)
    bev_image_path = "/mnt/ws-frb/projects/radar_splat/data/boreas/boreas-2021-09-02-11-42/radar/cart/1630597331060160.png"  # Change this to your actual file path
    bev_image = cv2.imread(bev_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Define max height for scaling
    max_height = 2.0  # Adjust this as needed

    # Normalize to height range
    bev_image = (bev_image / 255.0) * max_height
    
    bev_image = cv2.resize(bev_image, (100, 100), interpolation=cv2.INTER_AREA)

    bev_image_view3d(bev_image)