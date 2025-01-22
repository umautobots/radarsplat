import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from PIL import Image
import imageio


# This code is borrow from RadarFields github
# https://github.com/princeton-computational-imaging/RadarFields/blob/ee76d76570f58b3d8539eafd7df0c188b58af333/utils/vis.py#L36
def polar_to_cart(fft_img, num_bins_to_show, bin_size, num_azims, path,
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
    
    render[white_mask > 0.5] = np.nan # 1.0

    # if save:
    #     render = Image.fromarray((render*255).astype(np.uint8))
    #     render.save(path, compress_level=0)
    # if plot:
    #     plt.imshow(render)
    #     plt.show()
    # if not save and not plot: return render
    
    return render
