import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.transforms as mtf
from matplotlib.colors import Normalize
import numpy as np

from pyboreas import BoreasDataset
from pyboreas.data.splits import obj_train
from pyboreas.utils.utils import get_inverse_tf
from scipy.ndimage import gaussian_filter1d

import os
import cv2
from pyboreas.utils.utils import get_T_bev_metric
import open3d as o3d
import copy
from scipy.optimize import curve_fit
import argparse

def display_radar(index, azimuth_id):
    """Display the image at the given index."""
    print('index: ', index, ' azimuth_id: ', azimuth_id)
    rad = seq.get_radar(index)
    range_resolution=0.0596
    
    # Run FFT
    polar_fft_log, polar_fft, fft_result, fft_freq = FFT(rad.polar[:,:range_pixels], range_resolution=range_resolution)
    
    # Saturation noise detection
    constant_ratio_thres=0.21

    # Compute constant ratio
    constant_ratio = polar_fft[:,:polar_fft.shape[1]//2][:,0]/polar_fft[:,:polar_fft.shape[1]//2].sum(axis=1)
    # Get saturation azimuth mask
    saturate_azi_mask = constant_ratio>constant_ratio_thres

    # Multipath noise detection
    skip_first_n_freq = 5
    magnitude_thres = 30
    constant_ratio_thres_=0.2
    
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
    
    # Noise removal
    noise_azi_mask = saturate_azi_mask + multipath_azi_mask
    polar_image_filtered, polar_occ_filtered = apply_saturation_mask(rad.polar[:,:range_pixels], noise_azi_mask, smoothing_sigma=smoothing_sigma)
    cart_filtered = rad.polar_to_cart(
        cart_resolution=resolution,
        cart_pixel_width=width,
        polar=polar_image_filtered,
        in_place=False,
    )
    
    # get polar_image_multipath_segmenation
    polar_image_multipath_segmenation = np.zeros_like(rad.polar[:,:range_pixels])
    polar_image_multipath_segmenation[azi_id_list] = reconstructed_signal
    cart_multipath_segmenation = rad.polar_to_cart(
        cart_resolution=resolution,
        cart_pixel_width=width,
        polar=polar_image_multipath_segmenation,
        in_place=False,
    )

    # get polar_image_multipath_ifft
    polar_image_multipath_ifft = np.zeros_like(rad.polar[:,:range_pixels])
    polar_image_multipath_ifft[azi_id_list] = ifft_recovered_signal
    cart_multipath_ifft = rad.polar_to_cart(
        cart_resolution=resolution,
        cart_pixel_width=width,
        polar=polar_image_multipath_ifft,
        in_place=False,
    )
    
    # get polar_image_multipath_ifft
    polar_image_multipath = np.zeros_like(rad.polar[:,:range_pixels])
    polar_image_multipath[azi_id_list] = reconstructed_signal
    cart_multipath = rad.polar_to_cart(
        cart_resolution=resolution,
        cart_pixel_width=width,
        polar=polar_image_multipath,
        in_place=False,
    )
    size = 600
    radius = 300  # Adjust the circle radius as needed
    center = (size // 2, size // 2)
    image = np.ones((size, size))
    y, x = np.ogrid[:size, :size]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    cart_multipath[~mask] = 1 # set outside circle to be white

    colormap = plt.cm.gray
    cart_multipath_color = colormap(cart_multipath)
    cart_multipath_color = (cart_multipath_color[:, :, :3] * 255).astype(np.uint8) *2 # for viz !!!
    
    polar_image_multipath_color = colormap(polar_image_multipath)
    polar_image_multipath_color = (polar_image_multipath_color[:, :, :3] * 255).astype(np.uint8) *2 # for viz !!!
    for azi_id, range_id in zip(azi_id_list, range_id_list):
        azi_min = max(0, azi_id - 1)
        azi_max = min(polar_image_multipath_color.shape[0], azi_id + 2)  # +2 because Python slicing excludes the upper bound
        range_min = max(0, range_id - 5)
        range_max = min(polar_image_multipath_color.shape[1], range_id + 6)
        polar_image_multipath_color[azi_min:azi_max, range_min:range_max] = 1
    
    cart_multipath_color_ = rad.polar_to_cart(
        cart_resolution=resolution,
        cart_pixel_width=width,
        polar=polar_image_multipath_color,
        in_place=False,
    )
    
    # For viz
    polar_image = rad.polar.copy()
    polar_color_image = np.dstack([polar_image] * 3)
    polar_color_image[azimuth_id, :, 0] = 1  # Set Red channel to 1
    polar_color_image[azimuth_id, :, 1] = 0  # Set Green channel to 0
    polar_color_image[azimuth_id, :, 2] = 0  # Set Blue channel to 0
    
    # Make selected azimuth white
    # polar_image[azimuth_id, :] = 1.
    
    cart = rad.polar_to_cart(
        cart_resolution=resolution,
        cart_pixel_width=width,
        polar=polar_image,
        in_place=False,
    )
    
    fig.clear() 
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 3, 6)
    # ax3_top = plt.subplot(3, 3, 7)  # Top plot (Radar Intensity)
    # ax3_bottom = plt.subplot(3, 3, 9)  # Bottom plot (Second Plot)
    
    fig_save, ax_save = plt.subplots(figsize=(12, 6))
    ax_save = plt.subplot(1, 1, 1)
    
    fig.subplots_adjust()
    
    ### Radar Cart
    ax1.imshow(cart, cmap="gray")
    ax1.axis("off")  # Remove axes for cleaner display
    ax1.set_title(f"Radar {index + 1}/{data_len}: {rad.timestamp}")
    
    ### Radar Cart after noise removal
    ax4.imshow(cart_filtered, cmap="gray")
    ax4.axis("off")  # Remove axes for cleaner display
    
    ### Radar Polar
    polar_color_image[azi_id_list, range_id_list]=[0,1,0]
    ax2.imshow(polar_color_image[:,:range_pixels])
    ax2.axis("off")  # Remove axes for cleaner display
    
    ### Radar FFT
    # ax5.imshow(polar_fft_log[:,:polar_fft_log.shape[1]], cmap="gray")

    ### Radar multipath segmentation
    # ax5.imshow(cart_multipath_segmenation, cmap="gray")
    # ax5.imshow(cart_multipath_ifft, cmap="gray")
    ax5.imshow(cart_multipath_color) # cart_multipath, cart_multipath_color    
    ax5.axis("off")  # Remove axes for cleaner display
    ax5.set_title(f"Radar FFT")
    
    # Multipath Sources
    source_theta = rad.azimuths[azi_id_list].reshape(-1)
    source_d = range_id_list * range_resolution
    source_x = source_d * np.cos(source_theta)
    source_y = source_d * np.sin(source_theta)
    ax5.scatter(source_y/resolution + width/2, -source_x/resolution + width/2, s=5, c='lime')

    ### 1D signal on selected azimuth angle
    dist_values = (np.array([i for i in range(len(rad.polar[azimuth_id]))]) * 0.0596)[:range_pixels]
    intensity_values = rad.polar[azimuth_id,:range_pixels]
    ax3.plot(dist_values, intensity_values, color='b', label='Radar Intensity')

    # smoothed signal
    smoothed_values = gaussian_filter1d(intensity_values, sigma=smoothing_sigma)
    ax3.plot(dist_values, smoothed_values, color='g', label='Smoothed Radar Intensity')

    # selected point
    if range_id is not None:
        ax3.scatter(
            dist_values[range_id], rad.polar[azimuth_id, range_id], 
            color='red', s=50, label="Selected Point"
        )
    # # 1/4 median value
    # valid_intensity_values = intensity_values[min_range:]
    # smaller_values = valid_intensity_values[valid_intensity_values < np.median(valid_intensity_values)]
    # lower_median = np.median(smaller_values)
    # ax3.axhline(y=lower_median, color='r', linestyle='--', label=f"Selected Intensity: {lower_median:.2f}")  # Plot horizontal line

    ax3.set_xlabel("Range (m)")
    ax3.set_ylabel("Intensity")
    ax3.set_title("1D Radar Intensity Plot")
    ax3.set_ylim(0, 1)  # Intensity is between 0 and 1
    ax3.grid(True)
    ax3.legend()
    
    # Apply FFT on intensity values
    fft_magnitude = polar_fft[azimuth_id]
    fft_result_azi = fft_result[azimuth_id]
    constant_ratio = polar_fft[azimuth_id,:polar_fft.shape[1]//2][0]/polar_fft[azimuth_id,:polar_fft.shape[1]//2].sum()
    ax6.plot(fft_freq[:len(fft_freq)//2], fft_magnitude[:len(fft_magnitude)//2], color='r', label='FFT Magnitude')  # Show only positive frequencies
    ax6.set_xlabel("Frequency (1/m)")
    ax6.set_ylabel("Magnitude")
    ax6.set_title(f"FFT of Radar Intensity. constant ratio: {constant_ratio:.2f}. constant power: {fft_magnitude[0]:.2f}")
    ax6.set_xlim(0, 2)
    ax6.set_ylim(0, 200)
    ax6.grid(True)
    ax6.legend()
        
    fft_magnitude_ = fft_magnitude[skip_first_n_freq:len(fft_magnitude)//2]
    # peak_ratio = np.max(fft_magnitude_) / np.sum(fft_magnitude_)
    # multipath_index = np.array([np.argmax(fft_magnitude_)])
    # print('peak_ratio: ', peak_ratio)
    # if peak_ratio > 0.06:

    # multipath_index = np.where(fft_magnitude_ > magnitude_thres)[0]
    
    if np.max(fft_magnitude_) > magnitude_thres and constant_ratio>constant_ratio_thres:
        multipath_index = np.array([np.argmax(fft_magnitude_)])
    else:
        multipath_index=np.array([])
    
    if len(multipath_index)>0:
        print('multipath_index: ', multipath_index)
        multipath_freq = fft_freq[skip_first_n_freq:][multipath_index]
        multipath_dist = 1./multipath_freq
        for dist, idx in zip(multipath_dist, multipath_index):
            # ax3.axvline(x=dist, color='r', linestyle='--')  # Plot vertical line
            mask = np.zeros_like(fft_result_azi).astype(bool)
            mask[idx+skip_first_n_freq]=1
            # mask[-(idx+skip_first_n_freq+1)]=1

            filtered_fft = fft_result_azi * mask
            
            # ### Inverse FFT
            # recovered_signal = np.fft.ifft(filtered_fft).real
            # ax3.plot(dist_values, recovered_signal, color='c', label='Recovered Signal')
            
            ### Implement Inverse FFT
            # Extract amplitude and phase for sine/cosine reconstruction
            amplitudes = np.abs(filtered_fft) / len(dist_values)
            phases = np.angle(filtered_fft)

            # Reconstruct signal using sine and cosine waves
            reconstructed_signal = np.zeros_like(dist_values)
            for i, freq in enumerate(fft_freq[mask]):
                amp = amplitudes[mask][i]
                phase = phases[mask][i]
                print('reconstructed freq:', freq)
                reconstructed_signal += amp * np.cos(2 * np.pi * freq * dist_values + phase)
            reconstructed_signal[reconstructed_signal<0]=0
            reconstructed_signal[dist_values<(dist+ 0.5*(1/np.max(fft_freq[mask])))]=0
            # ax3.plot(dist_values, reconstructed_signal, label='Reconstructed Signal (Sine/Cosine)', color='m')
            
            # def model(dist_values, A, alpha):
            #     return A * np.exp(-alpha * dist_values) * reconstructed_signal
            #     # return A * reconstructed_signal
            # popt, _ = curve_fit(model, dist_values, intensity_values, p0=[3.0, 0.1], bounds=([1.0, 0.0], [100.0, 0.2]))
            # estimated_A, estimated_alpha = popt
            # reconstructed_signal = model(dist_values, estimated_A, estimated_alpha)
            # print('A: ', estimated_A, ' alpha:', estimated_alpha)
            # ax3.plot(dist_values, reconstructed_signal, label='Reconstructed Signal w/ scale and exp', color='orange')
            
            # ax3.plot(dist_values, polar_image_multipath_segmenation[azimuth_id], label='Reconstructed Signal vectorized', color='lime')
            # ax3.plot(dist_values, polar_image_multipath_ifft[azimuth_id], label='Reconstructed Signal IFFT', color='tomato')

    # ax_save.plot(dist_values, intensity_values, color='royalblue', linewidth=3)
    ax_save.bar(dist_values, intensity_values, color='royalblue', linewidth=5, width=0.2, alpha=1)
    # ax_save.bar(dist_values, intensity_values, color='royalblue', linewidth=5, width=0.2, alpha=0.3)
    # ax_save.fill_between(dist_values, intensity_values, color='royalblue', alpha=0.9)

    # ax_save.bar(dist_values, reconstructed_signal, color='brown', linewidth=5, width=0.1)
    
    # ax_save.plot(dist_values, polar_image_multipath_segmenation[azimuth_id], label='Reconstructed Multipath', color='darkorange', linewidth=5)
    # ax_save.bar(dist_values, polar_image_multipath_segmenation[azimuth_id], color='darkorange', linewidth=5, width=0.2, alpha=0.3)

    # ax_save.plot(dist_values, smoothed_values, color='darkorchid', label='Smoothed Radar Intensity', linewidth=5)
    # ax_save.bar(dist_values, smoothed_values, color='darkorchid', linewidth=5, width=0.1)

    # ax_save.plot(dist_values, polar_image_filtered[azimuth_id], label='Filtered Signal', color='tomato', linewidth=5)
    # ax_save.bar(dist_values, polar_image_filtered[azimuth_id], color='tomato', linewidth=5, width=0.2, alpha=0.3)

    # ax_save.plot(dist_values, reconstructed_signal, label='Reconstructed Signal vectorized', color='peru', linewidth=5)

    ax_save.set_xlim(0, 75)
    ax_save.set_ylim(0, 0.6)
    ax_save.set_xlabel('Range (m)', fontsize=20)
    ax_save.set_ylabel('Radar Power', fontsize=20)
    ax_save.spines['top'].set_visible(False)
    ax_save.spines['right'].set_visible(False)
    ax_save.spines['left'].set_linewidth(2)
    ax_save.spines['bottom'].set_linewidth(2)

    # ax_save.tick_params(axis='both', labelsize=14, width=2, length=6)

    # ax_save.plot(fft_freq[:len(fft_freq)//2], fft_magnitude[:len(fft_magnitude)//2]/len(fft_magnitude), color='green', label='FFT Magnitude', linewidth=5)  # Show only positive frequencies
    # ax_save.set_xlabel("Frequency (1/m)", fontsize=20)
    # ax_save.set_ylabel("Magnitude", fontsize=20)
    # # ax_save.set_title(f"FFT of Radar Intensity. constant ratio: {constant_ratio:.2f}. constant power: {fft_magnitude[0]:.2f}")
    # ax_save.set_xlim(0, 2)
    # ax_save.set_ylim(0, 0.09)
    # ax_save.spines['top'].set_visible(False)
    # ax_save.spines['right'].set_visible(False)

    # ax_save.grid(True)
    # ax_save.tick_params(axis='both', which='major', labelsize=20)  # Font size for major ticks
    # plt.close(fig_save)

    # plt.tight_layout()    
    # plt.show()
    
    fig_save.savefig('debug.png')

def multipath_modeling(polar_fft, fft_result, fft_freq, multipath_azi_mask, raw_polar_image, skip_first_n_freq, range_resolution, range_pixels):
    # fft_magnitude are symetric. use left-half to find maximum index
    fft_magnitude_half = polar_fft[:,skip_first_n_freq:polar_fft.shape[1]//2]

    # get index with max magnitude in FFT image 
    azi_max_magnitude_fft_id = np.argmax(fft_magnitude_half, axis=1) + skip_first_n_freq
    # get correspond distance from main freq.
    azi_max_magnitude_range = 1./fft_freq[azi_max_magnitude_fft_id] # (m)
    # # get more accurate correspond distance from main phase. (This is not working)
    # phases_positive = np.angle(fft_result[np.arange(fft_result.shape[0]),azi_max_magnitude_fft_id])
    # azi_max_magnitude_range += (phases_positive / (2 * torch.pi)) * azi_max_magnitude_range

    # get correspond index in range space 
    azi_max_magnitude_range_id = (azi_max_magnitude_range / range_resolution).astype(int) # (pixel)

    # azimuth index of multipath sources
    azi_id_list = np.where(multipath_azi_mask==True)[0]
    # frq. index of multipath sources
    fft_id_list = azi_max_magnitude_fft_id[azi_id_list]
    # range of multipath sources
    range_list = azi_max_magnitude_range[azi_id_list]
    # range index of multipath sources
    range_id_list = azi_max_magnitude_range_id[azi_id_list]

    # [azi_id_list, range_id_list]: indices list of multipath sources in range image
    # [azi_id_list, fft_id_list]: indices list of multipath sources in fft image

    # create a subset of fft_result with only selected azimuth
    fft_result_subset = fft_result[multipath_azi_mask]
    # mask for main magnitude selection
    mask = np.zeros_like(fft_result_subset).astype(bool)
    mask[np.arange(len(fft_id_list)), fft_id_list]=1
    mask[np.arange(len(fft_id_list)), -(fft_id_list+1)]=1
    # get filter FFT
    fft_result_subset_filtered = fft_result_subset * mask

    # # IFFT to recover signal from filtered FFT (For debug)
    ifft_recovered_signal = np.fft.ifft(fft_result_subset_filtered).real

    # get both positive and negative main freq.
    freq_positive = fft_freq[fft_id_list]
    freq_negative = fft_freq[-(fft_id_list+1)]

    # get main components of both positive and negative freq. 
    selected_main_fft_positive = fft_result_subset[np.arange(len(fft_id_list)), fft_id_list]
    selected_main_fft_negative = fft_result_subset[np.arange(len(fft_id_list)), -(fft_id_list+1)] # the right part is improtant for signal reconstruction
    
    # get amplitude & phase from complex value
    amplitudes_positive = np.abs(selected_main_fft_positive) / range_pixels
    phases_positive = np.angle(selected_main_fft_positive)
    amplitudes_negative = np.abs(selected_main_fft_negative) / range_pixels
    phases_negative = np.angle(selected_main_fft_negative)

    # reconstructed signal with cos function
    reconstructed_signal_positive = amplitudes_positive.reshape(-1,1) * np.cos(2 * np.pi * freq_positive.reshape(-1,1) * np.arange(range_pixels).reshape(1,-1)*range_resolution + phases_positive.reshape(-1,1))
    reconstructed_signal_negative = amplitudes_negative.reshape(-1,1) * np.cos(2 * np.pi * freq_negative.reshape(-1,1) * np.arange(range_pixels).reshape(1,-1)*range_resolution + phases_negative.reshape(-1,1))
    reconstructed_signal = reconstructed_signal_positive #+ reconstructed_signal_negative

    # make reconstructed signal always > 0
    reconstructed_signal[reconstructed_signal<0]=0

    # make reconstructed signal be zero before the multipath source
    source_dist = 1./freq_positive
    reconstructed_signal[reconstructed_signal<0]=0
    dist_values = np.arange(range_pixels)*range_resolution
    reconstructed_signal[dist_values.reshape(1,-1)<(1.5*source_dist.reshape(-1,1))]=0

    # # compute scale (not the best way to rescale. skip this.)
    # sources_peak_power = rad.polar[azi_id_list, range_id_list].reshape(-1,1)
    # scales_list = sources_peak_power / reconstructed_signal.max()
    # reconstructed_signal = reconstructed_signal * scales_list

    # fit raw radar signal with A*exp(-alpha*x) 
    N = reconstructed_signal.shape[0]
    estimated_A = np.zeros((N,1))
    estimated_alpha = np.zeros((N,1))
    for i in range(N):
        def model(dist_values, A, alpha):
            return A * np.exp(-alpha * dist_values) * reconstructed_signal[i]
        popt, _ = curve_fit(model, dist_values, raw_polar_image[azi_id_list,:range_pixels][i], p0=[3.0, 0.1], bounds=([1.0, 0.0], [100.0, 0.2]))
        estimated_A[i], estimated_alpha[i] = popt
    # rescale
    reconstructed_signal = estimated_A * np.exp(-estimated_alpha * dist_values) * reconstructed_signal

    return azi_id_list, range_id_list, \
            freq_positive, amplitudes_positive, phases_positive, \
            estimated_A, estimated_alpha, \
            reconstructed_signal, ifft_recovered_signal


def apply_saturation_mask(polar_image, saturate_azi_mask, smoothing_sigma, occ_thres=0.1):
    """
    Replace values outside the decay region with 0 for rows where polar_fft[:, 0] > threshold.

    Args:
    - polar_image (numpy array): 2D array of radar polar intensities (shape: [rows, range_pixels]).
    - polar_fft (numpy array): 2D FFT magnitude spectrum of the polar image.
    - saturate_azi_mask: Saturation mask for selecting affected rows.

    Returns:
    - modified_polar (numpy array): Modified polar image with values outside decay regions set to 0.
    - occ_polar (numpy array): Occupany polar image with values after decay regions set to 0.5 (unknown).
    """
    modified_polar = polar_image.copy()
    occ_polar = polar_image.copy()
    occ_polar[occ_polar>=occ_thres]=np.clip(occ_polar+0.5, 0, 1)[occ_polar>=occ_thres]
    occ_polar[occ_polar<occ_thres]=0.0

    for row_idx in np.where(saturate_azi_mask)[0]:  # Get indices where saturation occurs
        max_idx, decay_region = find_decay_region(polar_image[row_idx], sigma=smoothing_sigma)

        # Zero out values outside the decay region
        modified_polar[row_idx, :decay_region[0]] = 0  # Left side
        modified_polar[row_idx, decay_region[1] + 1:] = 0  # Right side
        
        occ_polar[row_idx, :decay_region[0]] = 0.5
        occ_polar[row_idx, decay_region[1] + 1:] = 0.5

    return modified_polar, occ_polar


def radarfields_occupancy(polar_image):

    H, W = polar_image.shape  # H = 400 (azimuth), W = 800 (range)

    # Step 1: Estimate per azimuth and per range noise thresholds (Eq. 1 & 2)
    n_phi = np.median(polar_image, axis=1, keepdims=True)  # Median per azimuth
    n_b = np.median(polar_image, axis=0, keepdims=True)    # Median per range

    # Step 2: Compute the final noise threshold T(phi, b) (Eq. 3)
    T_phi_b = 2 * np.maximum(n_phi, n_b)

    # Step 3: Threshold the radar image (Eq. 4)
    Pr_prime = np.where(polar_image >= T_phi_b, polar_image, 0)

    # Step 4: Compute occupancy probability (Eq. 5)
    Po = 0.1  # Hyperparameter
    delta = 10 #2.0 # Hyperparameter
    Pr = np.clip(Pr_prime * np.exp(delta*(Pr_prime - Po)), 0, 1)

    # Step 5: Apply occlusion model (Eq. 6)
    delta_b = 20. # Hyperparameter
    bp_all = np.argmax(polar_image, axis=1)
    processed_image = np.zeros_like(polar_image)
    for azimuth in range(H):
        for b in range(W - 1):
            bp = bp_all[azimuth]
            delta_x = b - bp
            if delta_x>0:
                attenuation = np.exp(-delta_x / delta_b)
                processed_image[azimuth, b] = np.maximum(Pr[azimuth, b], Pr[azimuth, bp] * attenuation)
            else:
                processed_image[azimuth, b] = Pr[azimuth, b]

    return processed_image


def find_decay_region(intensity_values, sigma=1.0):
    """
    Find the region where intensity values continuously decay after the maximum point.

    Args:
    - intensity_values (numpy array): 1D array of intensity values.

    Returns:
    - max_index (int): Index of the maximum intensity value.
    - decay_region (tuple): (start_index, end_index) of the decay region.
    """
    
    smoothed_values = gaussian_filter1d(intensity_values, sigma=sigma)

    # Step 1: Find max intensity value and its index
    max_index = np.argmax(smoothed_values)
    
    # Step 2: Find left boundary (backward search)
    start_index = max_index
    while start_index > 0 and smoothed_values[start_index - 1] <= smoothed_values[start_index]: # (Frank) bug fix 0225. TODO: Rerun experiments
        start_index -= 1
    
    # Step 3: Find right boundary (forward search)
    end_index = max_index
    while end_index < len(smoothed_values) - 1 and smoothed_values[end_index + 1] <= smoothed_values[end_index]:
        end_index += 1
    
    # Step 4: Return the max index and decay region
    return max_index, (start_index, end_index)

import torch
def FFT(polar_image, range_resolution=0.0596):
    # Convert to PyTorch tensor and move to GPU
    polar_tensor = torch.tensor(polar_image, dtype=torch.float32).to("cuda")  # Shape: (400, N)

    # Apply FFT on all rows (dim=1 means row-wise FFT)
    fft_result = torch.fft.fft(polar_tensor, dim=1)

    # Compute magnitude spectrum (absolute value of FFT)
    fft_magnitude = torch.abs(fft_result)
    
    fft_freq = np.fft.fftfreq(polar_image.shape[1], d=range_resolution)

    # Convert to CPU and normalize for visualization
    fft_image = fft_magnitude.cpu().numpy()
    fft_result = fft_result.cpu().numpy()
    fft_image_log = np.log1p(fft_image)  # Log scaling for better visibility
    return fft_image_log, fft_image, fft_result, fft_freq

def onclick(event):
    """Handle mouse clicks to determine the azimuth_id and refresh display."""
    global azimuth_id, range_id
    if event.inaxes is not None:
        if event.inaxes.get_title().startswith("Radar "):  # Ensure click is in ax1
            x, y = int(event.xdata), int(event.ydata)

            frame_x = -y
            frame_y = x

            # Compute theta in degrees (polar angle, counterclockwise from +x-axis)
            dx = frame_x + width // 2
            dy = frame_y - width // 2  # Invert y-axis since image coordinates are top-down
            theta = np.degrees(np.arctan2(dy, dx))

            # Normalize theta to [0, 360]
            if theta < 0:
                theta += 360

            print(f"Clicked at: (x={frame_x}, y={frame_y}), Theta: {theta:.2f}°")

            # Convert theta to azimuth_id
            azimuth_id = int(theta / 0.9)
            
            range_id = int(np.sqrt(dx**2 + dy**2) * resolution / 0.0596)
            
            display_radar(current_index, azimuth_id)  # Refresh display
            
def on_key_radar(event):
    """Handle key press events."""
    global current_index, azimuth_id
    data_len = len(seq.lidar_frames)
        
    if event.key == "right":  # Next image
        current_index = (current_index + 1) % data_len
    elif event.key == "left":  # Previous image
        current_index = (current_index - 1) % data_len
    elif event.key == "4":
        azimuth_id = azimuth_id+10
    elif event.key == "3":
        azimuth_id = azimuth_id+1
    elif event.key == "2":
        azimuth_id = azimuth_id-1
    elif event.key == "1":
        azimuth_id = azimuth_id-10
    elif event.key == "escape":  # Exit interaction
        print("Exiting...")
        return
    plt.clf()  # Clear the current figure
    display_radar(current_index, azimuth_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boreas Data Visualization")
    parser.add_argument("--data_root", default='/mnt/ws-frb/projects/radar_splat/data/boreas/', help="Boreas data root")
    parser.add_argument("--viz_plt_image", action="store_true", help="viz image")
    parser.add_argument("--viz_plt_radar", action="store_true", help="viz radar")
    parser.add_argument("--viz_o3d", action="store_true", help="viz pointcloud with o3d")
    parser.add_argument("--save_sync_lidar", action="store_true", help="path to save synced lidar data")
    parser.add_argument("--start_index", type=int, default=0, help="index to start")

    args = parser.parse_args()

    # split=[['boreas-objects-v1']]
    # split = [['boreas-2021-03-09-14-23']] # Testing Slice. No sensor pose.
    split = [['boreas-2021-09-02-11-42']]

    # bd = BoreasDataset(args.data_root, split=obj_train, verbose=True)
    bd = BoreasDataset(args.data_root, split=split, verbose=True, labelFolder='labels_detection')

    seq = bd.sequences[0]
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

    azimuth_id = 159
    # index:27 azi_id:159 paper demo no noise 
    # index:27 azi_id:21 paper demo saturation 
    # index:27 111 paper demo multipath
    # azimuth_id = 285
    range_id = None

    resolution = 0.25 # 0.2384
    max_dist = 75
    width = int(max_dist*2 / resolution) # 600
    min_range = int(round(2.5 / 0.0596))
    range_pixels = int(max_dist/0.0596)
    smoothing_sigma = 10.0

    fig = plt.figure()
    
    # Connect the event listener to the figure
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key_radar)
    
    display_radar(current_index, azimuth_id)
    
    plt.show()
    