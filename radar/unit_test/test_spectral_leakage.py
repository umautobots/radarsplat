from gsplat.rendering import spectral_leakage, azimuth_antenna_gain_projection
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

range_resolution=0.0596

# Input
polar_img = torch.zeros((400, 838, 1)).float().cuda()
polar_img[200,:]=0.5
polar_img[:,30]=1

# Spectral Leakage
out_img = spectral_leakage(polar_img, range_resolution, sinc_width=1.)

ax1 = plt.subplot(1,2,1)
ax1.imshow(polar_img.squeeze().detach().cpu())
ax2 = plt.subplot(1,2,2)
ax2.imshow(out_img.squeeze().detach().cpu())
plt.show()