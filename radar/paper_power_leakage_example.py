import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate time range for sine wave
t = np.linspace(0, 1, 1000, endpoint=False)
sine_wave = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

# Generate frequency range
f = np.linspace(-25, 25, 1000)
T = 1.0  # Duration for scaling

# Original sinc spectrum centered at f=5
original_sinc = np.sinc((f - 5) * T / np.pi)

# Components of the Hann window's Fourier Transform
main_sinc = T * np.sinc((f - 5) * T / np.pi)  # T sinc(wT)
shifted_sinc_plus = (0.5 * T) * np.sinc(((f - 5) + np.pi / T) * T / np.pi)  # 1/2 T sinc((w + π/T)T)
shifted_sinc_minus = (0.5 * T) * np.sinc(((f - 5) - np.pi / T) * T / np.pi)  # 1/2 T sinc((w - π/T)T)

# Summation to get the final Hann window Fourier transform
hann_spectrum = main_sinc + shifted_sinc_plus + shifted_sinc_minus

# Fit a normal distribution to approximate the main lobe of the red curve, centered at f=5
mu = 5  # Shifted mean to 5 Hz
sigma = 3  # Estimated width of the main lobe
gaussian_approx = norm.pdf(f, mu, sigma) * max(hann_spectrum) / max(norm.pdf(f, mu, sigma))

# Plotting
plt.figure(figsize=(9, 12))

linewidth=4
# Sine wave plot
plt.subplot(5, 1, 1)
plt.plot(t, sine_wave, color='black', linewidth=linewidth)
plt.xlabel(r'$t$', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.title(r'$f_0$'+' Hz Sine Wave', fontsize=14)
# plt.grid(True)
plt.axis('off')

plt.subplot(5, 1, 2)
idea = original_sinc.copy()
idea[idea==idea.max()]=1
idea[idea!=1]=0
plt.plot(f, idea, color='royalblue', linewidth=linewidth, label=r'$\delta(f - f_0)$')
# plt.axvline(x=5, color='black', linestyle='dashed', linewidth=2, label=r'$f=f_0$')
plt.xlabel(r'$f$', fontsize=12)
plt.ylabel('Magnitude', fontsize=12)
plt.title('Original Sinc Function Centered at $f=f_0$', fontsize=14)
# plt.grid(True)
# plt.legend()
plt.axis('off')

# Original sinc function centered at f=5
plt.subplot(5, 1, 3)
plt.plot(f, original_sinc, color='royalblue', linewidth=linewidth, label=r'$\mathrm{sinc}(f - f_0)$')
# plt.axvline(x=5, color='black', linestyle='dashed', linewidth=2, label=r'$f=f_0$')
plt.xlabel(r'$f$', fontsize=12)
plt.ylabel('Magnitude', fontsize=12)
plt.title('Original Sinc Function Centered at $f=f_0$', fontsize=14)
# plt.grid(True)
# plt.legend()
plt.axis('off')

# Hann window Fourier transform and Gaussian approximation
plt.subplot(5, 1, 4)
# plt.plot(f, main_sinc, color='black', linewidth=2, label=r'$T \, \mathrm{sinc}(\omega T)$')
# plt.plot(f, shifted_sinc_plus, color='green', linewidth=2, label=r'$\frac{1}{2} T \, \mathrm{sinc}((\omega + \pi/T)T)$')
# plt.plot(f, shifted_sinc_minus, color='blue', linewidth=2, label=r'$\frac{1}{2} T \, \mathrm{sinc}((\omega - \pi/T)T)$')
plt.plot(f, hann_spectrum, color='royalblue', linewidth=linewidth, label=r'Hamm Window Spectrum')
# plt.plot(f, gaussian_approx, color='purple', linewidth=linewidth, label=r'Gaussian Approximation')
# plt.axvline(x=5, color='black', linestyle='dashed', linewidth=2, label=r'$f=f_0$')
plt.xlabel(r'$\omega$', fontsize=12)
plt.ylabel(r'$W(\omega)$', fontsize=12)
plt.title('Fourier Transform of Hann Window', fontsize=14)
# plt.grid(True)
# plt.legend()
plt.axis('off')

# Hann window Fourier transform and Gaussian approximation
plt.subplot(5, 1, 5)
# plt.plot(f, main_sinc, color='black', linewidth=2, label=r'$T \, \mathrm{sinc}(\omega T)$')
# plt.plot(f, shifted_sinc_plus, color='green', linewidth=2, label=r'$\frac{1}{2} T \, \mathrm{sinc}((\omega + \pi/T)T)$')
# plt.plot(f, shifted_sinc_minus, color='blue', linewidth=2, label=r'$\frac{1}{2} T \, \mathrm{sinc}((\omega - \pi/T)T)$')
# plt.plot(f, hann_spectrum, color='royalblue', linewidth=linewidth, label=r'Hamm Window Spectrum')
plt.plot(f, gaussian_approx, color='purple', linewidth=linewidth, label=r'Gaussian Approximation')
# plt.axvline(x=5, color='black', linestyle='dashed', linewidth=2, label=r'$f=f_0$')
plt.xlabel(r'$\omega$', fontsize=12)
plt.ylabel(r'$W(\omega)$', fontsize=12)
plt.title('Gaussian Approximation', fontsize=14)
# plt.grid(True)
# plt.legend()
plt.axis('off')

plt.tight_layout()
# plt.show()

plt.savefig('./power_leakage.png')
