import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

def fit_spherical_harmonics_axisymmetric(theta, power_data, L_max):
    """
    Fit an axisymmetric spherical harmonics model (m=0 only) to given data.
    Here, f(theta) does not depend on phi.

    Parameters
    ----------
    theta : array_like
        Array of polar angles (0 <= theta <= 2*pi).
    power_data : array_like
        Measured radar power values at corresponding theta directions.
    L_max : int
        Maximum spherical harmonics degree.

    Returns
    -------
    coeffs : ndarray
        Array of spherical harmonic coefficients c_l for l=0...L_max.
    """
    theta = np.array(theta)
    power_data = np.array(power_data)

    N = len(power_data)
    # Number of terms: (L_max+1) because we only take m=0 for each l
    n_terms = L_max + 1

    # Construct design matrix
    M = np.zeros((N, n_terms), dtype=complex)

    # Choose phi = 0 (arbitrary since no phi dependence)
    phi = np.zeros_like(theta)

    # Fill the design matrix with Y_l^0(theta, 0)
    for l in range(L_max + 1):
        Y_l0 = sph_harm(0, l, phi, theta)
        M[:, l] = Y_l0

    # Solve the least squares problem: M * coeffs = power_data
    # power_data is real, M is complex but for m=0, Y_l^0 are real-valued.
    # Spherical harmonics with m=0 should yield real values, so we can take real part just in case.
    coeffs_complex, residuals, rank, s = np.linalg.lstsq(M.real, power_data, rcond=None)
    coeffs = coeffs_complex

    return coeffs

def evaluate_spherical_harmonics_axisymmetric(theta, coeffs):
    """
    Evaluate the axisymmetric spherical harmonic expansion at given theta values.

    Parameters
    ----------
    theta : array_like
        Array of polar angles.
    coeffs : ndarray
        Coefficients c_l corresponding to Y_l^0 terms.

    Returns
    -------
    values : ndarray
        Evaluated function at theta.
    """
    theta = np.array(theta)
    phi = np.zeros_like(theta)  # phi can be anything, taken as 0
    values = np.zeros_like(theta, dtype=complex)
    
    L_max = len(coeffs) - 1
    for l, c_l in enumerate(coeffs):
        Y_l0 = sph_harm(0, l, phi, theta)
        values += c_l * Y_l0

    return values.real  # should already be real for m=0


def vis_theta_power_plot(theta_samples, power_data):
    # Create a polar plot
    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')

    # Plot each point as a dot
    ax.scatter(theta_samples, power_data, c='blue', s=20)

    # Optional: Customize the plot
    ax.set_title("Power vs Theta in Polar Coordinates")
    ax.set_rlabel_position(-22.5)  # move radial labels away from plotted line
    ax.grid(True)

    plt.show()

np.random.seed(42)

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data for demonstration:
    # Suppose f(theta) = 1.0*Y_0^0 + 0.5*Y_2^0(theta) (just as an example)
    N_samples = 100
    theta_samples = np.linspace(0, 2*np.pi, N_samples)
    L_max_true = 2

    # True coefficients (l=0,1,2 for demonstration)
    # Y_0^0 = 1/(2*sqrt(pi))
    # Y_2^0 = sqrt(5/(16*pi))*(3cos^2(theta)-1)
    # We'll pick something simple: c_0 = 1.0, c_1 = 0.0, c_2 = 0.5, etc.
    true_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  

    # Evaluate true model
    power_data = evaluate_spherical_harmonics_axisymmetric(theta_samples, true_coeffs)
    power_data_noisy = power_data #+ 0.001 * np.random.randn(N_samples)
    
    power_data_noisy = np.zeros_like(theta_samples)
    power_data_noisy[:] = 0.0
    power_data_noisy[:20] = 0.1

    # theta_samples = np.linspace(0, 2*np.pi, 200)  # theta from 0 to pi
    # power_data = np.abs(np.sin(theta_samples))  # some example power pattern
    # vis_theta_power_plot(theta_samples, power_data)
    vis_theta_power_plot(theta_samples, power_data_noisy)

    # Fit the model up to L_max=2
    L_max_fit = 10 #3 #9
    fitted_coeffs = fit_spherical_harmonics_axisymmetric(theta_samples, power_data_noisy, L_max_fit)

    # print("True coefficients:", true_coeffs)
    print("Fitted coefficients:", fitted_coeffs)

    # Evaluate fitted model
    fitted_values = evaluate_spherical_harmonics_axisymmetric(theta_samples, fitted_coeffs)
    mse = np.mean((fitted_values - power_data_noisy)**2)
    print("Mean squared error of the fit:", mse)

    power_data_fit = evaluate_spherical_harmonics_axisymmetric(theta_samples, fitted_coeffs)

    vis_theta_power_plot(theta_samples, power_data_fit)
