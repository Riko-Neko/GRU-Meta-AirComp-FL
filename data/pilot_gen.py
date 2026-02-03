import numpy as np


def generate_pilot_pattern(P, N):
    """
    Generate a pilot phase pattern matrix of shape (P, N) for RIS.
    Each row corresponds to one pilot transmission configuration of RIS phases.
    """
    # Random continuous phase pattern
    pattern = np.exp(1j * 2 * np.pi * np.random.rand(P, N))
    return pattern


def simulate_pilot_observation(H_BR, h_RU, f_beam, theta_pattern, noise_std=0.0):
    """
    Simulate the pilot transmission for one user at current channels.
    - H_BR: numpy array of shape (N, M) for BS-RIS channel.
    - h_RU: numpy array of shape (N,) for RIS-user channel.
    - f_beam: numpy array of shape (M,) beamforming vector at BS for pilot.
    - theta_pattern: numpy array of shape (P, N) for RIS phase configurations (P pilots).
    - noise_std: standard deviation of complex Gaussian noise.
    Returns:
    - Y_pilot: numpy array of shape (P,) complex received pilot observations.
    - cascaded_channel: numpy array of shape (N,) complex cascaded channel (element-wise BS-RIS-user product).
    """
    # Effective BS->RIS channel with beamforming
    # If BS has multiple antennas, combine with f_beam
    g = H_BR.dot(f_beam)  # shape (N,)
    # Cascaded channel per RIS element: product of BS-RIS and RIS-user channels
    cascaded = g * h_RU  # shape (N,), complex
    # Simulate P pilot transmissions with different RIS phase configurations
    P, N = theta_pattern.shape
    # Received pilot for each transmission: sum_n theta_{i,n} * cascaded_n + noise
    # Using broadcasting: (theta_pattern * cascaded) sums over N for each row
    # Compute dot product of each pilot row with cascaded channel
    Y = theta_pattern.dot(cascaded)  # shape (P,), complex
    if noise_std > 0:
        noise = (np.random.randn(P) + 1j * np.random.randn(P)) * (noise_std / np.sqrt(2))
        Y = Y + noise
    return Y, cascaded



