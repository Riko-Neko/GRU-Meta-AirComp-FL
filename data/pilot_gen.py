import numpy as np


def generate_pilot_pattern(P, N):
    """
    Generate a pilot phase pattern matrix of shape (P, N) for RIS.
    Each row corresponds to one pilot transmission configuration of RIS phases.
    """
    # Random continuous phase pattern
    pattern = np.exp(1j * 2 * np.pi * np.random.rand(P, N))
    return pattern


def simulate_pilot_observation(H_BR, h_RU, f_beam, theta_pattern, noise_std=0.0, *, h_BU=None, link_switch=(1, 0)):
    """
    Simulate the pilot transmission for one user at current channels.
    - H_BR: numpy array of shape (N, M) for BS-RIS channel.
    - h_RU: numpy array of shape (N,) for RIS-user channel.
    - f_beam: numpy array of shape (M,) beamforming vector at BS for pilot.
    - theta_pattern: numpy array of shape (P, N) for RIS phase configurations (P pilots).
    - h_BU: optional numpy array of shape (M,) for BS-user direct channel.
    - link_switch: (reflect, direct) binary switch, e.g., (1,0), (0,1), (1,1).
    - noise_std: standard deviation of complex Gaussian noise.
    Returns:
    - Y_pilot: numpy array of shape (P,) complex received pilot observations.
    - cascaded_channel: numpy array of shape (N,) complex cascaded channel (element-wise BS-RIS-user product).
    - direct_effective: complex scalar direct-link projection after beamforming (f^H h_d).
    """
    if link_switch is None:
        link_switch = (1, 0)
    if len(link_switch) != 2:
        raise ValueError("link_switch must be length-2: [reflect, direct]")
    reflect_on, direct_on = int(link_switch[0]), int(link_switch[1])
    if (reflect_on not in (0, 1)) or (direct_on not in (0, 1)):
        raise ValueError("link_switch elements must be 0 or 1")
    if reflect_on == 0 and direct_on == 0:
        raise ValueError(f"Link is invalid: {link_switch}")

    # Effective BS->RIS channel seen after receive beamforming: f^H H_BR
    cascaded_prefix = f_beam.conj() @ H_BR.T  # shape (N,)
    # Cascaded channel per RIS element after beamforming
    cascaded = cascaded_prefix * h_RU  # shape (N,), complex

    if reflect_on == 0:
        cascaded = np.zeros_like(cascaded)
    # Simulate P pilot transmissions with different RIS phase configurations
    P, N = theta_pattern.shape
    # Received pilot for each transmission: sum_n theta_{i,n} * cascaded_n + noise
    # Using broadcasting: (theta_pattern * cascaded) sums over N for each row
    # Compute dot product of each pilot row with cascaded channel
    Y = np.zeros((P,), dtype=np.result_type(theta_pattern, cascaded))

    if reflect_on == 1:
        Y = Y + theta_pattern.dot(cascaded)  # shape (P,), complex

    direct_eff = 0.0 + 0.0j
    if direct_on == 1 and h_BU is not None:
        # Direct link projected by receive beamformer: f^H h_d
        direct_eff = f_beam.conj().dot(h_BU)
    if direct_on == 1:
        Y = Y + direct_eff

    if noise_std > 0:
        noise = (np.random.randn(P) + 1j * np.random.randn(P)) * (noise_std / np.sqrt(2))
        Y = Y + noise
    return Y, cascaded, direct_eff
