import numpy as np


def estimate_h_ru_lmmse(
    Y_pilot,
    H_BR,
    f_beam,
    theta_pattern,
    noise_std,
    *,
    h_BU=None,
    link_switch=(1, 0),
    prior_var=1.0,
    eps=1e-8,
):
    """
    Closed-form LMMSE estimator for h_RU under the current pilot observation model:
        y = Theta * diag(f^H H_BR^T) * h_RU + direct + noise

    The prior is modeled as h_RU ~ CN(0, prior_var * I). When direct-link CSI is
    available it is subtracted before reflection-channel estimation.
    """
    reflect_on, direct_on = int(link_switch[0]), int(link_switch[1])
    n_ris = int(H_BR.shape[0])
    if reflect_on == 0:
        return np.zeros((n_ris,), dtype=np.complex64)

    y_vec = np.asarray(Y_pilot, dtype=np.complex128).reshape(-1)
    theta_mat = np.asarray(theta_pattern, dtype=np.complex128)
    beam_vec = np.asarray(f_beam, dtype=np.complex128).reshape(-1)
    h_br = np.asarray(H_BR, dtype=np.complex128)

    if direct_on == 1 and h_BU is not None:
        direct_eff = beam_vec.conj().dot(np.asarray(h_BU, dtype=np.complex128).reshape(-1))
        y_vec = y_vec - direct_eff

    cascaded_prefix = beam_vec.conj() @ h_br.T  # (N,)
    sensing = theta_mat * cascaded_prefix[None, :]  # (P, N)

    noise_var = float(noise_std) ** 2
    prior_var = max(float(prior_var), float(eps))
    gram = sensing @ sensing.conj().T
    reg = (noise_var / prior_var) + float(eps)
    system = gram + reg * np.eye(gram.shape[0], dtype=np.complex128)
    gain = sensing.conj().T @ np.linalg.solve(system, y_vec)
    return np.asarray(gain, dtype=np.complex64)
