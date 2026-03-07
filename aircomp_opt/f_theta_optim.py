import numpy as np


def _normalize_vector(vec, fallback=None):
    norm = np.linalg.norm(vec)
    if norm > 0:
        return (vec / norm).astype(np.complex64)
    if fallback is None:
        fallback = np.ones_like(vec, dtype=np.complex64)
    fallback = np.asarray(fallback, dtype=np.complex64)
    fallback_norm = np.linalg.norm(fallback)
    if fallback_norm == 0:
        return np.ones_like(vec, dtype=np.complex64)
    return (fallback / fallback_norm).astype(np.complex64)


def _compute_effective_channels(H_BR, h_RUs, f_vec, theta_vec, h_BUs, reflect_on, direct_on):
    g_vec = H_BR.dot(f_vec)
    h_eff = np.zeros(h_RUs.shape[0], dtype=np.complex64)
    hk_list = []
    for user_idx in range(h_RUs.shape[0]):
        hk = np.zeros(H_BR.shape[1], dtype=np.complex64)
        if reflect_on:
            hk = hk + H_BR.conj().T.dot(theta_vec * h_RUs[user_idx])
        if direct_on and h_BUs is not None:
            hk = hk + h_BUs[user_idx]
        hk_list.append(hk.astype(np.complex64))
        h_eff[user_idx] = f_vec.conj().dot(hk)
    return g_vec.astype(np.complex64), h_eff, hk_list


def optimize_beam_ris(H_BR, h_RUs, h_BUs=None, theta_init=None, f_init=None, link_switch=(1, 0), user_weights=None,
                      update_vars=None, tx_power=0.1, noise_std=0.1, var_floor=1e-3, eps=1e-8):
    """
    OTA-aware heuristic aligned with the current AirComp simulator.
    It uses the current round update variances to maximize the weakest-user eta:
        eta_k = tx_power * |h_eff,k|^2 / (K_k^2 * var_k)
    and reports an aggregation-NMSE proxy derived from this eta.
    """
    K, N = h_RUs.shape
    M = H_BR.shape[1]
    reflect_on, direct_on = int(link_switch[0]), int(link_switch[1])

    if user_weights is None:
        user_weights = np.ones(K, dtype=np.float32)
    user_weights = np.asarray(user_weights, dtype=np.float32)

    if update_vars is None:
        update_vars = np.ones(K, dtype=np.float32)
    update_vars = np.maximum(np.asarray(update_vars, dtype=np.float32), float(var_floor))

    f_vec = _normalize_vector(f_init if f_init is not None else np.ones(M, dtype=np.complex64))
    theta_vec = np.asarray(theta_init, dtype=np.complex64) if theta_init is not None else np.ones(N, dtype=np.complex64)

    iterations = 5
    for _ in range(iterations):
        g_vec, h_eff, hk_list = _compute_effective_channels(
            H_BR, h_RUs, f_vec, theta_vec, h_BUs, reflect_on, direct_on
        )
        inner2 = np.abs(h_eff) ** 2 + eps
        eta_candidates = tx_power * inner2 / (np.square(user_weights) * update_vars + eps)

        # Focus the update on the current weakest users instead of average gain.
        weakness = 1.0 / (eta_candidates + eps)
        weakness = weakness / (np.sum(weakness) + eps)

        if reflect_on:
            phase_acc = np.zeros((N,), dtype=np.complex64)
            for user_idx in range(K):
                phase_acc += weakness[user_idx] * np.conj(g_vec) * h_RUs[user_idx]
            theta_vec = np.exp(-1j * np.angle(phase_acc)).astype(np.complex64)

        _, h_eff, hk_list = _compute_effective_channels(
            H_BR, h_RUs, f_vec, theta_vec, h_BUs, reflect_on, direct_on
        )
        inner2 = np.abs(h_eff) ** 2 + eps
        eta_candidates = tx_power * inner2 / (np.square(user_weights) * update_vars + eps)
        weakness = 1.0 / (eta_candidates + eps)
        weakness = weakness / (np.sum(weakness) + eps)

        scatter = np.zeros((M, M), dtype=np.complex128)
        for user_idx, hk in enumerate(hk_list):
            scatter += weakness[user_idx] * np.outer(hk, hk.conj())
        try:
            eigvals, eigvecs = np.linalg.eigh(scatter)
            f_vec = eigvecs[:, np.argmax(eigvals)]
        except np.linalg.LinAlgError:
            f_vec = hk_list[int(np.argmin(eta_candidates))]
        f_vec = _normalize_vector(f_vec, fallback=f_init)

    _, h_eff, _ = _compute_effective_channels(H_BR, h_RUs, f_vec, theta_vec, h_BUs, reflect_on, direct_on)
    inner2 = np.abs(h_eff) ** 2 + eps
    eta_candidates = tx_power * inner2 / (np.square(user_weights) * update_vars + eps)
    eta = float(np.min(eta_candidates).real)
    noise_power = float((noise_std ** 2) * tx_power)
    nmse_proxy = noise_power / (eta * (np.square(user_weights.sum()) + eps))

    return f_vec.astype(np.complex64), theta_vec.astype(np.complex64), nmse_proxy
