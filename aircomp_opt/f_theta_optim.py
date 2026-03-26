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


def _compute_eta_proxy(h_eff, user_weights, update_vars, tx_power, noise_std, eps):
    inner2 = np.abs(h_eff) ** 2 + eps
    eta_candidates = tx_power * inner2 / (np.square(user_weights) * update_vars + eps)
    eta = float(np.min(eta_candidates).real)
    noise_power = float((noise_std ** 2) * tx_power)
    nmse_proxy = noise_power / (eta * (np.square(user_weights.sum()) + eps))
    return eta_candidates, eta, nmse_proxy


def _build_affine_channel_maps(H_BR, h_RUs, h_BUs, reflect_on, direct_on):
    """
    Map the current effective channel model to the affine form used by the SCA example:
        h_k(theta) = h_d,k + G_k theta
    where G_k = H_BR^H diag(h_RU,k).
    """
    k_users, n_ris = h_RUs.shape
    m_bs = H_BR.shape[1]
    h_direct = np.zeros((m_bs, k_users), dtype=np.complex128)
    g_maps = np.zeros((m_bs, n_ris, k_users), dtype=np.complex128)
    H_herm = H_BR.conj().T.astype(np.complex128, copy=False)

    for user_idx in range(k_users):
        if direct_on and h_BUs is not None:
            h_direct[:, user_idx] = np.asarray(h_BUs[user_idx], dtype=np.complex128)
        if reflect_on:
            g_maps[:, :, user_idx] = H_herm * np.asarray(h_RUs[user_idx], dtype=np.complex128)[None, :]
    return h_direct, g_maps


def _solve_sca_mu(a_mat, b_mat, c_vec, k2_eff, mu_init, eps):
    from scipy.optimize import minimize

    def objective(mu_vec):
        linear_term = float(np.dot(c_vec, mu_vec))
        return float(2.0 * np.linalg.norm(a_mat @ mu_vec) + 2.0 * np.linalg.norm(b_mat @ mu_vec, ord=1) - linear_term)

    constraints = ({'type': 'eq', 'fun': lambda mu_vec: float(np.dot(k2_eff, mu_vec) - 1.0)},)
    bounds = [(0.0, None) for _ in range(k2_eff.size)]
    res = minimize(
        objective,
        mu_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9, "disp": False},
    )

    if res.x is None:
        mu_vec = np.asarray(mu_init, dtype=np.float64)
    else:
        mu_vec = np.maximum(np.asarray(res.x, dtype=np.float64), 0.0)
    denom = float(np.dot(k2_eff, mu_vec))
    if denom <= eps:
        mu_vec = np.asarray(mu_init, dtype=np.float64)
        denom = float(np.dot(k2_eff, mu_vec))
    mu_vec = mu_vec / max(denom, eps)
    return mu_vec


def optimize_beam_ris(H_BR, h_RUs, h_BUs=None, theta_init=None, f_init=None, link_switch=(1, 0), user_weights=None,
                      update_vars=None, tx_power=0.1, noise_std=0.1, var_floor=1e-3, eps=1e-8, oa_iters=5):
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

    iterations = int(max(1, oa_iters))
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
        eigvals, eigvecs = np.linalg.eigh(scatter)
        f_vec = eigvecs[:, np.argmax(eigvals)]
        f_vec = _normalize_vector(f_vec, fallback=f_init)

    _, h_eff, _ = _compute_effective_channels(H_BR, h_RUs, f_vec, theta_vec, h_BUs, reflect_on, direct_on)
    _, _, nmse_proxy = _compute_eta_proxy(
        h_eff=h_eff,
        user_weights=user_weights,
        update_vars=update_vars,
        tx_power=tx_power,
        noise_std=noise_std,
        eps=eps,
    )

    return f_vec.astype(np.complex64), theta_vec.astype(np.complex64), nmse_proxy


def optimize_beam_ris_sca(H_BR, h_RUs, h_BUs=None, theta_init=None, f_init=None, link_switch=(1, 0), user_weights=None,
                          update_vars=None, tx_power=0.1, noise_std=0.1, var_floor=1e-3, eps=1e-8,
                          sca_iters=100, sca_threshold=1e-2, sca_tau=1.0):
    """
    SCA version adapted from the reference `sca_fmincon()` implementation, but aligned
    with the current OTA model by absorbing per-user update variance into the effective
    denominator K_i^2 * var_i.
    """
    k_users, n_ris = h_RUs.shape
    m_bs = H_BR.shape[1]
    reflect_on, direct_on = int(link_switch[0]), int(link_switch[1])

    if user_weights is None:
        user_weights = np.ones(k_users, dtype=np.float32)
    user_weights = np.asarray(user_weights, dtype=np.float64).reshape(-1)

    if update_vars is None:
        update_vars = np.ones(k_users, dtype=np.float32)
    update_vars = np.maximum(np.asarray(update_vars, dtype=np.float64).reshape(-1), float(var_floor))
    k2_eff = np.maximum(np.square(user_weights) * update_vars, eps)

    h_direct, g_maps = _build_affine_channel_maps(H_BR, h_RUs, h_BUs, reflect_on, direct_on)

    if reflect_on:
        if theta_init is None:
            theta_vec = np.ones((n_ris,), dtype=np.complex128)
        else:
            theta_arr = np.asarray(theta_init, dtype=np.complex128).reshape(-1)
            theta_vec = np.exp(1j * np.angle(theta_arr))
    else:
        theta_vec = np.zeros((n_ris,), dtype=np.complex128)

    h_stack = np.zeros((m_bs, k_users), dtype=np.complex128)
    for user_idx in range(k_users):
        h_stack[:, user_idx] = h_direct[:, user_idx] + g_maps[:, :, user_idx] @ theta_vec

    if f_init is None:
        f_seed = h_stack[:, 0]
    else:
        f_seed = np.asarray(f_init, dtype=np.complex128).reshape(-1)
    f_vec = _normalize_vector(f_seed).astype(np.complex128)

    mu_vec = 1.0 / (k_users * k2_eff)
    obj_prev = float(np.min(np.abs(np.conj(f_vec) @ h_stack) ** 2 / k2_eff))

    for _ in range(int(max(1, sca_iters))):
        f_outer = np.outer(f_vec, np.conj(f_vec))
        a_mat = np.zeros((m_bs, k_users), dtype=np.complex128)
        b_mat = np.zeros((n_ris, k_users), dtype=np.complex128)
        c_vec = np.zeros((k_users,), dtype=np.float64)

        for user_idx in range(k_users):
            hk = h_stack[:, user_idx]
            gain = float(np.abs(np.conj(f_vec) @ hk) ** 2)
            a_mat[:, user_idx] = float(sca_tau) * k2_eff[user_idx] * f_vec + np.outer(hk, np.conj(hk)) @ f_vec
            if reflect_on:
                gk = g_maps[:, :, user_idx]
                b_mat[:, user_idx] = float(sca_tau) * k2_eff[user_idx] * theta_vec + gk.conj().T @ f_outer @ hk
                c_vec[user_idx] = (
                    gain
                    + 2.0 * float(sca_tau) * k2_eff[user_idx] * (n_ris + 1.0)
                    + 2.0 * np.real(theta_vec.conj().T @ gk.conj().T @ f_outer @ hk)
                )
            else:
                c_vec[user_idx] = gain + 2.0 * float(sca_tau) * k2_eff[user_idx]

        mu_vec = _solve_sca_mu(a_mat, b_mat, c_vec, k2_eff, mu_vec, eps)

        f_candidate = a_mat @ mu_vec
        f_vec = _normalize_vector(f_candidate, fallback=f_vec).astype(np.complex128)

        if reflect_on:
            theta_candidate = b_mat @ mu_vec
            mag = np.abs(theta_candidate)
            theta_safe = np.ones_like(theta_candidate, dtype=np.complex128)
            valid = mag > eps
            theta_safe[valid] = theta_candidate[valid] / mag[valid]
            theta_vec = theta_safe

        for user_idx in range(k_users):
            h_stack[:, user_idx] = h_direct[:, user_idx] + g_maps[:, :, user_idx] @ theta_vec

        obj_new = float(np.min(np.abs(np.conj(f_vec) @ h_stack) ** 2 / k2_eff))
        rel_gap = np.abs(obj_new - obj_prev) / max(1.0, np.abs(obj_new))
        obj_prev = obj_new
        if rel_gap <= float(sca_threshold):
            break

    _, h_eff, _ = _compute_effective_channels(
        H_BR,
        h_RUs,
        f_vec.astype(np.complex64),
        theta_vec.astype(np.complex64),
        h_BUs,
        reflect_on,
        direct_on,
    )
    _, _, nmse_proxy = _compute_eta_proxy(
        h_eff=h_eff,
        user_weights=user_weights,
        update_vars=update_vars,
        tx_power=tx_power,
        noise_std=noise_std,
        eps=eps,
    )
    return f_vec.astype(np.complex64), theta_vec.astype(np.complex64), nmse_proxy


def optimize_beam_ris_by_mode(mode="oa", **kwargs):
    optimizer_mode = str(mode).strip().lower()
    if optimizer_mode == "oa":
        kwargs.pop("sca_iters", None)
        kwargs.pop("sca_threshold", None)
        kwargs.pop("sca_tau", None)
        return optimize_beam_ris(**kwargs)
    if optimizer_mode == "sca":
        kwargs.pop("oa_iters", None)
        return optimize_beam_ris_sca(**kwargs)
    raise ValueError("optimizer mode must be 'oa' or 'sca'")
