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


def _build_hk_list(H_BR, h_RUs, theta_vec, h_BUs, reflect_on, direct_on):
    """Build per-user equivalent uplink channels hk(theta) at BS."""
    hk_list = []
    for user_idx in range(h_RUs.shape[0]):
        hk = np.zeros(H_BR.shape[1], dtype=np.complex64)
        if reflect_on:
            hk = hk + H_BR.conj().T.dot(theta_vec * h_RUs[user_idx])
        if direct_on and h_BUs is not None:
            hk = hk + h_BUs[user_idx]
        hk_list.append(hk.astype(np.complex64))
    return hk_list


def _update_f_closed_form(hk_list, b_vec, omega_vec, noise_var, eps=1e-8, normalize_f=True, f_fallback=None):
    """
    Closed-form update:
      f* = (sum_k omega_k |b_k|^2 h_k h_k^H + noise_var I)^(-1) sum_k omega_k b_k* h_k
    """
    if not hk_list:
        raise ValueError("hk_list is empty")
    m = hk_list[0].size
    scatter = noise_var * np.eye(m, dtype=np.complex128)
    rhs = np.zeros((m,), dtype=np.complex128)
    for user_idx, hk in enumerate(hk_list):
        omega_k = float(omega_vec[user_idx])
        b_k = b_vec[user_idx]
        scatter += omega_k * (np.abs(b_k) ** 2) * np.outer(hk, hk.conj())
        rhs += omega_k * np.conj(b_k) * hk
    try:
        f_vec = np.linalg.solve(scatter, rhs)
    except np.linalg.LinAlgError:
        f_vec = np.linalg.pinv(scatter, rcond=float(eps)).dot(rhs)
    f_vec = f_vec.astype(np.complex64)
    if normalize_f:
        f_vec = _normalize_vector(f_vec, fallback=f_fallback)
    return f_vec


def _update_theta_phase_gradient(H_BR, h_RUs, h_BUs, f_vec, theta_vec, b_vec, omega_vec,
                                 reflect_on, direct_on, theta_lr, theta_grad_steps):
    """
    Projected gradient update on RIS phases:
      theta_n = exp(j * phi_n), optimize over real-valued phi.
    """
    if not reflect_on or int(theta_grad_steps) <= 0:
        return theta_vec.astype(np.complex64)

    k_users, n_ris = h_RUs.shape
    g_vec = H_BR.dot(f_vec).astype(np.complex64)  # (N,)
    # Reflect term: f^H H^H (theta * h_ru,k) = sum_n theta_n * (conj(g_n) * h_ru,k,n)
    v_kn = (np.conj(g_vec)[None, :] * h_RUs).astype(np.complex64)  # (K, N)

    direct_terms = np.zeros((k_users,), dtype=np.complex64)
    if direct_on and h_BUs is not None:
        for user_idx in range(k_users):
            direct_terms[user_idx] = f_vec.conj().dot(h_BUs[user_idx])

    phi = np.angle(theta_vec).astype(np.float64)
    theta = np.exp(1j * phi).astype(np.complex64)
    b_arr = np.asarray(b_vec, dtype=np.complex64).reshape(-1)
    omega_arr = np.asarray(omega_vec, dtype=np.float64).reshape(-1)
    lr = float(theta_lr)

    for _ in range(int(theta_grad_steps)):
        reflect_terms = np.sum(theta[None, :] * v_kn, axis=1)  # (K,)
        a_k = b_arr * (direct_terms + reflect_terms)
        err_k = a_k - 1.0
        # dE/dphi_n = 2 * Re( sum_k omega_k * conj(err_k) * j*b_k*theta_n*v_kn )
        term_kn = (np.conj(err_k) * b_arr)[:, None] * (theta[None, :] * v_kn)
        grad_phi = 2.0 * np.real(1j * (omega_arr[:, None] * term_kn).sum(axis=0))
        phi = phi - lr * grad_phi
        theta = np.exp(1j * phi).astype(np.complex64)

    return theta.astype(np.complex64)


def _compute_state_aware_objective(hk_list, f_vec, b_vec, omega_vec, noise_var):
    residual = np.zeros((len(hk_list),), dtype=np.complex64)
    for user_idx, hk in enumerate(hk_list):
        residual[user_idx] = b_vec[user_idx] * f_vec.conj().dot(hk) - 1.0
    return float(np.sum(omega_vec * (np.abs(residual) ** 2)) + noise_var * (np.linalg.norm(f_vec) ** 2))


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
        eigvals, eigvecs = np.linalg.eigh(scatter)
        f_vec = eigvecs[:, np.argmax(eigvals)]
        f_vec = _normalize_vector(f_vec, fallback=f_init)

    _, h_eff, _ = _compute_effective_channels(H_BR, h_RUs, f_vec, theta_vec, h_BUs, reflect_on, direct_on)
    inner2 = np.abs(h_eff) ** 2 + eps
    eta_candidates = tx_power * inner2 / (np.square(user_weights) * update_vars + eps)
    eta = float(np.min(eta_candidates).real)
    noise_power = float((noise_std ** 2) * tx_power)
    nmse_proxy = noise_power / (eta * (np.square(user_weights.sum()) + eps))

    return f_vec.astype(np.complex64), theta_vec.astype(np.complex64), nmse_proxy


def optimize_beam_ris_state_aware(H_BR, h_RUs, h_BUs=None, theta_init=None, f_init=None, link_switch=(1, 0),
                                  user_weights=None, state_weights=None, update_vars=None, tx_power=0.1,
                                  noise_std=0.1, var_floor=1e-3, ao_iters=2, theta_lr=0.05,
                                  theta_grad_steps=1, normalize_f=True, eps=1e-8):
    """
    State-aware OA optimization:
      min_{f,theta} sum_k omega_k |f^H h_k^eff(theta) b_k - 1|^2 + noise_var ||f||^2
      s.t. |theta_n|=1.

    b_k is passed by user_weights (e.g., K_k normalization).
    omega_k is passed by state_weights derived from slow+fast user states.
    """
    k_users, n_ris = h_RUs.shape
    m_bs = H_BR.shape[1]
    reflect_on, direct_on = int(link_switch[0]), int(link_switch[1])

    if user_weights is None:
        user_weights = np.ones(k_users, dtype=np.float32)
    b_vec = np.asarray(user_weights, dtype=np.float32).reshape(-1).astype(np.complex64)
    if b_vec.size != k_users:
        raise ValueError(f"user_weights size mismatch: expected {k_users}, got {b_vec.size}")

    if state_weights is None:
        state_weights = np.ones(k_users, dtype=np.float32)
    omega_vec = np.asarray(state_weights, dtype=np.float32).reshape(-1)
    if omega_vec.size != k_users:
        raise ValueError(f"state_weights size mismatch: expected {k_users}, got {omega_vec.size}")
    omega_vec = np.maximum(omega_vec, 1e-6).astype(np.float64)

    if update_vars is None:
        update_vars = np.ones(k_users, dtype=np.float32)
    update_vars = np.maximum(np.asarray(update_vars, dtype=np.float32), float(var_floor))

    f_vec = _normalize_vector(f_init if f_init is not None else np.ones(m_bs, dtype=np.complex64))
    if theta_init is None:
        theta_vec = np.ones(n_ris, dtype=np.complex64)
    else:
        theta_vec = np.exp(1j * np.angle(np.asarray(theta_init, dtype=np.complex64))).astype(np.complex64)

    noise_var = float(noise_std ** 2)
    for _ in range(int(max(1, ao_iters))):
        hk_list = _build_hk_list(H_BR, h_RUs, theta_vec, h_BUs, reflect_on, direct_on)
        f_vec = _update_f_closed_form(
            hk_list=hk_list,
            b_vec=b_vec,
            omega_vec=omega_vec,
            noise_var=noise_var,
            eps=eps,
            normalize_f=bool(normalize_f),
            f_fallback=f_init,
        )
        theta_vec = _update_theta_phase_gradient(
            H_BR=H_BR,
            h_RUs=h_RUs,
            h_BUs=h_BUs,
            f_vec=f_vec,
            theta_vec=theta_vec,
            b_vec=b_vec,
            omega_vec=omega_vec,
            reflect_on=reflect_on,
            direct_on=direct_on,
            theta_lr=theta_lr,
            theta_grad_steps=theta_grad_steps,
        )

    hk_final = _build_hk_list(H_BR, h_RUs, theta_vec, h_BUs, reflect_on, direct_on)
    obj_val = _compute_state_aware_objective(hk_final, f_vec, b_vec, omega_vec, noise_var)
    h_eff = np.asarray([f_vec.conj().dot(hk) for hk in hk_final], dtype=np.complex64)
    inner2 = np.abs(h_eff) ** 2 + eps
    eta_candidates = tx_power * inner2 / (np.square(np.abs(b_vec)) * update_vars + eps)
    eta = float(np.min(eta_candidates).real)
    noise_power = float((noise_std ** 2) * tx_power)
    nmse_proxy = noise_power / (eta * (np.square(np.sum(np.abs(b_vec))) + eps))

    # Keep objective available to callers that need diagnostics in the future.
    _ = obj_val
    return f_vec.astype(np.complex64), theta_vec.astype(np.complex64), float(nmse_proxy)
