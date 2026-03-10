import numpy as np
import torch


def flatten_state_dict(state_dict):
    """Flatten a state-dict (e.g., per-user head) into a 1D numpy vector."""
    parts = []
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if isinstance(tensor, torch.Tensor):
            parts.append(tensor.detach().cpu().reshape(-1).float())
        else:
            parts.append(torch.tensor(tensor, dtype=torch.float32).reshape(-1))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return torch.cat(parts, dim=0).numpy()


def compute_head_distance(head_state_user, head_state_global, eps=1e-8):
    """Normalized head displacement d_k = ||w_k - w_global|| / (||w_global|| + eps)."""
    user_vec = flatten_state_dict(head_state_user)
    global_vec = flatten_state_dict(head_state_global)
    if user_vec.shape != global_vec.shape:
        raise ValueError(
            f"Head shape mismatch: user={user_vec.shape}, global={global_vec.shape}"
        )
    delta = np.linalg.norm(user_vec - global_vec)
    denom = np.linalg.norm(global_vec) + float(eps)
    return float(delta / denom)


def update_ewma_state(prev_value, current_value, beta):
    """EWMA update z <- (1-beta) * z + beta * current."""
    return float((1.0 - beta) * float(prev_value) + beta * float(current_value))


def compute_hidden_drift_ratio(hidden_prev, hidden_next, eps=1e-8, x_max=None):
    """
    Fast state proxy:
      x = ||h_t - h_{t-1}|| / (||h_{t-1}|| + eps), optional clipping at x_max.
    """
    if hidden_next is None:
        return 0.0
    vec_next = hidden_next.detach().cpu().reshape(-1).float().numpy()
    if hidden_prev is None:
        ratio = 0.0
    else:
        vec_prev = hidden_prev.detach().cpu().reshape(-1).float().numpy()
        ratio = float(np.linalg.norm(vec_next - vec_prev) / (np.linalg.norm(vec_prev) + float(eps)))
    if x_max is not None:
        ratio = min(ratio, float(x_max))
    return float(ratio)


def combine_state_scores(z_state, x_state, alpha_z=0.5, alpha_x=0.5, eps=1e-8):
    """
    eta_k = alpha_z * z_k / mean(z) + alpha_x * x_k / mean(x)
    Returns eta, z_bar, x_bar.
    """
    z_arr = np.asarray(z_state, dtype=np.float32).reshape(-1)
    x_arr = np.asarray(x_state, dtype=np.float32).reshape(-1)
    if z_arr.shape != x_arr.shape:
        raise ValueError(f"z/x shape mismatch: z={z_arr.shape}, x={x_arr.shape}")

    z_bar = float(np.mean(z_arr)) if z_arr.size > 0 else 0.0
    x_bar = float(np.mean(x_arr)) if x_arr.size > 0 else 0.0
    z_rel = z_arr / (z_bar + float(eps))
    x_rel = x_arr / (x_bar + float(eps))
    eta = float(alpha_z) * z_rel + float(alpha_x) * x_rel
    return eta.astype(np.float32), z_bar, x_bar


def build_state_weights(eta_state, mu=0.5, weight_min=0.5, weight_max=2.0, strategy="protect"):
    """
    Map eta to omega:
      protect:  omega = 1 + mu * eta
      stability: omega = 1 / (1 + mu * eta)
    Then clip to [weight_min, weight_max].
    """
    eta_arr = np.asarray(eta_state, dtype=np.float32).reshape(-1)
    mode = str(strategy).strip().lower()
    if mode == "protect":
        omega_raw = 1.0 + float(mu) * eta_arr
    elif mode == "stability":
        omega_raw = 1.0 / (1.0 + float(mu) * eta_arr)
    else:
        raise ValueError(f"Unknown strategy={strategy}; expected 'protect' or 'stability'")
    omega = np.clip(omega_raw, float(weight_min), float(weight_max))
    return omega.astype(np.float32), omega_raw.astype(np.float32)
