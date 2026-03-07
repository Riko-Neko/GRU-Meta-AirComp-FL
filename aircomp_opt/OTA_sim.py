import torch
import numpy as np


class AirCompSimulator:
    """
    Physical-domain OTA aggregation (Phase1, no device selection).
    Implements mean/var normalization + power-limited pre-equalization proxy.
    """
    def __init__(self, noise_std=0.0, tx_power=0.1, var_floor=1e-3, eps=1e-8):
        self.noise_std = noise_std
        self.tx_power = tx_power
        self.var_floor = var_floor
        self.eps = eps

    def aggregate_updates(self, updates, h_eff, user_weights, noise_std=None):
        """
        updates: tensor [K, d], real
        h_eff: tensor [K], complex, effective channel f^H h_k
        user_weights: tensor [K], usually K_k
        noise_std: optional override
        Returns: agg_update [d], real; diagnostics dict
        """
        if noise_std is None:
            noise_std = self.noise_std
        K, d = updates.shape
        device = updates.device

        # per-user stats
        mean_k = updates.mean(dim=1, keepdim=True)         # [K,1]
        var_k = updates.var(dim=1, keepdim=True, unbiased=False)  # [K,1]
        var_k = torch.clamp(var_k, min=self.var_floor)
        var_sqrt = torch.sqrt(var_k)

        # weights
        K_vec = user_weights.to(device).reshape(K, 1)      # [K,1]

        # eta (alignment / power limit proxy)
        inner2 = torch.abs(h_eff)**2 + self.eps            # [K]
        eta_candidates = self.tx_power * inner2 / (K_vec.view(-1) ** 2 * var_k.view(-1))
        eta = torch.min(eta_candidates).real
        eta_sqrt = torch.sqrt(torch.clamp(eta, min=self.eps))

        # pre-equalization
        b_k = K_vec * eta_sqrt * var_sqrt * h_eff.conj().unsqueeze(1) / (inner2.unsqueeze(1) + self.eps)
        x_signal = b_k / var_sqrt * (updates - mean_k)     # [K,d]

        # noise
        noise_power = (noise_std ** 2) * self.tx_power
        noise = torch.randn(d, device=device) * np.sqrt(noise_power / 2) + \
            1j * torch.randn(d, device=device) * np.sqrt(noise_power / 2)

        y = (h_eff.unsqueeze(1) * x_signal).sum(dim=0) + noise  # [d] complex
        g_bar = (K_vec * mean_k).sum().real                     # scalar
        sumK = K_vec.sum().real + self.eps
        w_hat = (y / eta_sqrt + g_bar) / sumK                   # complex

        # We need a real update vector (model params are real)
        agg_update = w_hat.real

        diagnostics = {
            "eta": eta.item(),
            "min_inner2": inner2.min().item(),
            "noise_power": noise_power,
        }
        return agg_update, diagnostics
