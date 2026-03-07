import torch
import torch.nn as nn


class CSICNNAblation(nn.Module):
    """
    CNN ablation model for cascaded channel estimation.
    Keeps the same per-step Conv1d encoder as CNN+GRU, but replaces GRU with
    simple temporal pooling over the time window.
    """

    def __init__(self, observation_dim, output_dim, conv_filters=8, conv_kernel=3, hidden_size=32, pool_mode="mean"):
        super(CSICNNAblation, self).__init__()
        self.backbone_conv = nn.Conv1d(in_channels=2, out_channels=conv_filters, kernel_size=conv_kernel)
        self.feature_length = observation_dim - conv_kernel + 1
        self.backbone_proj = nn.Linear(conv_filters * self.feature_length, hidden_size)
        self.relu = nn.ReLU()
        self.pool_mode = str(pool_mode).lower()

        # Keep a per-user head consistent with the current personalization logic.
        self.head = nn.Linear(hidden_size, output_dim)

    def forward_backbone(self, x):
        """
        x: (batch, seq_len, 2, obs_dim)
        Return pooled sequence representation: (batch, hidden_size)
        """
        batch_size, seq_len, _, obs_dim = x.shape
        x_reshaped = x.view(batch_size * seq_len, 2, obs_dim)
        conv_out = self.relu(self.backbone_conv(x_reshaped))
        conv_out = conv_out.view(batch_size, seq_len, -1)
        proj_out = self.relu(self.backbone_proj(conv_out))

        if self.pool_mode == "last":
            return proj_out[:, -1, :]
        return proj_out.mean(dim=1)

    def forward_head(self, feat):
        return self.head(feat)

    def forward(self, x):
        feat = self.forward_backbone(x)
        return self.forward_head(feat)
