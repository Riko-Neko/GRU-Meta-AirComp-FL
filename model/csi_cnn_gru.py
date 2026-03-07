import torch
import torch.nn as nn


class CSICNNGRU(nn.Module):
    """
    CNN+GRU model for cascaded channel estimation.
    Split into a shared backbone (conv + GRU) and a per-user head (single FC).
    Input: sequence of pilot observations (seq_len x observation_dim with real+imag channels).
    Output: estimated cascaded channel vector (real+imag stacked).
    """

    def __init__(self, observation_dim, output_dim, conv_filters=8, conv_kernel=3, hidden_size=32):
        super(CSICNNGRU, self).__init__()
        # Backbone: convolutional feature extractor + temporal GRU
        self.backbone_conv = nn.Conv1d(in_channels=2, out_channels=conv_filters, kernel_size=conv_kernel)
        self.feature_length = observation_dim - conv_kernel + 1
        self.gru_input_size = conv_filters * self.feature_length
        self.backbone_gru = nn.GRU(input_size=self.gru_input_size, hidden_size=hidden_size, batch_first=True)
        self.relu = nn.ReLU()

        # Head: single linear layer kept per-user (not OTA-aggregated)
        self.head = nn.Linear(hidden_size, output_dim)

    def forward_backbone(self, x):
        """Return shared representation from conv+GRU backbone."""
        batch_size, seq_len, _, obs_dim = x.shape
        x_reshaped = x.view(batch_size * seq_len, 2, obs_dim)
        conv_out = self.relu(self.backbone_conv(x_reshaped))
        conv_out = conv_out.view(batch_size * seq_len, -1)
        conv_out_seq = conv_out.view(batch_size, seq_len, -1)
        gru_out, _ = self.backbone_gru(conv_out_seq)
        final_out = gru_out[:, -1, :]  # (batch, hidden_size)
        return final_out

    def forward_head(self, feat):
        """Apply per-user head to backbone features."""
        return self.head(feat)

    def forward(self, x):
        feat = self.forward_backbone(x)
        return self.forward_head(feat)
