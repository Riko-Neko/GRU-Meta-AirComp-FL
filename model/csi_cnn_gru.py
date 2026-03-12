import torch
import torch.nn as nn


class CSICNNGRU(nn.Module):
    """
    CNN+GRU model for cascaded channel estimation.
    Split into a shared backbone (conv + GRU) and a per-user head (single FC).
    Input: sequence of pilot observations (seq_len x observation_dim with real+imag channels).
    Output: two estimated cascaded channel vectors (t and t+1, real+imag stacked).
    """

    def __init__(self, observation_dim, output_dim, conv_filters=8, conv_kernel=3, hidden_size=32):
        super(CSICNNGRU, self).__init__()
        # Backbone: convolutional feature extractor + temporal GRU
        self.backbone_conv = nn.Conv1d(in_channels=2, out_channels=conv_filters, kernel_size=conv_kernel)
        self.feature_length = observation_dim - conv_kernel + 1
        self.gru_input_size = conv_filters * self.feature_length
        self.backbone_gru = nn.GRU(input_size=self.gru_input_size, hidden_size=hidden_size, batch_first=True)
        self.relu = nn.ReLU()

        # Head: per-user dual-horizon predictor kept local (not OTA-aggregated).
        self.head = nn.Linear(hidden_size, 2 * output_dim)

    def forward_backbone(self, x, h0=None, return_hidden=False):
        """Return shared representation from conv+GRU backbone."""
        batch_size, seq_len, _, obs_dim = x.shape
        x_reshaped = x.view(batch_size * seq_len, 2, obs_dim)
        conv_out = self.relu(self.backbone_conv(x_reshaped))
        conv_out = conv_out.view(batch_size * seq_len, -1)
        conv_out_seq = conv_out.view(batch_size, seq_len, -1)
        if h0 is None:
            gru_out, h_n = self.backbone_gru(conv_out_seq)
        else:
            gru_out, h_n = self.backbone_gru(conv_out_seq, h0)
        final_out = gru_out[:, -1, :]  # (batch, hidden_size)
        if return_hidden:
            return final_out, h_n
        return final_out

    def forward_head(self, feat):
        """Apply per-user head to backbone features."""
        out = self.head(feat)
        csi_t_hat, csi_t1_hat = torch.chunk(out, 2, dim=-1)
        return csi_t_hat, csi_t1_hat

    def forward(self, x, h0=None, return_hidden=False):
        if return_hidden:
            feat, h_n = self.forward_backbone(x, h0=h0, return_hidden=True)
            return self.forward_head(feat), h_n
        feat = self.forward_backbone(x, h0=h0, return_hidden=False)
        return self.forward_head(feat)

    def forward_step(self, x_step, h0=None, return_hidden=False):
        """
        Stateful single-step forward.
        x_step: (batch, 2, obs_dim) or (batch, 1, 2, obs_dim)
        """
        if x_step.dim() == 3:
            x_step = x_step.unsqueeze(1)  # (batch, 1, 2, obs_dim)
        return self.forward(x_step, h0=h0, return_hidden=return_hidden)
