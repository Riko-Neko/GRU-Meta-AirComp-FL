import torch
import torch.nn as nn


class CSICNNBaseline(nn.Module):
    """
    Literature-style federated CNN baseline.
    - Single-step pilot input (memoryless, no recurrent state).
    - Full model sharing under FedAvg-style aggregation.
    """

    def __init__(self, observation_dim, output_dim, conv_filters=16, conv_kernel=3, hidden_size=64):
        super(CSICNNBaseline, self).__init__()
        self.conv = nn.Conv1d(in_channels=2, out_channels=conv_filters, kernel_size=conv_kernel)
        feature_len = observation_dim - conv_kernel + 1
        self.fc1 = nn.Linear(conv_filters * feature_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (batch, 2, obs_dim) or (batch, seq_len, 2, obs_dim)
        If seq input is provided, only the latest step is used.
        """
        if x.dim() == 4:
            x = x[:, -1, :, :]
        feat = self.relu(self.conv(x))
        feat = feat.view(feat.size(0), -1)
        feat = self.relu(self.fc1(feat))
        return self.fc2(feat)
