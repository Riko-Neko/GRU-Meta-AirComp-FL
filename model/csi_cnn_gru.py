import torch
import torch.nn as nn

class CSICNNGRU(nn.Module):
    """
    CNN+GRU model for cascaded channel estimation.
    Input: sequence of pilot observations (seq_len x observation_dim (with real+imag channels)).
    Output: estimated cascaded channel vector (complex, output as real and imag parts).
    """
    def __init__(self, observation_dim, output_dim, conv_filters=8, conv_kernel=3, hidden_size=32):
        super(CSICNNGRU, self).__init__()
        # Convolutional feature extractor for each time step's observation
        # Input has 2 channels (real & imag), length = observation_dim
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=conv_filters, kernel_size=conv_kernel)
        # We'll flatten conv output and feed to GRU
        # Compute feature length after conv: 
        # If no padding: out_length = observation_dim - conv_kernel + 1
        self.feature_length = observation_dim - conv_kernel + 1
        self.gru_input_size = conv_filters * self.feature_length
        # GRU for temporal sequence
        self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=hidden_size, batch_first=True)
        # Final linear layer to output 2*N (for complex vector real+imag)
        self.fc = nn.Linear(hidden_size, output_dim)
        # Non-linearities
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, seq_len, 2, observation_dim)
        batch_size, seq_len, _, obs_dim = x.shape
        # Reshape to combine batch and seq dims for conv
        # new shape for conv input: (batch*seq_len, 2, observation_dim)
        x_reshaped = x.view(batch_size * seq_len, 2, obs_dim)
        # Apply convolution
        conv_out = self.conv1(x_reshaped)  # shape: (batch*seq_len, conv_filters, feature_length)
        conv_out = self.relu(conv_out)
        # Flatten conv output (except batch dimension)
        conv_out = conv_out.view(batch_size * seq_len, -1)
        # Reshape back to (batch, seq_len, features)
        conv_out_seq = conv_out.view(batch_size, seq_len, -1)
        # GRU processing
        gru_out, _ = self.gru(conv_out_seq)  # gru_out: (batch, seq_len, hidden_size)
        # Take last time step output
        final_out = gru_out[:, -1, :]  # shape: (batch, hidden_size)
        # Linear to output predicted channel (2*N)
        pred = self.fc(final_out)  # shape: (batch, output_dim)
        return pred