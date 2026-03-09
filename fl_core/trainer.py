import torch
import torch.nn as nn
import torch.optim as optim

class GRUTrainer:
    """
    Trainer for CNN+GRU model on cascaded channel estimation task.
    Handles local training on a device's data.
    """
    def __init__(self, learning_rate=1e-3, epochs=1, batch_size=None, device=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
        self.criterion = nn.MSELoss()
    
    def train(self, model, data):
        """
        Train the model on the given local data.
        data: list of (X, y) tuples, where X is input tensor (seq_len x 2 x obs_dim),
              y is target tensor (output_dim).
        Returns the trained model (in-place) and final loss value.
        """
        model.to(self.device)
        model.train()
        # Create optimizer for this model's parameters
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        final_loss = None
        # If batch_size not set, use all data as one batch
        batch_size = self.batch_size or len(data)
        # Training loop
        for epoch in range(self.epochs):
            # Optionally shuffle data each epoch
            if len(data) > 1:
                import random
                random.shuffle(data)
            # Batch training
            for i in range(0, len(data), batch_size):
                batch_samples = data[i:i+batch_size]
                # Stack batch samples into tensors
                X_batch = torch.stack([torch.tensor(X, dtype=torch.float32, device=self.device) for (X, y) in batch_samples])
                y_batch = torch.stack([torch.tensor(y, dtype=torch.float32, device=self.device) for (X, y) in batch_samples])
                optimizer.zero_grad()
                # Forward pass
                outputs = model(X_batch)
                loss = self.criterion(outputs, y_batch)
                # Backward and optimize
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
        return model, final_loss
    
    def evaluate(self, model, data):
        """
        Evaluate the model on given data (list of (X, y) tuples), returns MSE.
        """
        model.to(self.device)
        model.eval()
        total_loss = 0.0
        count = 0
        criterion = nn.MSELoss(reduction='sum')
        with torch.no_grad():
            for X, y in data:
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
                y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)
                pred = model(X_tensor)
                loss = criterion(pred, y_tensor)
                total_loss += loss.item()
                count += 1
        mse = total_loss / count if count > 0 else 0.0
        return mse

    def train_stateful_step(self, model, sample, hidden_state=None):
        """
        Stateful single-step local update.
        sample: (X, y), where X has shape (1, 2, obs_dim) or (seq_len, 2, obs_dim).
        hidden_state: previous local hidden state (num_layers, 1, hidden_size), CPU tensor or None.
        Returns: (model, final_loss, hidden_next_cpu_detached)
        """
        model.to(self.device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        X, y = sample
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, seq_len, 2, obs_dim)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

        h_prev = None
        if hidden_state is not None:
            h_prev = hidden_state.detach().to(self.device)

        final_loss = None
        for _ in range(self.epochs):
            optimizer.zero_grad()
            outputs, _ = model(X_tensor, h0=h_prev, return_hidden=True)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        model.eval()
        with torch.no_grad():
            _, hidden_next = model(X_tensor, h0=h_prev, return_hidden=True)
        model.train()
        return model, final_loss, hidden_next.detach().cpu()
