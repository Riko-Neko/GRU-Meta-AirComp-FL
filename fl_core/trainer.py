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
        self.last_loss_stats = {}

    def _compute_loss_and_pack(self, outputs, targets):
        """
        Handle both single-output models and the GRU dual-head predictor.
        Returns:
          loss: scalar tensor
          packed_outputs: tensor shaped like targets for downstream logging/prediction
        """
        if isinstance(outputs, tuple):
            pred_t, pred_delta = outputs
            target_t, target_delta = torch.chunk(targets, 2, dim=-1)
            loss_t = self.criterion(pred_t, target_t)
            loss_delta = self.criterion(pred_delta, target_delta)
            loss = 0.5 * (loss_t + loss_delta)
            packed_outputs = torch.cat([pred_t, pred_delta], dim=-1)
            self.last_loss_stats = {
                "loss": float(loss.detach().item()),
                "loss_t": float(loss_t.detach().item()),
                "loss_delta": float(loss_delta.detach().item()),
            }
            return loss, packed_outputs
        loss = self.criterion(outputs, targets)
        self.last_loss_stats = {
            "loss": float(loss.detach().item()),
            "loss_t": None,
            "loss_delta": None,
        }
        return loss, outputs
    
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
                loss, _ = self._compute_loss_and_pack(outputs, y_batch)
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
                loss, _ = self._compute_loss_and_pack(pred, y_tensor)
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
            loss, _ = self._compute_loss_and_pack(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        model.eval()
        with torch.no_grad():
            _, hidden_next = model(X_tensor, h0=h_prev, return_hidden=True)
        model.train()
        return model, final_loss, hidden_next.detach().cpu()

    def train_stateful_sequence(self, model, samples, hidden_state=None):
        """
        Stateful sequential local update on continuous segments.
        samples: list of (X, y), each X has shape (1, 2, obs_dim) or (seq_len, 2, obs_dim).
        hidden_state: previous local hidden state (num_layers, 1, hidden_size), CPU tensor or None.
        Returns: (model, final_loss, hidden_next_cpu_detached, last_pred_cpu_tensor)
        """
        if not samples:
            raise ValueError("samples must be non-empty for train_stateful_sequence")

        model.to(self.device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        h_init = None
        if hidden_state is not None:
            h_init = hidden_state.detach().to(self.device)
        h_prev = h_init

        final_loss = None
        last_pred = None
        for _ in range(self.epochs):
            for X, y in samples:
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
                y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)
                optimizer.zero_grad()
                outputs, h_next = model(X_tensor, h0=h_prev, return_hidden=True)
                loss, packed_outputs = self._compute_loss_and_pack(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
                # Keep hidden state continuity while truncating gradient history between segments.
                h_prev = h_next.detach()
                last_pred = packed_outputs.detach().cpu().squeeze(0).float()

        model.eval()
        with torch.no_grad():
            h_eval = None if h_init is None else h_init.detach().clone()
            for X, _ in samples:
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
                outputs, h_eval = model(X_tensor, h0=h_eval, return_hidden=True)
                if isinstance(outputs, tuple):
                    packed_outputs = torch.cat([outputs[0], outputs[1]], dim=-1)
                else:
                    packed_outputs = outputs
                last_pred = packed_outputs.detach().cpu().squeeze(0).float()
        model.train()

        if h_eval is None:
            hidden_next_cpu = None
        else:
            hidden_next_cpu = h_eval.detach().cpu()
        return model, final_loss, hidden_next_cpu, last_pred

    def train_stateful_independent(self, model, samples, hidden_state=None):
        """
        Train on multiple same-round samples with a shared previous hidden state.
        Physical time does not advance across samples; each sample is conditioned on
        the same hidden_state from the previous round.
        Returns: (model, final_loss)
        """
        if not samples:
            raise ValueError("samples must be non-empty for train_stateful_independent")

        model.to(self.device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        final_loss = None
        batch_size = self.batch_size or len(samples)

        h_init = None
        if hidden_state is not None:
            h_init = hidden_state.detach().to(self.device)

        for _ in range(self.epochs):
            if len(samples) > 1:
                import random
                random.shuffle(samples)
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i + batch_size]
                X_batch = torch.stack([
                    torch.tensor(X, dtype=torch.float32, device=self.device) for (X, _) in batch_samples
                ])
                y_batch = torch.stack([
                    torch.tensor(y, dtype=torch.float32, device=self.device) for (_, y) in batch_samples
                ])
                optimizer.zero_grad()
                h0_batch = None
                if h_init is not None:
                    h0_batch = h_init.repeat(1, X_batch.shape[0], 1)
                outputs, _ = model(X_batch, h0=h0_batch, return_hidden=True)
                loss, _ = self._compute_loss_and_pack(outputs, y_batch)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
        return model, final_loss

    def infer_stateful_sample(self, model, sample, hidden_state=None):
        """
        Run one state update/inference step for the canonical current-round sample.
        Returns: (packed_output_cpu, hidden_next_cpu)
        """
        model.to(self.device)
        model.eval()
        X, _ = sample
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
        h_prev = None
        if hidden_state is not None:
            h_prev = hidden_state.detach().to(self.device)
        with torch.no_grad():
            outputs, hidden_next = model(X_tensor, h0=h_prev, return_hidden=True)
            if isinstance(outputs, tuple):
                packed_outputs = torch.cat([outputs[0], outputs[1]], dim=-1)
            else:
                packed_outputs = outputs
        model.train()
        return packed_outputs.detach().cpu().squeeze(0).float(), hidden_next.detach().cpu()
