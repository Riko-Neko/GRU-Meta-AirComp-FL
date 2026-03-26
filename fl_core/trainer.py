import torch
import torch.nn as nn
import torch.optim as optim

class GRUTrainer:
    """
    Trainer for CNN+GRU model on cascaded channel estimation task.
    Handles local training on a device's data.
    """
    def __init__(self, learning_rate=1e-3, epochs=1, batch_size=None, device=None,
                 optimizer_name="adam", momentum=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
        self.optimizer_name = str(optimizer_name).strip().lower()
        self.momentum = float(momentum)
        if self.optimizer_name not in {"adam", "sgd"}:
            raise ValueError("optimizer_name must be 'adam' or 'sgd'")
        if self.momentum < 0.0:
            raise ValueError("momentum must be nonnegative")
        self.criterion = nn.MSELoss()
        self.criterion_pl = nn.SmoothL1Loss()
        self.last_loss_stats = {}

    @staticmethod
    def _unpack_sample(sample):
        if len(sample) == 2:
            X, y = sample
            meta = None
        elif len(sample) == 3:
            X, y, meta = sample
        else:
            raise ValueError(f"Expected sample as (X, y) or (X, y, meta), got tuple length {len(sample)}")
        return X, y, meta

    def _build_aux_tensors(self, metas, batch_size):
        tau_loss_weight = torch.ones((batch_size,), dtype=torch.float32, device=self.device)
        has_weight = False
        pl_sel = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
        log_pl_sel = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        has_pl = False
        pl_loss_weight = torch.ones((batch_size,), dtype=torch.float32, device=self.device)
        pl_loss_scale = 0.0
        for idx, meta in enumerate(metas):
            if not meta:
                continue
            if "tau_loss_weight" in meta:
                tau_loss_weight[idx] = float(meta["tau_loss_weight"])
                has_weight = True
            if "pl_sel" in meta:
                pl_sel[idx, 0] = float(meta["pl_sel"])
                has_pl = True
            if "log_pl_sel" in meta:
                log_pl_sel[idx, 0] = float(meta["log_pl_sel"])
                has_pl = True
            if "pl_loss_weight" in meta:
                pl_loss_weight[idx] = float(meta["pl_loss_weight"])
            if "pl_loss_scale" in meta:
                pl_loss_scale = float(meta["pl_loss_scale"])
        aux = {}
        if has_weight:
            aux["tau_loss_weight"] = tau_loss_weight
        if has_pl:
            aux["pl_sel"] = pl_sel
            aux["log_pl_sel"] = log_pl_sel
            aux["pl_loss_weight"] = pl_loss_weight
            aux["pl_loss_scale"] = pl_loss_scale
        return aux or None

    def _build_optimizer(self, model):
        if self.optimizer_name == "adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate)
        return optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def _compute_loss_and_pack(self, outputs, targets, aux_tensors=None):
        """
        Handle both single-output models and the GRU dual-head predictor.
        Returns:
          loss: scalar tensor
          packed_outputs: tensor shaped like targets for downstream logging/prediction
        """
        if isinstance(outputs, tuple) and len(outputs) == 3:
            pred_t, pred_tau, pred_pl = outputs
            target_t, target_tau = torch.chunk(targets, 2, dim=-1)
            loss_t = self.criterion(pred_t, target_t)
            tau_sqerr = torch.mean((pred_tau - target_tau) ** 2, dim=-1)
            loss_tau = torch.mean(tau_sqerr)
            tau_loss_weight = None
            pl_target = None
            pl_loss_weight = None
            if aux_tensors is not None:
                tau_loss_weight = aux_tensors.get("tau_loss_weight")
                pl_target = aux_tensors.get("pl_sel")
                pl_loss_weight = aux_tensors.get("pl_loss_weight")
            if tau_loss_weight is None:
                weighted_loss_tau = loss_tau
                tau_weight_mean = 1.0
            else:
                weighted_loss_tau = torch.mean(tau_sqerr * tau_loss_weight)
                tau_weight_mean = float(torch.mean(tau_loss_weight).detach().item())
            if pl_target is None:
                loss_pl = torch.zeros((), dtype=pred_t.dtype, device=pred_t.device)
                pl_weight_mean = 0.0
            else:
                log_pred_pl = torch.log(torch.clamp(pred_pl, min=1e-12))
                log_target_pl = torch.log(torch.clamp(pl_target, min=1e-12))
                if pl_loss_weight is None:
                    loss_pl = self.criterion_pl(log_pred_pl, log_target_pl)
                    pl_weight_mean = 1.0
                else:
                    pl_err = torch.abs(log_pred_pl - log_target_pl)
                    smooth = torch.where(pl_err < 1.0, 0.5 * (pl_err ** 2), pl_err - 0.5)
                    loss_pl = torch.mean(smooth.squeeze(-1) * pl_loss_weight)
                    pl_weight_mean = float(torch.mean(pl_loss_weight).detach().item())
            pl_scale = 0.0
            if aux_tensors is not None:
                pl_scale = float(aux_tensors.get("pl_loss_scale", 0.0))
            loss = 0.5 * (loss_t + weighted_loss_tau) + (pred_t.new_tensor(pl_scale) * loss_pl)
            packed_outputs = torch.cat([pred_t, pred_tau], dim=-1)
            self.last_loss_stats = {
                "loss": float(loss.detach().item()),
                "loss_t": float(loss_t.detach().item()),
                "loss_tau": float(loss_tau.detach().item()),
                "loss_tau_weighted": float(weighted_loss_tau.detach().item()),
                "tau_loss_weight_mean": tau_weight_mean,
                "loss_pl": float(loss_pl.detach().item()),
                "pl_loss_weight_mean": pl_weight_mean,
            }
            return loss, packed_outputs
        if isinstance(outputs, tuple):
            pred_t, pred_tau = outputs
            target_t, target_tau = torch.chunk(targets, 2, dim=-1)
            loss_t = self.criterion(pred_t, target_t)
            tau_sqerr = torch.mean((pred_tau - target_tau) ** 2, dim=-1)
            loss_tau = torch.mean(tau_sqerr)
            tau_loss_weight = None
            if aux_tensors is not None:
                tau_loss_weight = aux_tensors.get("tau_loss_weight")
            if tau_loss_weight is None:
                weighted_loss_tau = loss_tau
                tau_weight_mean = 1.0
            else:
                weighted_loss_tau = torch.mean(tau_sqerr * tau_loss_weight)
                tau_weight_mean = float(torch.mean(tau_loss_weight).detach().item())
            loss = 0.5 * (loss_t + weighted_loss_tau)
            packed_outputs = torch.cat([pred_t, pred_tau], dim=-1)
            self.last_loss_stats = {
                "loss": float(loss.detach().item()),
                "loss_t": float(loss_t.detach().item()),
                "loss_tau": float(loss_tau.detach().item()),
                "loss_tau_weighted": float(weighted_loss_tau.detach().item()),
                "tau_loss_weight_mean": tau_weight_mean,
                "loss_pl": None,
                "pl_loss_weight_mean": None,
            }
            return loss, packed_outputs
        loss = self.criterion(outputs, targets)
        self.last_loss_stats = {
            "loss": float(loss.detach().item()),
            "loss_t": None,
            "loss_tau": None,
            "loss_tau_weighted": None,
            "tau_loss_weight_mean": None,
            "loss_pl": None,
            "pl_loss_weight_mean": None,
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
        optimizer = self._build_optimizer(model)
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
                unpacked = [self._unpack_sample(sample) for sample in batch_samples]
                X_batch = torch.stack([torch.tensor(X, dtype=torch.float32, device=self.device) for (X, _, _) in unpacked])
                y_batch = torch.stack([torch.tensor(y, dtype=torch.float32, device=self.device) for (_, y, _) in unpacked])
                aux_tensors = self._build_aux_tensors([meta for (_, _, meta) in unpacked], X_batch.shape[0])
                optimizer.zero_grad()
                # Forward pass
                outputs = model(X_batch)
                loss, _ = self._compute_loss_and_pack(outputs, y_batch, aux_tensors=aux_tensors)
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
            for sample in data:
                X, y, _ = self._unpack_sample(sample)
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
        optimizer = self._build_optimizer(model)
        X, y, meta = self._unpack_sample(sample)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, seq_len, 2, obs_dim)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)
        aux_tensors = self._build_aux_tensors([meta], 1)

        h_prev = None
        if hidden_state is not None:
            h_prev = hidden_state.detach().to(self.device)

        final_loss = None
        for _ in range(self.epochs):
            optimizer.zero_grad()
            outputs, _ = model(X_tensor, h0=h_prev, return_hidden=True)
            loss, _ = self._compute_loss_and_pack(outputs, y_tensor, aux_tensors=aux_tensors)
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
        optimizer = self._build_optimizer(model)

        h_init = None
        if hidden_state is not None:
            h_init = hidden_state.detach().to(self.device)
        h_prev = h_init

        final_loss = None
        last_pred = None
        for _ in range(self.epochs):
            for sample in samples:
                X, y, meta = self._unpack_sample(sample)
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
                y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)
                aux_tensors = self._build_aux_tensors([meta], 1)
                optimizer.zero_grad()
                outputs, h_next = model(X_tensor, h0=h_prev, return_hidden=True)
                loss, packed_outputs = self._compute_loss_and_pack(outputs, y_tensor, aux_tensors=aux_tensors)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
                # Keep hidden state continuity while truncating gradient history between segments.
                h_prev = h_next.detach()
                last_pred = packed_outputs.detach().cpu().squeeze(0).float()

        model.eval()
        with torch.no_grad():
            h_eval = None if h_init is None else h_init.detach().clone()
            for sample in samples:
                X, _, _ = self._unpack_sample(sample)
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
        optimizer = self._build_optimizer(model)
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
                unpacked = [self._unpack_sample(sample) for sample in batch_samples]
                X_batch = torch.stack([
                    torch.tensor(X, dtype=torch.float32, device=self.device) for (X, _, _) in unpacked
                ])
                y_batch = torch.stack([
                    torch.tensor(y, dtype=torch.float32, device=self.device) for (_, y, _) in unpacked
                ])
                aux_tensors = self._build_aux_tensors([meta for (_, _, meta) in unpacked], X_batch.shape[0])
                optimizer.zero_grad()
                h0_batch = None
                if h_init is not None:
                    h0_batch = h_init.repeat(1, X_batch.shape[0], 1)
                outputs, _ = model(X_batch, h0=h0_batch, return_hidden=True)
                loss, _ = self._compute_loss_and_pack(outputs, y_batch, aux_tensors=aux_tensors)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
        return model, final_loss

    def infer_stateful_sample(self, model, sample, hidden_state=None, return_aux=False):
        """
        Run one state update/inference step for the canonical current-round sample.
        Returns: (packed_output_cpu, hidden_next_cpu)
        """
        model.to(self.device)
        model.eval()
        X, _, _ = self._unpack_sample(sample)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
        h_prev = None
        if hidden_state is not None:
            h_prev = hidden_state.detach().to(self.device)
        with torch.no_grad():
            outputs, hidden_next = model(X_tensor, h0=h_prev, return_hidden=True)
            aux = None
            if isinstance(outputs, tuple) and len(outputs) == 3:
                packed_outputs = torch.cat([outputs[0], outputs[1]], dim=-1)
                aux = {
                    "pl_hat": outputs[2].detach().cpu().squeeze(0).float(),
                }
            elif isinstance(outputs, tuple):
                packed_outputs = torch.cat([outputs[0], outputs[1]], dim=-1)
            else:
                packed_outputs = outputs
        model.train()
        if return_aux:
            return packed_outputs.detach().cpu().squeeze(0).float(), hidden_next.detach().cpu(), aux
        return packed_outputs.detach().cpu().squeeze(0).float(), hidden_next.detach().cpu()
