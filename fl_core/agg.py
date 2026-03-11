import copy
import torch

class MetaUpdater:
    """
    Base Meta-FL aggregator class.
    Provides interface for aggregating local model updates into the global model.
    """
    def __init__(self, meta_algorithm="Reptile", step_size=1.0, use_aircomp=False, aircomp_simulator=None):
        self.meta_algorithm = meta_algorithm
        self.step_size = step_size  # meta step size (e.g., beta for Reptile)
        self.use_aircomp = use_aircomp
        self.aircomp = aircomp_simulator
    
    def aggregate(
        self,
        global_model,
        local_models,
        H_BR=None,
        h_RUs=None,
        f=None,
        theta=None,
        *,
        backbone_only=False,
        prefix="backbone",
        client_weights=None,
    ):
        """
        Aggregate local model parameters into the global model.
        Optionally simulate AirComp communication error if use_aircomp is True.
        """
        # Default: FedAvg (if step_size = 1.0 and no special meta algorithm)
        # This method is meant to be overridden by specific algorithms if needed.
        # Implement a generic FedAvg as default.
        # Get state dicts of local models
        state_dicts = [lm.state_dict() for lm in local_models]
        K = len(state_dicts)
        if client_weights is None:
            weight_vec = torch.ones((K,), dtype=torch.float32)
        else:
            weight_vec = torch.as_tensor(client_weights, dtype=torch.float32).reshape(-1)
            if weight_vec.numel() != K:
                raise ValueError(f"client_weights size mismatch: expected {K}, got {weight_vec.numel()}")
            weight_vec = torch.clamp(weight_vec, min=0.0)
        weight_sum = float(torch.sum(weight_vec).item())
        if weight_sum <= 0.0:
            raise ValueError("client_weights must have positive sum")
        # Initialize aggregated state as zeros
        base_state = global_model.state_dict()
        agg_state = {key: torch.zeros_like(val) for key, val in base_state.items()}
        # Sum all local parameters (optionally backbone only)
        for idx, state in enumerate(state_dicts):
            coeff = float(weight_vec[idx].item())
            for key, val in state.items():
                if backbone_only and not key.startswith(prefix):
                    continue
                agg_state[key] += coeff * val.cpu()  # ensure on CPU for aggregation
        # Simulate AirComp noise if enabled
        if self.use_aircomp and self.aircomp is not None:
            agg_state = self.aircomp.aggregate(agg_state, K)
        else:
            # Average directly; weighted FedAvg if client_weights is provided.
            for key in agg_state:
                if backbone_only and not key.startswith(prefix):
                    continue
                agg_state[key] /= weight_sum
        # If using a meta-learning update (Reptile) with step_size < 1
        if self.meta_algorithm.lower() == "reptile" and self.step_size < 1.0:
            new_state = copy.deepcopy(base_state)
            for key in new_state:
                # Move global weights towards aggregated weights
                if backbone_only and not key.startswith(prefix):
                    new_state[key] = base_state[key]
                else:
                    new_state[key] = base_state[key] + self.step_size * (agg_state[key] - base_state[key])
            global_model.load_state_dict(new_state)
        else:
            # For FedAvg (or step_size = 1), just set global to aggregated
            if backbone_only:
                new_state = copy.deepcopy(base_state)
                for key in new_state:
                    if key.startswith(prefix):
                        new_state[key] = agg_state[key]
                global_model.load_state_dict(new_state)
            else:
                global_model.load_state_dict(agg_state)
        return global_model

    def apply_aggregated_delta(self, global_model, avg_delta_vector: torch.Tensor, *, backbone_only=False, prefix="backbone"):
        """Apply aggregated update vector; optionally restrict to backbone params."""
        state = global_model.state_dict()
        new_state = {}
        offset = 0
        for k, v in state.items():
            if backbone_only and not k.startswith(prefix):
                new_state[k] = v
                continue
            numel = v.numel()
            new_state[k] = (v.detach().reshape(-1) + avg_delta_vector[offset:offset + numel].to(v.device)).view_as(v)
            offset += numel
        global_model.load_state_dict(new_state)
        return global_model
