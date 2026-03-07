import copy
import torch
from fl_core.agg import MetaUpdater

class ReptileAggregator(MetaUpdater):
    """
    Reptile algorithm aggregator (MetaUpdater subclass).
    Moves global model parameters slightly toward the average of local model parameters.
    """
    def __init__(self, step_size=0.1, use_aircomp=False, aircomp_simulator=None):
        super().__init__(meta_algorithm="Reptile", step_size=step_size, use_aircomp=use_aircomp, aircomp_simulator=aircomp_simulator)
    
    def aggregate(self, global_model, local_models, H_BR=None, h_RUs=None, f=None, theta=None, *, backbone_only=False, prefix="backbone"):
        # Get global model state (before update)
        base_state = global_model.state_dict()
        # Sum local model parameters
        K = len(local_models)
        sum_state = {key: torch.zeros_like(val) for key, val in base_state.items()}
        for model in local_models:
            for key, val in model.state_dict().items():
                if backbone_only and not key.startswith(prefix):
                    continue
                sum_state[key] += val.cpu()
        # Simulate AirComp if enabled
        if self.use_aircomp and self.aircomp is not None:
            # Add noise to the sum of parameters
            sum_state = self.aircomp.aggregate_sum(sum_state)
        # Compute average of local parameters
        avg_state = {}
        for key in sum_state:
            if backbone_only and not key.startswith(prefix):
                continue
            avg_state[key] = sum_state[key] / K
        # Reptile update: global = global + beta * (avg_local - global)
        new_state = copy.deepcopy(base_state)
        beta = self.step_size
        for key in base_state:
            if backbone_only and not key.startswith(prefix):
                new_state[key] = base_state[key]
            else:
                new_state[key] = base_state[key] + beta * (avg_state[key] - base_state[key])
        global_model.load_state_dict(new_state)
        return global_model

    def apply_aggregated_delta(self, global_model, avg_delta_vector: torch.Tensor, *, backbone_only=False, prefix="backbone"):
        """Apply aggregated update vector; optionally restrict to backbone params."""
        beta = self.step_size
        state = global_model.state_dict()
        new_state = {}
        offset = 0
        for k, v in state.items():
            if backbone_only and not k.startswith(prefix):
                new_state[k] = v
                continue
            numel = v.numel()
            new_state[k] = (v.detach().reshape(-1) + beta * avg_delta_vector[offset:offset + numel].to(v.device)).view_as(v)
            offset += numel
        global_model.load_state_dict(new_state)
        return global_model
