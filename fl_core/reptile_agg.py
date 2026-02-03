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
    
    def aggregate(self, global_model, local_models, H_BR=None, h_RUs=None, f=None, theta=None):
        # Get global model state (before update)
        base_state = global_model.state_dict()
        # Sum local model parameters
        K = len(local_models)
        sum_state = {key: torch.zeros_like(val) for key, val in base_state.items()}
        for model in local_models:
            for key, val in model.state_dict().items():
                sum_state[key] += val.cpu()
        # Simulate AirComp if enabled
        if self.use_aircomp and self.aircomp is not None:
            # Add noise to the sum of parameters
            sum_state = self.aircomp.aggregate_sum(sum_state)
        # Compute average of local parameters
        avg_state = {key: sum_state[key] / K for key in sum_state}
        # Reptile update: global = global + beta * (avg_local - global)
        new_state = copy.deepcopy(base_state)
        beta = self.step_size
        for key in base_state:
            new_state[key] = base_state[key] + beta * (avg_state[key] - base_state[key])
        global_model.load_state_dict(new_state)
        return global_model