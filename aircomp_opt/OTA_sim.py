import torch
import numpy as np

class AirCompSimulator:
    """
    Simulator for over-the-air aggregation.
    Simulates the summation of local updates over wireless channels with noise.
    """
    def __init__(self, noise_std=0.0):
        self.noise_std = noise_std
    
    def aggregate(self, param_sum_dict, num_devices):
        """
        Given a dictionary of summed parameters (sum of all devices' parameters),
        add noise and divide by number of devices to simulate AirComp aggregated average.
        Returns a dictionary of aggregated (noisy) average parameters.
        """
        agg_state = {}
        for key, summed_val in param_sum_dict.items():
            # Add Gaussian noise to summed parameters
            if self.noise_std > 0:
                noise = torch.normal(mean=0.0, std=self.noise_std, size=summed_val.shape)
                summed_val = summed_val + noise
            # Divide by number of devices to get average
            agg_state[key] = summed_val / num_devices
        return agg_state
    
    def aggregate_sum(self, param_sum_dict):
        """
        Add noise to the summed parameters without dividing by number of devices.
        Useful if we want to handle averaging outside (e.g., for Reptile).
        """
        noisy_sum = {}
        for key, summed_val in param_sum_dict.items():
            if self.noise_std > 0:
                noise = torch.normal(mean=0.0, std=self.noise_std, size=summed_val.shape)
                noisy_sum[key] = summed_val + noise
            else:
                noisy_sum[key] = summed_val
        return noisy_sum