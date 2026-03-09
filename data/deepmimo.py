import numpy as np


def load_data(file_path, num_users=None):
    """
    Load DeepMIMO dataset from a .npy or .npz file.
    Returns channel data (H_BR, h_RUs, h_BUs) for the simulation.
    - H_BR: BS-RIS channel
    - h_RUs: RIS-UE channel
    - h_BUs: optional BS-UE direct channel (may be None if not present)
    If file not found, np.load will raise an error directly.
    """
    data = np.load(file_path, allow_pickle=True)
    H_BR = None
    h_RUs = None
    h_BUs = None
    # If data is an .npz archive
    if isinstance(data, np.lib.npyio.NpzFile):
        H_BR = data["H_BR"]
        h_RUs = data["h_RU"]
        h_BUs = data["h_BU"] if "h_BU" in data.files else None
    else:
        # If data is a dict (pickled in .npy)
        if isinstance(data.item(), dict):
            data_dict = data.item()
            H_BR = data_dict["H_BR"]
            h_RUs = data_dict["h_RU"]
            h_BUs = data_dict["h_BU"] if "h_BU" in data_dict else None
        else:
            # If data is an array, interpret each row as h_RU and use unit H_BR.
            arr = data
            h_RUs = arr[:num_users] if (num_users is not None and arr.shape[0] >= num_users) else arr
            H_BR = np.ones((h_RUs.shape[1], 1), dtype=np.complex64)
    # If specific number of users requested, slice
    if num_users is not None and h_RUs.shape[0] >= num_users:
        h_RUs = h_RUs[:num_users]
    if h_BUs is not None and num_users is not None and h_BUs.shape[0] >= num_users:
        h_BUs = h_BUs[:num_users]
    return H_BR, h_RUs, h_BUs
