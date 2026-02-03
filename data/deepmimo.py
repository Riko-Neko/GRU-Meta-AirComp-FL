import os
import numpy as np


def load_data(file_path, num_users=None):
    """
    Load DeepMIMO dataset from a .npy or .npz file.
    Returns channel data (H_BR, h_RUs, h_BUs) for the simulation.
    - H_BR: BS-RIS channel
    - h_RUs: RIS-UE channel
    - h_BUs: optional BS-UE direct channel (may be None if not present)
    If file not found or format not recognized, raises an error.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"DeepMIMO data file not found: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    H_BR = None
    h_RUs = None
    h_BUs = None
    try:
        # If data is an .npz archive
        if isinstance(data, np.lib.npyio.NpzFile):
            if "H_BR" in data.files:
                H_BR = data["H_BR"]
            if "h_RU" in data.files:
                h_RUs = data["h_RU"]
            elif "h_RUs" in data.files:
                h_RUs = data["h_RUs"]
            elif "H_RU" in data.files:
                h_RUs = data["H_RU"]

            if "h_BU" in data.files:
                h_BUs = data["h_BU"]
            elif "h_BUs" in data.files:
                h_BUs = data["h_BUs"]
            elif "H_BU" in data.files:
                h_BUs = data["H_BU"]
        else:
            # If data is a dict (pickled in .npy)
            if isinstance(data.item(), dict):
                data_dict = data.item()
                H_BR = data_dict.get("H_BR", None)
                h_RUs = data_dict.get("h_RU", None) or data_dict.get("h_RUs", None) or data_dict.get("H_RU", None)
                h_BUs = data_dict.get("h_BU", None) or data_dict.get("h_BUs", None) or data_dict.get("H_BU", None)
            else:
                # If data is an array, interpret as cascaded channels for multiple users?
                arr = data
                if arr.ndim == 2 and num_users is not None:
                    # If shape matches (num_users, N) treat each row as h_RU (no separate H_BR given)
                    h_RUs = arr[:num_users] if arr.shape[0] >= num_users else arr
                    # Without H_BR, assume H_BR is ones (no separate BS-RIS effect)
                    H_BR = np.ones((h_RUs.shape[1], 1), dtype=np.complex64)
                else:
                    raise ValueError("Unrecognized data format in DeepMIMO file.")
    except Exception as e:
        raise e
    if H_BR is None or h_RUs is None:
        raise ValueError("DeepMIMO data file missing required entries (H_BR, h_RU).")
    # If specific number of users requested, slice
    if num_users is not None and h_RUs.shape[0] >= num_users:
        h_RUs = h_RUs[:num_users]
    if h_BUs is not None and num_users is not None and h_BUs.shape[0] >= num_users:
        h_BUs = h_BUs[:num_users]
    return H_BR, h_RUs, h_BUs
