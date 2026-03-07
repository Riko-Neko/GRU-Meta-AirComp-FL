import torch


def state_dict_to_vector(model):
    """Flatten model parameters into a 1D torch tensor."""
    params = []
    for _, v in model.state_dict().items():
        params.append(v.detach().reshape(-1))
    return torch.cat(params)


def vector_to_state_dict(model, vec):
    """Load a 1D tensor back into model state_dict order/shape."""
    state_dict = model.state_dict()
    offset = 0
    new_state = {}
    for k, v in state_dict.items():
        numel = v.numel()
        new_state[k] = vec[offset:offset + numel].view_as(v)
        offset += numel
    model.load_state_dict(new_state)
    return model


def model_delta_to_vector(local_model, global_model):
    """Return flattened (local - global) parameters as 1D tensor."""
    local_state = local_model.state_dict()
    global_state = global_model.state_dict()
    deltas = []
    for k in local_state.keys():
        deltas.append((local_state[k] - global_state[k]).reshape(-1))
    return torch.cat(deltas)


def _filter_state_dict(state_dict, *, include_prefix=None):
    if include_prefix is None:
        return state_dict
    return {k: v for k, v in state_dict.items() if k.startswith(include_prefix)}


def state_dict_to_vector_backbone(model, prefix="backbone"):
    """Flatten backbone parameters (prefix match) into 1D tensor."""
    state = _filter_state_dict(model.state_dict(), include_prefix=prefix)
    return torch.cat([v.detach().reshape(-1) for v in state.values()])


def vector_to_state_dict_backbone(model, vec, prefix="backbone"):
    """Load backbone vector back into model (only prefix-matched params)."""
    state_dict = model.state_dict()
    offset = 0
    new_state = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            new_state[k] = v
            continue
        numel = v.numel()
        new_state[k] = vec[offset:offset + numel].view_as(v)
        offset += numel
    model.load_state_dict(new_state)
    return model


def model_delta_to_vector_backbone(local_model, global_model, prefix="backbone"):
    """Flatten (local - global) for backbone-only parameters."""
    local_state = _filter_state_dict(local_model.state_dict(), include_prefix=prefix)
    global_state = _filter_state_dict(global_model.state_dict(), include_prefix=prefix)
    deltas = []
    for k in local_state.keys():
        deltas.append((local_state[k] - global_state[k]).reshape(-1))
    return torch.cat(deltas)
