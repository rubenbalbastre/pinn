import torch


def encode_u_type(u_type: str):

    u_type_map = {"wave": 0, "diffusion": 1, "heat": 2}
    assert u_type in u_type_map.keys()
    u_type_onehot = torch.nn.functional.one_hot(torch.tensor(u_type_map[u_type]), num_classes=3).float()

    return u_type_onehot