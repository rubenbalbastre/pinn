import torch


def encode_u_type(u_type: str):

    u_type_map = {"wave": 0, "diffusion": 1}
    assert u_type in u_type_map.keys()
    u_type_onehot = torch.nn.functional.one_hot(torch.tensor(u_type_map[u_type]), num_classes=len(u_type_map.keys())).float()

    return u_type_onehot


def concat_encoder_input(u_xt, xt, u_type):
    
    u_type = u_type.reshape(1,-1).expand(u_xt.reshape(-1, 1).size(0), -1)
    inp = torch.cat([xt, u_xt.flatten().unsqueeze(1), u_type], dim=1)

    return inp