from typing import *
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.init as init
from torch.optim import Optimizer
from omegaconf import OmegaConf
from diffusers.training_utils import EMAModel


def unfilling(params,masks):
    return params[masks]

def filling(params,masks):
    filling_params=torch.zeros((masks.shape[0],masks.shape[1],*(params.shape[1:])),device=params.device)
    filling_params[masks] = params
    return filling_params


def ef_adj_to_edge_index(ef_adj, const=True, max_faces=30):

    B, LE = ef_adj.shape[0], ef_adj.shape[1]
    LE = torch.tensor(LE).to(ef_adj.device)
    # offset
    if const:
        node_offsets = torch.arange(B, dtype=torch.long, device=ef_adj.device) * max_faces
    else:
        max_node_per_graph = ef_adj.amax(dim=(1,2)) + 1
        node_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=ef_adj.device), 
                                  max_node_per_graph.cumsum(0)[:-1]])

    # reshape
    ef_adj = ef_adj.view(B * LE, 2)

    # remove padding
    mask = (ef_adj >= 0).all(dim=1).to(ef_adj.device)
    ef_adj_flatten = ef_adj[mask]   # (num_valid_edges, 2)

    # add offset
    batch_ids = torch.arange(B, device=ef_adj.device).repeat_interleave(LE)[mask]
    ef_adj_flatten = ef_adj_flatten + node_offsets[batch_ids].unsqueeze(1)

    # bidirection
    u = ef_adj_flatten[:,0]
    v = ef_adj_flatten[:,1]
    edge_index = torch.cat([torch.stack([u,v], dim=0),
                            torch.stack([v,u], dim=0)], dim=1)  # (2, num_edges*2)

    return edge_index

def lower_to_full_symmetric(x_lower, N,offset = -1): #(B, N*(N-1)//2) -> (B, N, N)
    B, K = x_lower.shape
    x_full = torch.zeros(B, N, N,device=x_lower.device,dtype=x_lower.dtype)
    idx = torch.tril_indices(N, N, offset=offset)
    i, j = idx[0], idx[1]
    x_full[:, i, j] = x_lower
    mask = i != j
    x_full[:, j[mask], i[mask]] = x_lower[:, mask]

    return x_full


def permute_adj_list(adj_list):  #maybe useless
    permuted = []
    perms = []
    for adj in adj_list:
        n = adj.shape[0]
        perm = torch.randperm(n, device=adj.device)
        permuted.append(adj[perm][:, perm])
        perms.append(perm)
    return permuted

def pad_adj_matrices(adj_list,max_n = 30): #list ->  (B, N, N)
    B = len(adj_list)
    device = adj_list[0].device
    dtype = adj_list[0].dtype

    adj_matrix_batch = torch.zeros((B, max_n, max_n), device=device, dtype=dtype)
    f_mask_batch = torch.zeros((B, max_n), device=device, dtype=dtype)
    for i, a in enumerate(adj_list):
        n = a.shape[0]
        adj_matrix_batch[i, :n, :n] = a
        f_mask_batch[i,:n] = 1

    return adj_matrix_batch.bool(),f_mask_batch.bool()


def from_v_predict_x0(scheduler, x_t, v, t):
    device = x_t.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    t = t.to(device)
    v = v.to(device)
    alpha_cumprod_t = alphas_cumprod[t].view(-1, 1, 1).float() 
    sqrt_alpha = torch.sqrt(alpha_cumprod_t)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod_t)
    x0 = sqrt_alpha * x_t - sqrt_one_minus_alpha * v

    return x0


def rotate_points(point_cloud: torch.Tensor, angle_degrees: float, axis: str) -> torch.Tensor:
    device = point_cloud.device
    dtype = point_cloud.dtype

    # angle
    angle = angle_degrees * torch.pi / 180.0
    c, s = torch.cos(angle), torch.sin(angle)

    # rotation matrix
    if axis == 'x':
        R = torch.tensor([[1, 0, 0],
                          [0, c, -s],
                          [0, s,  c]], device=device, dtype=dtype)
    elif axis == 'y':
        R = torch.tensor([[ c, 0, s],
                          [0, 1, 0],
                          [-s, 0, c]], device=device, dtype=dtype)
    elif axis == 'z':
        R = torch.tensor([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]], device=device, dtype=dtype)
    else:
        raise ValueError

    # 1. center
    center = point_cloud.mean(dim=0, keepdim=True)
    pc = point_cloud - center

    # 2. rotate
    pc = pc @ R.T

    # 3. back-translate
    pc = pc + center

    # 4. normalize
    max_abs = pc.abs().max()
    pc = pc / max_abs

    return pc

def rotate_normals(normals: torch.Tensor, angle_degrees: float, axis: str) -> torch.Tensor:
    device = normals.device
    dtype = normals.dtype

    angle = angle_degrees * torch.pi / 180.0
    c, s = torch.cos(angle), torch.sin(angle)

    # rotation matrix
    if axis == 'x':
        R = torch.tensor([[1, 0, 0],
                          [0, c, -s],
                          [0, s,  c]], device=device, dtype=dtype)
    elif axis == 'y':
        R = torch.tensor([[ c, 0, s],
                          [0, 1, 0],
                          [-s, 0, c]], device=device, dtype=dtype)
    elif axis == 'z':
        R = torch.tensor([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]], device=device, dtype=dtype)
    else:
        raise ValueError

    n = normals @ R.T
    n = torch.nn.functional.normalize(n, dim=-1)

    return n


def load_checkpoints(
    model: Module,
    ckpt_path: str,
    ema_states: Optional[EMAModel]=None,
    optimizer: Optional[Optimizer]=None,
    device=torch.device("cpu"),
    strict: bool = True
) -> int:
    """Load checkpoint from the given experiment directory and return the epoch of this checkpoint."""

    if not os.path.exists(ckpt_path):  # checkpoint file not found
        print(f"Checkpoint file {ckpt_path} not found, starting from scratch\n")
        assert False

    print(f"Load checkpoint from {ckpt_path}\n")
    checkpoint = torch.load(ckpt_path, map_location=device,weights_only=False)

    if strict == False:
        new_state_dict = model.state_dict()
        state_dict = checkpoint["model"]
        for k in list(state_dict.keys()):
            if k not in new_state_dict or state_dict[k].shape != new_state_dict[k].shape:
                print(f"Skip loading {k}")
                del state_dict[k]
        model.load_state_dict(checkpoint["model"],strict=False)
    else:
        model.load_state_dict(checkpoint["model"])

    if ema_states is not None:
        ema_states.load_state_dict(checkpoint["ema_states"])
        ema_states.copy_to(model.parameters())
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return 


def save_checkpoints(model: Module, ckpt_dir: str, filename: str, optimizer: Optimizer=None, ema_states: Optional[EMAModel]=None, parallel = True) -> None:
    """Save checkpoint to the given experiment directory."""
    if parallel:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_dict = {"model": model_state_dict}
    if ema_states is not None:
        save_dict["ema_states"] = ema_states.state_dict()
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()

    save_path = os.path.join(ckpt_dir, filename)
    torch.save(save_dict, save_path)


def initialize_weights(model, method='xavier', mode='fan_in', nonlinearity='leaky_relu', verbose=False):

    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            continue
            
        # linear conv 
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if method == 'kaiming':
                init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
            elif method == 'xavier':
                init.xavier_normal_(module.weight)
            elif method == 'orthogonal':
                init.orthogonal_(module.weight)
            elif method == 'normal':
                init.normal_(module.weight, mean=0, std=0.02)
            elif method == 'zeros':
                init.zeros_(module.weight)
            else:
                raise ValueError(f"Unknown initialization method: {method}")
            
            # bias
            if module.bias is not None:
                init.zeros_(module.bias)
                        
        # Embedding
        elif isinstance(module, nn.Embedding):
            if method == 'normal':
                init.normal_(module.weight, mean=0, std=0.02)
            elif method == 'xavier':
                init.xavier_normal_(module.weight)
                
        if verbose:
            print(f"Initialized layer: {name} ({module.__class__.__name__})")

def seed_everything(seed: int = 42):
    random.seed(seed)          
    np.random.seed(seed)     
    torch.manual_seed(seed)    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False     
    os.environ['PYTHONHASHSEED'] = str(seed)   


def data_to_device(data, device, ignore_types=[str,int]):
    for k in data:
        if type(data[k][0]) in ignore_types: #skip 
            continue
        if isinstance(data[k], tuple) or isinstance(data[k], list) :
            data[k] = [b.to(device) for b in data[k]]
        elif isinstance(data[k], dict):
            for b in data[k]:
                data[k][b] = data[k][b].to(device)
        else:
            data[k] = data[k].to(device)
    return data


def load_config(yaml_path=None):
    cfg = OmegaConf.load(yaml_path)
    return cfg


class simple_logger():
    def __init__(self,exp_dir):
        self.exp_dir = exp_dir
        self.f = open(self.exp_dir, "w")
    def log(self,epoch,value):
        value_str = ""
        for i,loss in enumerate(value):
            value_str += "[{}]:{:.6f}, ".format(i,loss.item())
        fmt = "[{}] [epoch {:4d}] |loss: "
        msg = fmt.format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch) + value_str
        if self.f.isatty():  # if the file stream is interactive
            print(msg + "\b"*len(msg), end="", flush=True, file=self.f)
        else:
            print(msg, flush=True, file=self.f)
    
    
def compute_snr_weights(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    snr = alphas_cumprod / (1 - alphas_cumprod + 1e-6)
    weights = snr / (snr + 1.0)  
    weights = 2*(0.5 + 1 * weights)
    return weights

