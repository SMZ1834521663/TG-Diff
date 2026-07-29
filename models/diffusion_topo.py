import torch
import torch.nn as nn
import numpy as np

from models.networks.embeddings import Embedder
from models.networks.diffusion_module_topo import TimestepEmbedder,DiTBlock,FinalLayer,RMSNorm
from models.networks.embeddings import sincos_embedding
from utils.utils import lower_to_full_symmetric


class Diffusion_Topo(nn.Module): 
    def __init__(
        self,
        hidden_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        max_faces=30,
        edge_emb_dim=8,
        edge_num_classes=2,
    ):
        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.max_faces = max_faces
        self.edge_emb_dim = edge_emb_dim

        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.num_embed = nn.Sequential(         
            Embedder(max_faces+1, hidden_dim), 
        ) 
        self.adj_embed = nn.Sequential(         
            Embedder(edge_num_classes+2, self.edge_emb_dim), 
        ) 

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
            ) for _ in range(depth)
        ])

        self.in_mlp = nn.Sequential(
            nn.Linear(max_faces*edge_emb_dim, hidden_dim), 
            RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), 
        ) 

        self.final_layer_e = FinalLayer(hidden_dim, 1, edge_emb_dim*max_faces)
        self.final_layer_ee = nn.Linear(edge_emb_dim,edge_num_classes+1)
        self.initialize_weights()
        self.pos = torch.tensor(get_2d_sincos_pos_embed(edge_emb_dim,max_faces)).reshape(max_faces,max_faces,edge_emb_dim)


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_e.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_e.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_e.linear.weight, 0)
        nn.init.constant_(self.final_layer_e.linear.bias, 0)

    def forward(
            self, 
            x_t_class,    #B,N*N+1 
            timesteps,
            mask,
            log_x_t = None,
            log_x_start=None,
            d3pm_scheduler=None,
            train=False
        ):

        # param
        B,LF = mask.shape
        device = x_t_class.device

        #t
        t_embeds = self.t_embedder(timesteps)     # (B, D)
        
        #e
        x_t_class = lower_to_full_symmetric(x_t_class,LF,offset=-1)
        adj_embeds = self.adj_embed(x_t_class.long()) #B,N,N,8
        adj_embeds = adj_embeds + self.pos.unsqueeze(0).to(device).float()
        adj_embeds = adj_embeds.reshape(B,LF,-1) #B,N,N*8
        adj_embeds = self.in_mlp(adj_embeds) #B,N,512
        
        #mask
        num_mask = torch.sum(mask,dim=1)
        num_embeds = self.num_embed(num_mask)
        
        #center process
        adj_embeds = adj_embeds + sincos_embedding(torch.arange(adj_embeds.shape[1], device=adj_embeds.device),adj_embeds.shape[-1], adj_embeds.shape[1]).unsqueeze(0)
        t_embeds = t_embeds+num_embeds
        for i in range(self.depth):
            adj_embeds = self.blocks[i](adj_embeds, t_embeds)

        #out
        e_class_pred = self.final_layer_e(adj_embeds,t_embeds) #B,N,N*8
        e_class_pred = e_class_pred.reshape(B,LF,LF,self.edge_emb_dim)
        e_class_pred = (e_class_pred + e_class_pred.transpose(1,2)) / 2
        e_class_pred = self.final_layer_ee(e_class_pred).contiguous()   #B,N,N,3
        idx = torch.tril_indices(LF, LF, offset=-1)
        e_class_pred_tril = e_class_pred[:, idx[0], idx[1], :]  #B,N*(N+1),C
    
        
        if train:
            #loss discrete
            log_x0_recon = d3pm_scheduler.log_pred_from_denoise_out(e_class_pred_tril) # p_theta(x0|xt)
            loss_edge = d3pm_scheduler.compute_kl_loss(log_x_start, log_x0_recon, log_x_t, timesteps)
            loss_edge = loss_edge.mean(dim=1)
            loss_aux = d3pm_scheduler.compute_aux_loss(log_x_start, log_x0_recon, timesteps)
            loss_aux = loss_aux.mean(dim=1)

            loss_edge = loss_edge + 0.02*loss_aux

            return 100*loss_edge
        
        else:
            return e_class_pred_tril 
        
        

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



