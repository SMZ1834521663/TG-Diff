import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.embeddings import Embedder
from models.networks.diffusion_module_geom import PatchEmbed1D,TimestepEmbedder,DiTBlock,FinalLayer


class Diffusion_Geom(nn.Module): 
    def __init__(
        self,
        input_size=128,
        patch_size=1,
        hidden_dim=[512,512],
        depth=[6,2],
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth1,self.depth2 = depth

        self.s_embedder = PatchEmbed1D(input_size, patch_size, hidden_dim[0], bias=True)
        self.x_embedder = PatchEmbed1D(input_size, patch_size, hidden_dim[1], bias=True)
        self.t_embedder = TimestepEmbedder(hidden_dim[0])

        edge_hidden_dim = hidden_dim[0]//4
        self.adj_embed = nn.Sequential(
            Embedder(2, edge_hidden_dim), 
        ) 
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim[0] if i < depth[0] else hidden_dim[1],
                edge_hidden_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
            ) for i in range(depth[0]+depth[1])
        ])
        self.s_projector = nn.Sequential(
            nn.Linear(hidden_dim[0], hidden_dim[1])
        )  

        self.final_layer = FinalLayer(hidden_dim[1], patch_size, self.x_embedder.num_patches)
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, z_f_diffused, f_mask, adj_matrix, timesteps, target_v=None, train = True):
        # param
        B,LF = f_mask.shape
        device = z_f_diffused.device

        #t
        t_embeds = self.t_embedder(timesteps)     # (B, D)
        adj_matrix = ((adj_matrix>0) | torch.eye(LF, device=device,dtype=bool).unsqueeze(0).expand(B, -1, -1))* (f_mask.unsqueeze(1) & f_mask.unsqueeze(2))
        adj_embeds = self.adj_embed(adj_matrix.long())   
        c = nn.functional.silu(t_embeds)
        
        #s
        s_embeds = self.s_embedder(z_f_diffused)*(f_mask.unsqueeze(-1)) 
        for i in range(self.depth1):
            s_embeds = self.blocks[i](s_embeds, adj_embeds, c, mask = f_mask)

        #x
        t_embeds = t_embeds.unsqueeze(1).repeat(1, s_embeds.shape[1], 1)
        s_embeds = nn.functional.silu(t_embeds + s_embeds)
        s_embeds = self.s_projector(s_embeds)

        x_embeds = self.x_embedder(z_f_diffused)*(f_mask.unsqueeze(-1))  # (B, N, D)
        
        for i in range(self.depth1, self.depth1+self.depth2):
            x_embeds = self.blocks[i](x_embeds, adj_embeds, s_embeds, mask = f_mask)
        z_f_pred = self.final_layer(x_embeds, s_embeds)* f_mask.unsqueeze(-1) 

        if train:
            #loss
            loss_surf = F.mse_loss(z_f_pred*f_mask.unsqueeze(-1), target_v*f_mask.unsqueeze(-1), reduction='none').mean(-1) #B,N 
            loss_surf = loss_surf.sum(dim=1) / f_mask.sum(dim=1) 

            return loss_surf
        
        else:
            return z_f_pred