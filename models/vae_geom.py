from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from utils.utils import unfilling,filling
from models.networks.vae_module_geom import Encoder2D,Decoder2D,GraphModelGCN2
from models.networks.embeddings import Embedder

class MMD_Computer(): 
    def __init__(self):
        pass

    def rbf_kernel(self, x, y, scales=[0.01,0.08,0.5,1,5,10]):
        dist_sq = torch.cdist(x, y, p=2)**2  # (N, M)
        dim = x.shape[1]
        
        kernel_val = 0
        for scale in scales:
            kernel_val += torch.exp(-dist_sq / (dim * scale))
        
        return kernel_val/len(scales)

    def compute_mmd(self,x, y):
        x_kernel = self.rbf_kernel(x, x)
        y_kernel = self.rbf_kernel(y, y)
        xy_kernel = self.rbf_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel) 

    def gaussian_global_fitting(self,z):#N,D
        true_samples_f = torch.randn((z.shape),device=z.device)
        mmd_loss_f = self.compute_mmd(true_samples_f, z)
        return mmd_loss_f


class VAE_Geom(nn.Module):
    def __init__(
            self, 
            in_channels = 3, 
            mid_channels = 8, 
            out_channels = 3,
            conv_norm_group = 4,
            attn_nhead = 8,
            use_mmd = False
        ): 
        super().__init__()
        self.mid_channels = mid_channels
        self.use_mmd = use_mmd 
        self.mmd_computer = MMD_Computer()
        ############################################################################ 
        # surf + surf mask
        self.conv_in = nn.Conv2d(in_channels,16,kernel_size=3,stride=1,padding=1)
        self.surf_mask_embed = Embedder(2,4)
        self.encoder_surf = Encoder2D(
            in_channels=in_channels,
            out_channels=mid_channels,
            block_out_channels=(16+4, 32, 64),
            layers_per_block=2,
            norm_num_groups=conv_norm_group,
            act_fn='silu',
            in_conv=False,
            double_z=False
        )

        #center graphconv
        self.graph_conv = GraphModelGCN2(
            in_dim=mid_channels*4*4,
            layers = 2, 
            alpha = 0.1, 
            theta = 0.5, 
            shared_weights=False
        )

        #center enc attn
        encoder_attn_layer = nn.TransformerEncoderLayer(d_model=mid_channels*4*4, nhead=attn_nhead, norm_first=True,dim_feedforward=512, dropout=0.1)
        self.encoder_attn = nn.TransformerEncoder(encoder_attn_layer, 2)
        
        # to μ,σ
        self.to_gaussian_surf = nn.Sequential(
            nn.GroupNorm(num_channels=mid_channels ,num_groups=conv_norm_group, eps=1e-6),
            nn.Conv2d(mid_channels, 2*mid_channels, 3, padding=1)
        )

        #center dec attn
        decoder_attn_layer = nn.TransformerEncoderLayer(d_model=mid_channels*4*4, nhead=attn_nhead, norm_first=True,dim_feedforward=512, dropout=0.1)
        self.decoder_attn = nn.TransformerEncoder(decoder_attn_layer, 2, nn.LayerNorm(mid_channels*4*4))

        self.decoder_surf = Decoder2D(
            in_channels=mid_channels,
            out_channels=out_channels,
            block_out_channels=(16, 32, 64),
            layers_per_block=2,
            norm_num_groups=conv_norm_group,
            act_fn='silu',
            out_conv=False
        )

        #out 
        self.mask_out_head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1, padding=0),
            nn.Sigmoid()
        )
        self.pos_out_head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(), 
            nn.Conv2d(16, 3, 1, padding=0),
        )


    def encode(self, surf_points, surf_mask, surf_points_mask, ef_adj):
        B,LF = surf_mask.shape
        
        #surf encode
        surf_attr_unbatch = unfilling(surf_points, surf_mask) # L'F,3,16,16
        surf_attr_mask_unbatch = unfilling(surf_points_mask, surf_mask) # L'F,1,16,16
        surf_attr_unbatch = self.conv_in(surf_attr_unbatch)
        surf_attr_mask_unbatch = self.surf_mask_embed(surf_attr_mask_unbatch.squeeze(1).long()).permute(0,3,1,2)
        all_surf_attr = torch.cat([surf_attr_unbatch,surf_attr_mask_unbatch],dim = 1)
        enc_f_conv,level_states = self.encoder_surf(all_surf_attr)   # L'F,C,4,4   

        #center 
        enc_f_conv = filling(enc_f_conv,surf_mask).reshape(B,LF,-1) #B,LF,C*4*4
        enc_f_conv = self.graph_conv(enc_f_conv,surf_mask,ef_adj)
        enc_f_conv = self.encoder_attn(src=enc_f_conv.permute(1,0,2),src_key_padding_mask=~surf_mask).transpose(0,1)

        #to gaussian 
        enc_f_conv = unfilling(enc_f_conv,surf_mask).reshape(-1,8,4,4) # L'F,C,4,4
        params_f = self.to_gaussian_surf(enc_f_conv)

        return params_f,level_states

    def gaussian_sample(self, params_f, surf_mask, mode = False):
        B,LF = surf_mask.shape
        posterior_f = MY_DiagonalGaussianDistribution(params_f) 
        
        z_f = posterior_f.sample()  # L'F,D
        if mode:
            z_f = posterior_f.mode()  # L'F,D

        kl_loss_f = posterior_f.kl_f().mean()
   
        # active_dims_f = (posterior_f.mean.abs() > 0.1) | (posterior_f.std < 0.9)
        # active_dims = [active_dims_f.float().mean()]
        # print(active_dims)
        # print(torch.mean(posterior_f.mean),torch.mean(posterior_f.mean.abs()),torch.mean(posterior_f.std))

        z_f = filling(z_f,surf_mask).reshape(B,LF,-1)
        return z_f, kl_loss_f

    def decode(self, z_f, surf_mask, level_states = None):
        #center
        z_f = self.decoder_attn(src=z_f.permute(1,0,2),src_key_padding_mask=~surf_mask).transpose(0,1)
        z_f = unfilling(z_f, surf_mask).reshape(-1,self.mid_channels,4,4)

        # surf decode
        rec_f_latent = self.decoder_surf(z_f,level_states=level_states) # L'F,16,16,16  
        rec_f = self.pos_out_head(rec_f_latent)
        rec_f_mask = self.mask_out_head(rec_f_latent)
        rec_f_batch = filling(rec_f,surf_mask)
        rec_f_mask_batch = filling(rec_f_mask,surf_mask)

        return rec_f_batch, rec_f_mask_batch

    def forward(self, surf_points, surf_mask, surf_points_mask, ef_adj, train=False): 
        device = surf_points.device
        params_f ,level_states = self.encode(surf_points,surf_mask,surf_points_mask,ef_adj)  
        z_f, kl_loss_f = self.gaussian_sample(params_f, surf_mask)
        recon_surf,rec_surf_mask = self.decode(z_f,surf_mask,level_states)

        if train:
            points_loss_f = F.mse_loss(recon_surf*surf_mask[..., None, None, None], surf_points*surf_mask[..., None, None, None], reduction='none').mean(dim=[2,3,4]) #B,N
            points_loss_f = points_loss_f.sum(dim=1) / surf_mask.sum(dim=1) 

            surf_points_mask = surf_points_mask.float()
            points_mask_loss_f = F.binary_cross_entropy(rec_surf_mask*surf_mask[..., None, None, None], surf_points_mask*surf_mask[..., None, None, None], reduction='none').mean(dim=[2,3,4]) #B,N
            points_mask_loss_f = points_mask_loss_f.sum(dim=1) / surf_mask.sum(dim=1) 
            
            if self.use_mmd:
                mmd_loss_f = self.mmd_computer.gaussian_global_fitting(unfilling(z_f,surf_mask))
            else:
                mmd_loss_f = torch.tensor(0,device=device)
            return points_loss_f,points_mask_loss_f,kl_loss_f,mmd_loss_f
            
        else:
            return recon_surf,rec_surf_mask 


##################################################################### 

class MY_DiagonalGaussianDistribution(DiagonalGaussianDistribution):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        super().__init__(parameters, deterministic)

    def kl_f(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1,2,3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1,2,3],
                )
            
