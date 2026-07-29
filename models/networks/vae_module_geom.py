from typing import *
import torch
import torch.nn as nn
import numpy as np
import math
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D
from diffusers.models.attention_processor import Attention
from diffusers.models.resnet import ResnetBlock2D
from torch_geometric.nn import GCN2Conv

from utils.utils import ef_adj_to_edge_index


class GraphModelGCN2(nn.Module): 
    def __init__(self, in_dim=128, layers = 3, alpha = 0.2, theta = 0.5, shared_weights=False):
        super(GraphModelGCN2, self).__init__()
        self.conv = nn.ModuleList([
            GCN2Conv(
                channels=in_dim, 
                alpha=alpha, 
                theta=theta, 
                layer=layer+1, 
                shared_weights=shared_weights
            )
            for layer in range(layers)
        ])
        self.act = nn.GELU() 
        self.norms = nn.ModuleList([nn.LayerNorm(in_dim) for _ in range(layers)])
 
    def forward(self, x, x_mask, ef_adj): #B,LF,D  B,LF  B,LE,2  
        B,LE,LF = ef_adj.shape[0],ef_adj.shape[1],x_mask.shape[1]
        layers = len(self.conv)
        edge_index = ef_adj_to_edge_index(ef_adj,const=True,max_faces=LF) #2,LE'
        x_flatten = x.reshape(B*LF,-1)
        x0 = x_flatten.clone() 
        x = x_flatten
        for i in range(layers):
            conv_out = self.conv[i](x,x0,edge_index)
            x = (x + conv_out)/math.sqrt(2)
            x = self.norms[i](x)
            x = self.act(x)

        #out
        x = x.reshape(B,LF,-1) * x_mask.unsqueeze(-1)
        return x


class AttnSkipDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = np.sqrt(2.0),
        add_downsample: bool = False,
    ):
        super().__init__()
        self.add_downsample = add_downsample
        self.attentions = nn.ModuleList([])
        self.resnets = nn.ModuleList([])

        for i in range(num_layers):
            
            in_channels = in_channels if i == 0 else out_channels
            
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=min(in_channels // 4, 32),
                    groups_out=min(out_channels // 4, 32),
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_in_shortcut=None          
                )
            )
            
            self.attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=min(out_channels // 4, 32),
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        if self.add_downsample:
            self.resnet_down = ResnetBlock2D(
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=min(out_channels // 4, 32),
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                use_in_shortcut=None,
                down=True,
                kernel="fir",
            )
        else:
            self.resnet_down = None


    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states += (hidden_states,)

        if self.add_downsample:
            hidden_states = self.resnet_down(hidden_states, temb)
            
        return hidden_states, output_states
    

class AttnSkipUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = np.sqrt(2.0),
        add_upsample: bool = False,
        use_skip: bool = True
    ):
        super().__init__()
        self.add_upsample = add_upsample
        self.use_skip = use_skip
        self.attentions = nn.ModuleList([])
        self.resnets = nn.ModuleList([])

        for i in range(num_layers):
            res_skip_channels = out_channels #in_channels if (i == num_layers - 1) else 
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            if not use_skip: res_skip_channels = 0
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=(resnet_in_channels + res_skip_channels) // 4,
                    groups_out=min(out_channels // 4, 32),
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_in_shortcut=None
                )
            )

        self.attentions.append(
            Attention(
                out_channels,
                heads=out_channels // attention_head_dim,
                dim_head=attention_head_dim,
                rescale_output_factor=output_scale_factor,
                eps=resnet_eps,
                norm_num_groups=min(out_channels // 4, 32),
                residual_connection=True,
                bias=True,
                upcast_softmax=True,
                _from_deprecated_attn_block=True,
            )
        )
        
        if self.add_upsample:
            self.resnet_up = ResnetBlock2D(
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=min(out_channels // 4, 32),
                groups_out=min(out_channels // 4, 32),
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                use_in_shortcut=None,
                up=True,
                kernel="fir",
            )
        else:
            self.resnet_up = None

        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states = None,
        temb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        for resnet in self.resnets:
            # pop res hidden states
            if self.use_skip:
                if isinstance(res_hidden_states,Tuple) or isinstance(res_hidden_states,list):
                    res_hidden_state = res_hidden_states[-1]
                    res_hidden_states = res_hidden_states[:-1]
                else:
                    res_hidden_state = res_hidden_states
                hidden_states = torch.cat([hidden_states, res_hidden_state], dim=1)

            hidden_states = resnet(hidden_states, temb)

        hidden_states = self.attentions[0](hidden_states)
        
        if self.resnet_up:
            hidden_states = self.resnet_up(hidden_states, temb)

        return hidden_states
    

class Encoder2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True,
        in_conv: bool = True,
        out_conv: bool = True,
        double_z = True
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.block_num = len(block_out_channels)
        self.in_conv = in_conv
        self.out_conv = out_conv

        self.conv_in = nn.Conv2d(in_channels,block_out_channels[0],kernel_size=3,stride=1,padding=1)

        #down
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i in range(self.block_num):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = AttnSkipDownBlock2D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                dropout=0,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                attention_head_dim=output_channel,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1] ,num_groups=norm_num_groups, eps=1e-6)#norm_num_groups
        self.conv_act = nn.SiLU()
        conv_out_channel = out_channels*2 if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channel, 3, padding=1)

        
    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        if self.in_conv:sample = self.conv_in(sample)  # B,c,h,w
        # down
        level_states = []
        
        for down_block in self.down_blocks:
            sample, output_states = down_block(sample)
            level_states.append(output_states)

        # middle
        sample = self.mid_block(sample)  

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        if self.out_conv: sample = self.conv_out(sample)

        return sample, level_states


class Decoder2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True,
        use_skip = False,
        in_conv=True,
        out_conv=True

    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.block_num = len(block_out_channels)
        self.use_skip = use_skip
        self.in_conv = in_conv
        self.out_conv = out_conv

        self.conv_in = nn.Conv2d(in_channels,block_out_channels[-1],kernel_size=3,stride=1,padding=1)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(self.block_num):  
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i ==  len(block_out_channels) - 1

            up_block = AttnSkipUpBlock2D(
                num_layers=self.layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                attention_head_dim=output_channel,
                use_skip=use_skip
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        level_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if self.in_conv: sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample)

        # up
        for up_block in self.up_blocks:
            if self.use_skip:
                sample = up_block(sample, res_hidden_states = level_states.pop())
            else:
                sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        if self.out_conv: sample = self.conv_out(sample)

        return sample
    

        