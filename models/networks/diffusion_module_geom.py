import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, Type
from timm.models.vision_transformer import Mlp
from collections.abc import Callable


class PatchEmbed1D(nn.Module):
    def __init__(
            self,
            input_size: Union[int, Tuple[int, int]] = 32,
            patch_size: int = 2,
            hidden_size: int = 384,
            bias: bool = True,
    ):
        super().__init__()
        assert input_size % patch_size==0
        assert hidden_size % input_size == 0
        self.patch_size = patch_size
        self.num_patches = input_size//patch_size
        self.map_channels = hidden_size//self.num_patches
        self.proj = nn.Conv1d(1, self.map_channels, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):  
        B, N, D = x.shape
        x = x.reshape(B*N,D).unsqueeze(1) #B*N,1,D
        x = self.proj(x)#B*N,C,D//patch_size
        x = x.permute(0,2,1).reshape(B,N,-1) #B,N,hidden_size
        return x
    
    
class FlimAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            edge_dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
    ):

        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # FiLM-style edge and node conditioning
        self.to_e_mul_add = nn.Sequential(
            nn.SiLU(),
            nn.Linear(edge_dim, 2*self.num_heads , bias=True),
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        if mask is not None:
            x_mask = mask.unsqueeze(-1)     # (b, n) -> (b, n, 1)
            e_mask1 = x_mask.unsqueeze(-1)  # (b, n, 1, 1)
            e_mask2 = x_mask.unsqueeze(1)   # (b, 1, n, 1)
        
        qkv = self.qkv(x) 
        if mask is not None: qkv = qkv * x_mask
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        #e flim
        if e is not None:
            e_mul, e_add = self.to_e_mul_add(e).chunk(2, dim=-1)  # (b, n, n, h*2)
            if mask is not None:
                e_mul = e_mul * e_mask1 * e_mask2
                e_add = e_add * e_mask1 * e_mask2
            e_mul = e_mul.permute(0,3,1,2)  # (B, h, N, N)
            e_add = e_add.permute(0,3,1,2)  # (B, h, N, N)
            attn = (1. + e_mul) * attn + e_add 

        if mask is not None:
            attn_mask = e_mask2.expand(-1, N, -1, self.num_heads).permute(0,3,1,2)  # (b, n, n, h)-> (b, h, n, n)
            attn = attn.masked_fill(~attn_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        if mask is not None:
            x = x * x_mask

        return x

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, edge_hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attn = FlimAttention(dim=hidden_dim,edge_dim=edge_hidden_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp_ff = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, e, c, mask=None):

        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), e, mask) * mask.unsqueeze(-1)
        x = x + gate_mlp * self.mlp_ff(modulate(self.norm2(x), shift_mlp, scale_mlp)) * mask.unsqueeze(-1)
        return x
    
    
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, patch_num):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_num, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
def modulate(x, shift, scale):
    if len(scale.shape) < len(x.shape):scale = scale.unsqueeze(1)
    if len(shift.shape) < len(x.shape):shift = shift.unsqueeze(1)
    return x * (1 + scale) + shift


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

