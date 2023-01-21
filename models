import torch
from torch import einsum

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads)
        
        # Attention weights
        weights = einsum("bhid,bhjd->bhij", q, k) * self.scale
        weights = torch.softmax(weights, dim=-1)
        
        # Apply attention
        output = einsum("bhij,bhjd->bhid", weights, v)
        output = output.reshape(B, N, C)
        
        return output
   
   
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, qk_scale=None):
        super().__init__()
        self.attn = CrossAttention(dim, num_heads, qk_scale)
        self.ff = nn.Linear(dim, dim)
        
    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        x = F.relu(x)
        return x

class MultiScaleBlock(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.conv_layers = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=3, dilation=2 ** i, padding=2 ** i) for i in range(num_layers)])
        
    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, dim, num_heads, num_layers, qk_scale=None):
        super().__init__()
        self.embed = PatchEmbed()
        self.msb = MultiScaleBlock(dim, num_layers)
        self.attn = nn.ModuleList([CrossAttentionBlock(dim, num_heads, qk_scale) for _ in range(num_layers)])
        
    def forward(self, x):
        x = self.embed(x)
        x = self.msb(x)
        for attn in self.attn:
            x = attn(x)
        return x
            x = F.relu(x)
        return x
