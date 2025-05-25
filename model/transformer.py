import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):   #dim_head 64->32
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)## 对tensor张量分块 x :1 197 1024   qkv 最后是一个元祖，tuple，长度是3，每个元素形状：1 197 1024
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim = 64, depth = 1, heads = 16, mlp_dim = 128, dim_head = 8, dropout = 0.1, emb_dropout = 0.1): #dim_head 64->8
        super().__init__()
        channels, image_height, image_width = image_size   # 5,280,448
        patch_height, patch_width = pair(patch_size)       # 8*8   4*4

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)     # 35*56   70*112
        patch_dim = 16 * patch_height * patch_width    # 16*8*8   16*4*4      patch_dim:64->16     

        self.conv1 = nn.Conv2d(5, 16, 1)

        self.to_patch_embedding = nn.Sequential(
            # (b,64,64,80) -> (b,320,1024)    16*20=320  4*4*64=1024
            # (b,16,280,448) -> (b,1960,1024)    56*35=1960  8*8*16=1024
            # (b,16,280,448) -> (b,7480,256)    70*112=7840  4*4*16=256
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),  # h70, w112
            nn.Linear(patch_dim, dim),    # (b,256,64)
        )

        self.to_img = nn.Sequential(
            # b c (h p1) (w p2) -> (b,64,64,80)      16*20=320  4*4*64=1024
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', \
                      p1 = patch_height, p2 = patch_width, h = image_height // patch_height, w = image_width // patch_width),
            nn.Conv2d(4, 5, 1),      # (b,64,64,80) -> (b,256,64,80)
        )
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.conv1(img)                     # img 1 5 280 448 -> 1 16 280 448
        x = self.to_patch_embedding(x)          # 1 320 1024   # 1 1960 1024   # 1 7840 256
        b, n, _ = x.shape                       # 1 320  # 1 1960   # 1 7840

        x += self.pos_embedding[:, :(n + 1)]    # (1,7840,64)
        x = self.dropout(x)                     # (1,7840,64)

        x = self.transformer(x)                 # (1,320,1024)

        x = self.to_img(x)

        return x                                # (1 256 64 80)


if __name__ == '__main__':


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    v = ViT(image_size = (5,280,448), patch_size = 4)

    v = v.to(device)

    img = torch.randn(2, 5, 280, 448)

    img = img.to(device)

    preds = v(img)         # (1, 256, 64, 80)

    print(preds.shape)
