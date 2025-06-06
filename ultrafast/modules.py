import torch
from torch import nn
import torch.nn.functional as F
from torch import nn, optim

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    A single layer of multi-head self-attention.
    """
    def __init__(self, embed_dim, num_heads, head_dim, dropout=0.0, **kwargs):
        super().__init__()
        self.heads = num_heads
        self.inner_dim = num_heads * head_dim
        self.to_qkv = nn.Linear(embed_dim, self.inner_dim * 3, bias=False)
        self.multihead_attn = nn.MultiheadAttention(self.inner_dim, num_heads, dropout=dropout, batch_first=True, **kwargs)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        return self.multihead_attn(q, k , v)[0]

class Transformer(nn.Module):
    """
    Layers of self-attention after appending a CLS token to the input.

    The input to this module is expected to be the frozen embeddings of a protein foundation model,
    such as ProtBert or ESM. 

    `dim` is the dimension of the embeddings.
    `depth` is the number of transformer layers.
    `num_heads` is the number of heads in the multi-head attention.
    `head_dim` is the dimension of the heads.
    `mlp_dim` is the hidden dimension of the feed forward network. 
    """
    def __init__(self, dim, depth, num_heads, head_dim, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, num_heads, head_dim, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # import pdb; pdb.set_trace()
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class TargetEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout = 0., out_type="cls"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        self.transformer = Transformer(embedding_dim, num_layers, 8, embedding_dim // 8, hidden_dim, dropout = dropout)
        self.out_type = "cls" if out_type == "cls" else "mean"

    def forward(self, x):
        """
        Returns the embedding of the CLS token after passing through the transformer.
        """
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        return x[:,0] if self.out_type == "cls" else x[:,1:].mean(dim=1)

# from https://github.com/facebookresearch/deit/blob/main/patchconvnet_models.py
class Learned_Aggregation_Layer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale if qk_scale is not None else head_dim**-0.5

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.id = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, N, C = x.shape
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        q = self.q(cls_tokens).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = self.id(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        import pickle 
        file_path = 'biosnap_protein_after_agg.pkl'
        try:
            with open(file_path, "rb") as f:
                data_list = pickle.load(f)  # Đọc danh sách cũ
        except (FileNotFoundError, EOFError):
            data_list = []  # Nếu file chưa tồn tại hoặc rỗng, tạo danh sách mới

        # Thêm mảng mới vào danh sách
        data_list.append(x_cls.detach().cpu().numpy())

        # Ghi lại danh sách vào file
        with open(file_path, "wb") as f:
            pickle.dump(data_list, f)
        print(len(data_list))
        return x_cls.squeeze()

class AverageNonZeroVectors(nn.Module):
    def __init__(self, eps=1e-8):
        super(AverageNonZeroVectors, self).__init__()
        self.eps = eps

    def forward(self, input_batch):
        non_zero_mask = (input_batch.sum(dim=-1) != 0).float()
        num_non_zero = non_zero_mask.sum(dim=1)
        batch_sum = torch.sum(input_batch * non_zero_mask.unsqueeze(-1), dim=1)
        batch_avg = batch_sum / (num_non_zero.unsqueeze(-1) + self.eps)

        if torch.isnan(batch_avg).any() or torch.isinf(batch_avg).any():
            print("NaN or Inf found in averaging operation")
            batch_avg = torch.nan_to_num(batch_avg, nan=0.0, posinf=1.0, neginf=-1.0)

        return batch_avg

class LargeProteinProjector(nn.Module):
    def __init__(self, target_dim, latent_dim, activation, num_heads=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.proj = Learned_Aggregation_Layer(target_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.res = nn.Sequential(
                nn.LayerNorm(target_dim),
                activation(),
                nn.Linear(target_dim, latent_dim),
                nn.Dropout(proj_drop),
                nn.LayerNorm(latent_dim),
                activation(),
                nn.Linear(latent_dim, latent_dim),
                nn.Dropout(proj_drop),
                nn.LayerNorm(latent_dim),
                activation(),
                nn.Linear(latent_dim, latent_dim),
                nn.Dropout(proj_drop),
                nn.LayerNorm(latent_dim),
            )
        self.non_linearity = activation()

    def forward(self, x):
        proj = self.proj(x)
        proj = self.res(proj)
        return self.non_linearity(proj)

class LargeDrugProjector(nn.Module):
    def __init__(self, drug_dim, latent_dim, activation, dropout):
        super().__init__()
        self.linear1 = nn.Linear(drug_dim, 1260)
        self.drop1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(1260)
        self.linear2 = nn.Linear(1260, latent_dim)
        self.drop2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.linear3 = nn.Linear(latent_dim, latent_dim)
        self.drop3 = nn.Dropout(dropout)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.linear4 = nn.Linear(latent_dim, latent_dim)

        self.non_linearity = activation()

        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.linear3.weight)
        nn.init.xavier_normal_(self.linear4.weight)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = self.non_linearity(x)

        return x

# ====================== Self supervised ============================

class VICReg(nn.Module):
    def forward(self, x, y):
        
        batch_size = x.size(0)
        feature_dim = x.size(1)
        repr_loss = F.mse_loss(x, y)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            feature_dim
        ) + off_diagonal(cov_y).pow_(2).sum().div(feature_dim)
        self.sim_coeff = 0.5
        self.std_coeff = 0.3
        self.cov_coeff = 0.3
        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss



def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()