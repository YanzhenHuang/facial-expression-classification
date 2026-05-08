import math
import torch
import numpy as np
from torch import nn, Tensor
from faceformer import NUM_POINTS, DIM_MODEL, NUM_HEADS, HEAD_DIM, NUM_CLASSES

NUM_LAYERS = 2  # 保持你的层数
DROP_RATE = 0.1


class FacePointEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_id_emb = nn.Embedding(NUM_POINTS, DIM_MODEL)

        self.dim_xy = DIM_MODEL * 3 // 8
        self.dim_z = DIM_MODEL - 2 * self.dim_xy

        i_xy = torch.arange(0, self.dim_xy).float()
        freq_bands_xy = torch.exp(i_xy * (-math.log(10000.0) / self.dim_xy))
        self.register_buffer('freq_bands_xy', freq_bands_xy)

        i_z = torch.arange(0, self.dim_z).float()
        freq_bands_z = torch.exp(i_z * (-math.log(10000.0) / self.dim_z))
        self.register_buffer('freq_bands_z', freq_bands_z)

        self.coord_proj = nn.Linear(3, DIM_MODEL)

    def forward(self, aligned_coords: Tensor):
        B, N, _ = aligned_coords.shape
        x, y, z = aligned_coords[:, :, 0], aligned_coords[:, :, 1], aligned_coords[:, :, 2]

        # sinusoidal位置编码
        pe_x = torch.sin(x.unsqueeze(-1) * self.freq_bands_xy)
        pe_y = torch.cos(y.unsqueeze(-1) * self.freq_bands_xy)
        pe_z = torch.sin(z.unsqueeze(-1) * self.freq_bands_z)
        coord_pe = torch.cat([pe_x, pe_y, pe_z], dim=-1) # (B, N, DIM_MODEL)

        # (B, N, Model_DIM) -> (B, N * Model_DIM) --proj--> (B, Model_DIM)

        # 点位语义编码
        point_ids = torch.arange(NUM_POINTS, device=aligned_coords.device).unsqueeze(0) # (478)

        # 融合：语义ID + 坐标特征 + 你的正弦PE
        token = self.point_id_emb(point_ids) + self.coord_proj(aligned_coords) + coord_pe
        return token


class Relative3DAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(DIM_MODEL, DIM_MODEL)
        self.k_proj = nn.Linear(DIM_MODEL, DIM_MODEL)
        self.v_proj = nn.Linear(DIM_MODEL, DIM_MODEL)
        self.out_proj = nn.Linear(DIM_MODEL, DIM_MODEL)

    def forward(self, x: Tensor, coords: Tensor):
        B, N, _ = x.shape
        # 多头QKV
        q = self.q_proj(x).view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, N, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        # Attention
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)

        # 计算两两 landmarks 之间的 3D 欧氏距离
        dist_matrix = torch.cdist(coords, coords, p=2)  # (B, N, N)

        # 对每个点，找到距离最近的 K 个点（包括自己）
        K = 5
        _, nearest_indices = torch.topk(dist_matrix, k=K, dim=-1, largest=False)  # (B, N, K)

        # 创建 mask：初始化为 False
        band_mask = torch.zeros(B, N, N, device=x.device, dtype=torch.bool)

        # 将最近邻的位置设为 True
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).unsqueeze(2).expand(-1, N, K)
        row_indices = torch.arange(N, device=x.device).unsqueeze(0).unsqueeze(2).expand(B, -1, K)
        band_mask[batch_indices, row_indices, nearest_indices] = True

        # 应用 mask：非最近邻的位置填充 -inf
        attn_score_ = attn_score.masked_fill(~band_mask.unsqueeze(1), float('-inf'))

        # 把mask过的和没mask的都作softmax，然后做平均，即相当于手动强化了每个点和附近点的权重
        attn_w = torch.softmax(attn_score, dim=-1)
        attn_w_ = torch.softmax(attn_score_, dim=-1)

        out = torch.matmul((attn_w + attn_w_) / 2, v).transpose(1, 2).reshape(B, N, DIM_MODEL)
        return self.out_proj(out)


class FaceTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Relative3DAttention()
        self.norm1 = nn.LayerNorm(DIM_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(DIM_MODEL, DIM_MODEL * 2),
            nn.GELU(),
            nn.Dropout(DROP_RATE),
            nn.Linear(DIM_MODEL * 2, DIM_MODEL)
        )
        self.norm2 = nn.LayerNorm(DIM_MODEL)
        self.dropout = nn.Dropout(DROP_RATE)

    def forward(self, x: Tensor, coords: Tensor):
        x = self.norm1(x + self.dropout(self.attn(x, coords)))
        x = self.norm2(x + self.ffn(x))
        return x, coords


class FaceFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = FacePointEmbedding()
        self.layers = nn.ModuleList([FaceTransformerLayer() for _ in range(NUM_LAYERS)])
        self.norm = nn.LayerNorm(DIM_MODEL)
        self.classifier = nn.Linear(DIM_MODEL, NUM_CLASSES)

    def forward(self, coords: Tensor):
        # coords: (B, 478, 3)
        x = self.embedding(coords) # (B, 478, D_MODEL)
        for layer in self.layers:
            x, coords = layer(x, coords)
        x = self.norm(x)
        global_feat = torch.mean(x, dim=1)
        out = self.classifier(global_feat)
        return out