from copy import deepcopy
from typing import cast

import torch
from torch import nn, Tensor
from torch.nn import ModuleList

TEN_THOUSAND = torch.tensor(10000.0)


class GraphEmbedding(nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()

        if dim_model % 8 != 0:
            raise ValueError(f"Dimension model must be divisible by 8, but got {dim_model}.")

        self.dim_model = dim_model

        # Determines the distribution of PEs on a token.
        # The first 3/8 and last 3/8 are for x and y, and the middle 1/4 is for z.
        self.dim_xy = self.dim_model * 3 // 8
        self.dim_z = self.dim_model - 2 * self.dim_xy

        i_xy = torch.arange(0, self.dim_xy).float()
        freq_bands_xy = torch.exp(i_xy * (-torch.log(TEN_THOUSAND) / self.dim_xy))
        self.register_buffer('freq_bands_xy', freq_bands_xy)

        i_z = torch.arange(0, self.dim_z).float()
        freq_bands_z = torch.exp(i_z * (-torch.log(TEN_THOUSAND) / self.dim_z))
        self.register_buffer('freq_bands_z', freq_bands_z)

        self.coord_proj = nn.Linear(3, dim_model)

    def forward(self, coords: Tensor):
        """
        Input vertex coordinates, output the embedding-based graph
        representation using these vertexes.
        :param coords: Vertex coordinates in the shape of (B, N, 3).
        :returns: A graph representation of shape (B, N, dim_model).
        """

        x, y, z = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]

        # sinusoidal PE: (B, N, 3) --calc--> (B, N, dim_model)
        pe_x = torch.sin(x.unsqueeze(-1) * self.freq_bands_xy)
        pe_y = torch.cos(y.unsqueeze(-1) * self.freq_bands_xy)
        pe_z = torch.sin(z.unsqueeze(-1) * self.freq_bands_z)
        coord_pe = torch.cat([pe_x, pe_z, pe_y], dim=-1)

        # coord projection: (B, N, 3) --project--> (B, N, dim_model)
        coord_proj = self.coord_proj(coords)

        # Graph Representation: (B, N, dim_model)
        # Each vertex is represented by a `dim_model` dimensional vector.
        graph_rep = coord_proj + coord_pe

        return graph_rep


class JEAttention(nn.Module):
    """
    Jumping Embedding Attention
    """

    def __init__(self, dim_model: int, num_heads: int, nearest_k: int):
        super().__init__()
        # Model Parameters
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.K = nearest_k

        # Q, K, and V projections
        self.q_proj = nn.Linear(self.dim_model, self.dim_model)
        self.k_proj = nn.Linear(self.dim_model, self.dim_model)
        self.v_proj = nn.Linear(self.dim_model, self.dim_model)
        self.out_proj = nn.Linear(self.dim_model, self.dim_model)

    def forward(self, x: Tensor, coords: Tensor):
        B, N, _ = x.shape
        # Q, K, and V matrices, multi-head attention.
        # (B, N, dim_model) --project--> (B, N, num_heads, head_dim)
        #                   --transpose--> (B, num_heads, N, head_dim)
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard Dot-product Attention
        # (B, num_heads, N, N)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))

        # Calculate 3D Euclidean distance between each pair of vertexes.
        # (B, N, N)
        dist_matrix = torch.cdist(coords, coords, p=2)

        # For each vertex, find the nearest K + 1 points, including itself.
        # (B, N, K + 1)
        _, nearest_indices = torch.topk(dist_matrix, k=self.K + 1, dim=-1, largest=False)

        # Initialize an all-False mask.
        # (B, N, N)
        band_mask = torch.zeros(B, N, N, device=x.device, dtype=torch.bool)

        # Mark the nearest K + 1 points as True.
        # batch_indices & row_indices: (B, N, K + 1)
        # band_mask: (B, N, N)
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).unsqueeze(2).expand(-1, N, self.K + 1)
        row_indices = torch.arange(N, device=x.device).unsqueeze(0).unsqueeze(2).expand(B, -1, self.K + 1)
        band_mask[batch_indices, row_indices, nearest_indices] = True

        # Apply Mask
        # attn_w: (B, N, N)
        attn_score_ = attn_score.masked_fill(~band_mask.unsqueeze(1), float('-inf'))
        attn_score = (attn_score_ + attn_score) / 2
        attn_w = torch.softmax(attn_score, dim=-1)

        # out: (B, N, dim_model)
        out = torch.matmul(attn_w, v).transpose(1, 2).reshape(B, N, self.dim_model)

        # Output Representation
        # out_rep: (B, N, dim_model)
        out_rep = self.out_proj(out)
        return out_rep


class JETransformerLayer(nn.Module):
    def __init__(self, dim_model: int, drop_rate: float, je_attention: JEAttention):
        super().__init__()

        if dim_model != je_attention.dim_model:
            raise ValueError(
                f"Dimension model must be equal to "
                f"je_attention.dim_model ({je_attention.dim_model}), "
                f"but got {dim_model}.")

        # Model Parameters
        self.dim_model = dim_model
        self.drop_rate = drop_rate

        # Model Components
        self.attn = je_attention
        self.norm1 = nn.LayerNorm(self.dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim_model, self.dim_model * 2),
            nn.GELU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.dim_model * 2, self.dim_model)
        )
        self.norm2 = nn.LayerNorm(self.dim_model)
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, x: Tensor, coords: Tensor):
        # First pass
        x = self.norm1(x + self.dropout(self.attn(x, coords)))

        # Second Pass
        x = self.norm2(x + self.ffn(x))
        return x, coords


class JETT(nn.Module):
    def __init__(self, dim_model: int, jets: ModuleList, out: nn.Module):
        super().__init__()
        self.dim_model = dim_model
        self.embedding = GraphEmbedding(dim_model=self.dim_model)
        self.jet_layers: ModuleList = jets
        self.norm = nn.LayerNorm(self.dim_model)
        # self.classifier = nn.Linear(self.dim_model, 8)
        self.out = out

    def forward(self, coords: Tensor):
        # Embed coordinates into features.
        # coords: (B, N, 3) --embed--> x: (B, N, dim_model)
        x = self.embedding(coords)

        # Pass through JETransformer layers.
        # x: (B, N, dim_model) --...--> (B, N, dim_model)
        for layer in self.jet_layers:
            layer = cast(JETransformerLayer, layer)
            x, coords = layer(x, coords)
        x = self.norm(x)

        # Aggregate global features.
        # global_feat = torch.mean(x, dim=1)

        # (B, N, dim_model) --> (B, N * dim_model)
        B, _, _ = x.shape
        global_feat = x.reshape(B, -1)
        out = self.out(global_feat)
        return out


class JETTBuilder:
    def __init__(self):
        self._dim_model: int = 256
        self._num_heads: int = 3
        self._K: int = 5
        self._drop_rate: float = 0.1
        self._num_layers: int = 3
        self._out_module: nn.Module = nn.Identity()

    def dim_model(self, dim_model: int):
        self._dim_model = dim_model
        return self

    def num_heads(self, num_heads: int):
        self._num_heads = num_heads
        return self

    def nearest_k(self, nearest_k: int):
        self._K = nearest_k
        return self

    def drop_rate(self, drop_rate: float):
        self._drop_rate = drop_rate
        return self

    def num_layers(self, num_layers: int):
        self._num_layers = num_layers
        return self

    def out_module(self, out_module: nn.Module):
        self._out_module = out_module
        return self

    def build(self):
        je_attention = JEAttention(
            dim_model=self._dim_model,
            num_heads=self._num_heads,
            nearest_k=self._K)

        jets = ModuleList([
            JETransformerLayer(
                dim_model=self._dim_model,
                drop_rate=self._drop_rate,
                je_attention=deepcopy(je_attention))
            for _ in range(self._num_layers)
        ])

        return JETT(dim_model=self._dim_model, jets=jets, out=self._out_module)
