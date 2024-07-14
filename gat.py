"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch.nn as nn
from dgl.nn.pytorch import GATConv

class Cluster_layer(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Cluster_layer, self).__init__()


        self.l =  nn.Sequential(nn.Linear(in_dims, out_dims),
                                 nn.Softmax())


    def forward(self, h):
        c = self.l(h)
        return  c

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 n_clusters):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers # 1
        self.num_hidden = num_hidden # 32
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.n_clusters = n_clusters
        self.cluster = nn.ModuleList()

        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))

        for l in range(heads[0]):
            self.cluster.append(Cluster_layer(
                in_dims=num_hidden,
                out_dims=n_clusters,
            ))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))

    def forward(self, inputs):
        heads = []
        h = inputs
        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)
            h =self.gat_layers[l](self.g, temp)

        cs = []
        # get heads
        for i in range(h.shape[1]):
            heads.append(h[:, i])
            cs.append(self.cluster[l](h[:, i]))

        return heads, cs
