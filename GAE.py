import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module,Parameter,Dropout


class GNNLayer(Module):

    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.Tanh()


        self.w = Parameter(torch.FloatTensor(out_features, in_features))


        torch.nn.init.xavier_uniform_(self.w)


    def forward(self, features, adj, active):
        if active:
          support = self.act(F.linear(features, self.w))  # add bias
        else:
          support = F.linear(features, self.w)  # add bias
        output = torch.mm(adj, support)
        return output

class IGAE_encoder(nn.Module):

    def __init__(self, gae_n_enc_1, n_input):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):

        z_igae = self.gnn_1(x, adj, active=False)

        return z_igae



class Cluster_layer(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Cluster_layer, self).__init__()


        self.l =  nn.Sequential(nn.Linear(in_dims, out_dims),
                                 nn.Softmax())


    def forward(self, h):
        c = self.l(h)
        return  c

class IGAE(nn.Module):

    def __init__(self, n_input, n_clusters):  # 1000 500 500
        super(IGAE, self).__init__()

        self.cluster = Cluster_layer(
            in_dims=n_input,
            out_dims=n_clusters,
        )


    def forward(self, x):

        c = self.cluster(x)

        return c

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        batch_size = x.shape[0]
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)

        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        if sym:

            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        #    print(pos_sim,sim_matrix.sum(dim=0))
            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
        else:
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()

        return loss

