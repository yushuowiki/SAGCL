# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn

def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, adj,
                     mean: bool = True, tau: float = 1.0, q_: float = 0.1, lam: float = 0.015, hidden_norm: bool = True):
    l1 = rob_con_loss(z1, z2, tau, adj, q_, lam, hidden_norm)
    l2 = rob_con_loss(z2, z1, tau, adj, q_, lam, hidden_norm)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret

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

def multihead_contrastive_loss(heads, adj, cs, q_: float = 1.0, lam: float = 1.0, tau: float = 1.0):
    loss = torch.tensor(0, dtype=float, requires_grad=True)
    for i in range(1, len(heads)):
        loss = loss + contrastive_loss(heads[0], heads[i], adj, tau=tau, q_=q_, lam=lam) \
               #+ 0.025 * calc_loss(cs[0], cs[i]) + 0.025 * calc_loss(cs[0].T, cs[i].T)
    return loss / (len(heads) - 1)


def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def nei_con_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj, hidden_norm: bool = True):
    '''neighbor contrastive loss'''
    adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
    adj[adj > 0] = 1
    nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(torch.tensor(nei_count))

    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z1, z1, hidden_norm))
    inter_view_sim = f(sim(z1, z2, hidden_norm))

    loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
            intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
    loss = loss / nei_count  # divided by the number of positive pairs for each node

    return -torch.log(loss)

def rob_con_loss(z1: torch.Tensor, z2: torch.Tensor, T, adj, q_: float = 0.25, lam: float = 0.015, hidden_norm: bool = True):
    '''robust contrastive loss'''
    adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
    adj[adj > 0] = 1
    nei_count = torch.sum(adj, 1) * 2 + 1  # intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(torch.tensor(nei_count))

    intra_view_sim = rob_loss(z1, z1, 1., hidden_norm) #视角内
    inter_view_sim = rob_loss(z1, z2, 1., hidden_norm)

    pos = inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)
    pos = -(pos ** q_) / q_
    neg = intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag()
    neg = ((lam * (neg)) ** q_) / q_
    loss = (pos.mean() + neg.mean())*(2 * T)
    return loss

def rob_loss(q: torch.Tensor, k: torch.Tensor, T, hidden_norm: bool = True): # lam 0.025
    # normalize
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
    # gather all targets
    # k = concat_all_gather(k)


    neg = torch.exp(q.matmul(k.transpose(0, 1)) / T)
    pos = torch.exp(torch.sum(q * k, dim=-1) / T)
    return neg + pos

# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
#
#     output = torch.cat(tensors_gather, dim=0)
#     return output