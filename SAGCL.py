# -*- coding: utf-8 -*-
import argparse
import numpy as np
import time
import random
import torch
import torch.nn.functional as F
import dgl
from gat import GAT
from GAE import IGAE

from utils import eva, LoadDataset, load_graph_data, preprocess_features
from loss import multihead_contrastive_loss
import warnings
import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans


from sklearn.decomposition import PCA
from scipy.fftpack import fft
import itertools
import scipy.io as sio

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GAT')
parser.add_argument("--gpu", type=int, default=0,
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument("--epochs", type=int, default=200,
                    help="number of training epochs")
parser.add_argument("--epoch-secondstep", type=int, default=200,
                    help="number of training epochs")
parser.add_argument("--dataset", type=str, default="cora",
                    help="which dataset for training")
parser.add_argument("--num-heads", type=int, default=4,
                    help="number of hidden attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--num-hidden", type=int, default=32,
                    help="number of hidden units")
parser.add_argument("--tau", type=float, default=1,
                    help="temperature-scales")
parser.add_argument("--q-value", type=float, default=0.1,
                    help="q-scales")
parser.add_argument("--lam", type=float, default=0.025,
                    help="lambda-scales")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed")
parser.add_argument("--in-drop", type=float, default=0.6,
                    help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0.5,
                    help="attention dropout")
parser.add_argument("--lr", type=float, default=0.1,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help="weight decay")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--n-components', type=int, default=350) # ACM
parser.add_argument('--n-components-view2', type=int, default=350) # ACM
parser.add_argument('--gae_n_enc_1', type=int, default=32)

args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

features, Y, adj2, n_classes = load_graph_data('acm', show_details=True)
features, _ = preprocess_features(features)

# adj2, features, Y = load_network_data(args.dataset)
# features[features > 0] = 1
g = dgl.from_scipy(adj2)

if args.gpu >= 0 and torch.cuda.is_available():
    cuda = False
    # g = g.int().to(args.gpu)
else:
    cuda = False
print(cuda)

features_view2 = torch.Tensor(fft(features).astype(float))
features = torch.Tensor(features)

print(features_view2.shape)

# pca1 = PCA(n_components=args.n_components)
# x1 = pca1.fit_transform(features)
# dataset = LoadDataset(x1)
# features = torch.Tensor(dataset.x)
# print(features.shape)
#
# pca2 = PCA(n_components=args.n_components_view2)
# x1 = pca2.fit_transform(features_view2)
# dataset = LoadDataset(x1)
# features_view2 = torch.Tensor(dataset.x)
# print(features_view2.shape)
# km = KMeans(n_clusters=n_classes, n_init=20)
# y_pred = km.fit_predict(np.array(features))
#
# print(y_pred.shape)

# ACC = metrics.acc(Y, y_pred)
# nmi = metrics.NMI(Y, y_pred)
# ari = metrics.ARI(Y, y_pred)
# f1 = metrics.f_score(Y, y_pred)
# purity = metrics.purity_score(Y, y_pred)
#
# print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f,  f1: %.4f,  purity: %.4f  <==|'
#       % (ACC, nmi, ari, f1, purity))

f = open('NCLA_' + args.dataset + '_SAGCL.txt', 'a+')
f.write('\n\n\n{}\n'.format(args))
f.flush()

labels = Y
adj = torch.tensor(adj2.todense())

all_time = time.time()
num_feats = features.shape[1]
n_edges = g.number_of_edges()


# add self loop
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)

# create model
heads = ([args.num_heads] * args.num_layers)  # 4
model = GAT(g,
            args.num_layers, # 1
            num_feats, # 1433
            args.num_hidden, # 32
            heads, # 4
            F.elu,
            args.in_drop, # 0.6
            args.attn_drop, # 0.5
            args.negative_slope,
            n_classes)

model_view2 = GAT(g,
            args.num_layers, # 1
            features_view2.shape[1], # 1433
            args.num_hidden, # 32
            heads, # 4
            F.elu,
            args.in_drop, # 0.6
            args.attn_drop, # 0.5
            args.negative_slope, n_classes)


model.load_state_dict(torch.load('best_NCLA_acm.pkl'))
model.eval()
with torch.no_grad():
    heads, _ = model(features)

embeds = torch.cat(heads, axis=1)  # concatenate emb learned by all heads

km = KMeans(n_clusters=n_classes, n_init=20)
y_pred = km.fit_predict(embeds.detach().cpu())
# y_pred = cs[3].argmax(dim=1).detach().cpu()
# y_pred = np.array(y_pred).reshape(labels.shape)
print(y_pred.shape)

ACC = metrics.acc(labels, y_pred)
nmi = metrics.NMI(labels, y_pred)
ari = metrics.ARI(labels, y_pred)
f1 = metrics.f_score(labels, y_pred)
purity = metrics.purity_score(labels, y_pred)

print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f,  f1: %.4f,  purity: %.4f  <==|'
      % (ACC, nmi, ari, f1, purity))

model_view2.load_state_dict(torch.load('best_NCLA_acm_fft.pkl'))
model_view2.eval()
with torch.no_grad():
    heads_view2, _ = model_view2(features_view2)

embeds_view2 = torch.cat(heads_view2, axis=1)  # concatenate emb learned by all heads
# embeds = torch.cat((embeds, embeds_view2), axis=1)  # concatenate emb learned by all heads
# embeds = embeds.detach().cpu()
km = KMeans(n_clusters=n_classes, n_init=20)
y_pred = km.fit_predict(embeds_view2.detach().cpu())


ACC = metrics.acc(labels, y_pred)
nmi = metrics.NMI(labels, y_pred)
ari = metrics.ARI(labels, y_pred)
f1 = metrics.f_score(labels, y_pred)
purity = metrics.purity_score(labels, y_pred)

print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f,  f1: %.4f,  purity: %.4f  <==|'
      % (ACC, nmi, ari, f1, purity))

model_gae = IGAE(
        n_input=embeds.shape[1],
    n_clusters=n_classes

)

model_gae_view2 = IGAE(
        n_input=embeds_view2.shape[1],
        n_clusters=n_classes
    )

optimizer_gae = torch.optim.Adam(
    itertools.chain(model_gae.parameters(), model_gae_view2.parameters()), lr=args.lr, weight_decay=args.weight_decay)

best_acc = 0
best_nmi = 0
best_ari = 0
best_f1 = 0
best_epoch = 0
acc_reuslt = []
acc_reuslt.append(0)
nmi_result = []
ari_result = []
f1_result = []

for epoch in range(args.epoch_secondstep):

    model_gae.train()
    model_gae.zero_grad()
    model_gae_view2.train()
    model_gae_view2.zero_grad()
    c = model_gae(embeds)

    n = embeds_view2.shape[0]


    aug_c = model_gae_view2(embeds_view2)

    z_mat = torch.matmul(embeds, embeds_view2.T)

    # model_loss = 0.001*(model_gae.calc_loss(c.T, aug_c.T) + F.mse_loss(z_mat, torch.eye(n)) + model_gae.calc_loss(c, aug_c))
    model_loss = model_gae.calc_loss(c.T, aug_c.T) + (F.mse_loss(z_mat, torch.eye(n)) + model_gae.calc_loss(c, aug_c))

    model_loss.backward()
    optimizer_gae.step()
    model_gae.eval()
    model_gae_view2.eval()

    print('{} loss: {}'.format(epoch, model_loss))
    z = (c + aug_c) / 2
    i = z.argmax(dim=-1)
    acc, nmi, ari, f1 = eva(Y, i.data.numpy(), epoch)
    if acc > best_acc:
        print(z.shape)
        sio.savemat('sagcn.mat', {"fea": z.detach().numpy(), "label": Y})

        np.savez('acm_fea_wocac', x=z.detach().numpy(), y=Y)
        best_epoch = epoch
        best_acc = acc
        best_ari = ari
        best_f1 = f1
        best_nmi = nmi
    #    acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)
    acc_reuslt.append(acc)
    nmi_result.append(nmi)
    ari_result.append(ari)
    f1_result.append(f1)
print('Best Epoch_{}'.format(best_epoch), ':acc {:.4f}'.format(best_acc), ', nmi {:.4f}'.format(best_nmi),
      ', ari {:.4f}'.format(best_ari),
      ', f1 {:.4f}'.format(best_f1))
f.write('Best Epoch: %.4d, acc: %.4f, nmi: %.4f, ari: %.4f, f1: %.4f\n' % (best_epoch, best_acc, best_nmi, best_ari, best_f1))
f.flush()

f.close()
