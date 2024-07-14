# -*- coding: utf-8 -*-
import argparse
import numpy as np
import time
import random
import torch
import torch.nn.functional as F
import dgl
from gat import GAT

from utils import preprocess_features, LoadDataset, load_graph_data, random_planetoid_splits
from loss import multihead_contrastive_loss
import warnings
import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans


from sklearn.decomposition import PCA
from scipy.fftpack import fft

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GAT')
parser.add_argument("--gpu", type=int, default=0,
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument("--epochs", type=int, default=500,
                    help="number of training epochs")
parser.add_argument("--dataset", type=str, default="acm",
                    help="which dataset for training")
parser.add_argument("--num-heads", type=int, default=4,
                    help="number of hidden attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--num-hidden", type=int, default=32,
                    help="number of hidden units")
parser.add_argument("--tau", type=float, default=1,
                    help="temperature-scales")
parser.add_argument("--q-value", type=float, default=0.25,
                    help="q-scales")
parser.add_argument("--lam", type=float, default=0.025,
                    help="lambda-scales")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed")
parser.add_argument("--in-drop", type=float, default=0.6,
                    help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0.5,
                    help="attention dropout")
parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help="weight decay")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--n-components', type=int, default=350) # ACM

args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

features, Y, adj2, n_classes = load_graph_data(args.dataset, show_details=True)
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
features = torch.Tensor(features.astype(float))

# features = torch.Tensor(fft(features).astype(float))
# print(features.shape)

# pca1 = PCA(n_components=args.n_components)
# x1 = pca1.fit_transform(features)
# dataset = LoadDataset(x1)
# features = torch.Tensor(dataset.x)
print(features.shape)

km = KMeans(n_clusters=n_classes, n_init=20)
y_pred = km.fit_predict(np.array(features))

print(y_pred.shape)

ACC = metrics.acc(Y, y_pred)
nmi = metrics.NMI(Y, y_pred)
ari = metrics.ARI(Y, y_pred)
f1 = metrics.f_score(Y, y_pred)
purity = metrics.purity_score(Y, y_pred)

print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f,  f1: %.4f,  purity: %.4f  <==|'
      % (ACC, nmi, ari, f1, purity))

f = open('NCLA_' + args.dataset + '.txt', 'a+')
f.write('\n\n\n{}\n'.format(args))
f.flush()

labels = Y
adj = torch.tensor(adj2.todense())

all_time = time.time()
num_feats = features.shape[1]
print(n_classes)
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

# if cuda:
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()

# use optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# initialize graph
dur = []
test_acc = 0

counter = 0
min_train_loss = 1000
early_stop_counter = 100
best_t = -1
best_acc = 0
for epoch in range(args.epochs):
    if epoch >= 0:
        t0 = time.time()
    model.train()
    optimizer.zero_grad()
    heads, cs = model(features)

    loss = multihead_contrastive_loss(heads, adj, cs, args.q_value, args.lam, tau=args.tau)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        heads, _ = model(features)
        loss_train = multihead_contrastive_loss(heads, adj, cs, args.q_value, args.lam, tau=args.tau)
        embeds = torch.cat(heads, axis=1)
        km = KMeans(n_clusters=n_classes, n_init=20)
        y_pred = km.fit_predict(np.array(embeds))

        print(y_pred.shape)

        ACC = metrics.acc(Y, y_pred)
        nmi = metrics.NMI(Y, y_pred)
        ari = metrics.ARI(Y, y_pred)
        f1 = metrics.f_score(Y, y_pred)
        purity = metrics.purity_score(Y, y_pred)

        print(' ' * 8 + '|==>  epoch:%.4d  acc: %.4f,  nmi: %.4f,  ari: %.4f,  f1: %.4f,  purity: %.4f  <==|'
              % (epoch, ACC, nmi, ari, f1, purity))

        if ACC > best_acc:
            torch.save(model.state_dict(), 'best_NCLA_{}.pkl'.format(args.dataset))
    # early stop if loss does not decrease for 100 consecutive epochs
    if loss_train < min_train_loss:
        counter = 0
        min_train_loss = loss_train
        best_t = epoch
    else:
        counter += 1

    if counter >= early_stop_counter:
        print('early stop')
        break

    if epoch >= 0:
        dur.append(time.time() - t0)

    print("Epoch {:04d} | Time(s) {:.4f} | TrainLoss {:.4f} ".
          format(epoch + 1, np.mean(dur), loss_train.item()))

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_NCLA_{}.pkl'.format(args.dataset)))
model.eval()
with torch.no_grad():
    heads, cs = model(features)
embeds = torch.cat(heads, axis=1)  # concatenate emb learned by all heads
embeds = embeds.detach().cpu()
# print(embeds.shape)
print(labels.shape)

km = KMeans(n_clusters=n_classes, n_init=20)
y_pred = km.fit_predict(embeds)
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


f.close()
