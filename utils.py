import numpy as np
import torch
import scipy.io as sio
from torch.utils.data import Dataset
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn import metrics
from munkres import Munkres


class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]

from scipy.sparse import csr_matrix

def load_graph_data(dataset_name, show_details=False):
    """
    load graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :returns feat, label, adj: the features, labels and adj
    """
    load_path = "../data/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    category_num = max(label) - min(label) + 1

    row, col = np.nonzero(adj)
    values = adj[row, col]
    csr_adj = csr_matrix((values, (row, col)), shape=adj.shape)
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", csr_adj.shape)
        print("edge num:   ", int(csr_adj.sum() / 2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    import scipy.sparse as sp
    feat = sp.lil_matrix(feat)
    return feat, label, csr_adj, category_num

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    """For Reddit and PPI, recommend using the preprocessing strategy that is employed in William L. Hamilton/GraphSAGE"""
    #features = features.todense()
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #train_feats = features[train_index]
    #scaler.fit(train_feats)
    #features = scaler.transform(features)
    return features.todense(), sparse_to_tuple(features)

def load_network_data(name):
    net = sio.loadmat('../data/' + name + '.mat')
    X, A, Y = net['attrb'], net['network'], net['group']
    if name in ['cs', 'photo']:
        Y = Y.flatten()
        Y = one_hot_encode(Y, Y.max() + 1).astype(np.int)
    return A, X, Y


def random_planetoid_splits(num_classes, y, train_num, seed):
    # Set new random planetoid splits:
    # *  train_num * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    np.random.seed(seed)
    indices = []

    for i in range(num_classes):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:train_num] for i in indices], dim=0)

    rest_index = torch.cat([i[train_num:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    val_index = rest_index[:500]
    test_index = rest_index[500:1500]

    return train_index, val_index, test_index


def get_train_data(labels, tr_num, val_num, seed):
    np.random.seed(seed)
    labels_vec = labels.argmax(1)
    labels_num = labels_vec.max() + 1

    idx_train = []
    idx_val = []
    for label_idx in range(labels_num):
        pos0 = np.argwhere(labels_vec == label_idx).flatten()
        pos0 = np.random.permutation(pos0)
        idx_train.append(pos0[0:tr_num])
        idx_val.append(pos0[tr_num:val_num + tr_num])

    idx_train = np.array(idx_train).flatten()
    idx_val = np.array(idx_val).flatten()
    idx_test = np.setdiff1d(range(labels.shape[0]), np.union1d(idx_train, idx_val))

    idx_train = torch.LongTensor(np.random.permutation(idx_train))
    idx_val = torch.LongTensor(np.random.permutation(idx_val))
    idx_test = torch.LongTensor(np.random.permutation(idx_test))

    return idx_train, idx_val, idx_test

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]

        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')

    return acc, f1_macro

def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1