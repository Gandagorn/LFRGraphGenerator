"""
Definition of ML models for training and testing on graph data.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch_sparse
from scipy.sparse import csr_matrix
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, SGConv
from torch_geometric.utils import (dense_to_sparse, to_dense_adj)
from torch_sparse import SparseTensor


class GCN(torch.nn.Module):
    """
    Source:
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
    """

    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    
class GAT(torch.nn.Module):
    """
    Source:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
    This implementation uses 8 heads and a dropout probability of 0.6.
    """
    
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        heads = 8
        self.conv1 = GATConv(num_node_features, 8, heads=heads, dropout=0.6)
        self.conv2 = GATConv(8*heads, num_classes, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
   
class GraphSAGE(torch.nn.Module):
    """
    Source:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
    """
    
    def __init__(self, num_node_features, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, 16)
        self.conv2 = SAGEConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.sigmoid(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    
class SGC(torch.nn.Module):
    """
    Source:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SGConv
    """
    
    def __init__(self, num_node_features, num_classes):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_node_features, num_classes, K=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    
class MLP(torch.nn.Module):
    """
    This MLP model uses a hidden layer with 16 neurons.
    """

    def __init__(self, num_node_features, num_classes):
        super(MLP, self).__init__()
        self.lin1 = Linear(num_node_features, 16)
        self.lin2 = Linear(16, num_classes)

    def forward(self, data):
        x = data.x

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)


class H2GCN(torch.nn.Module):
    """
    Source:
    https://arxiv.org/pdf/2006.11468.pdf
    In this implementation we use the max-pooling combination of the 
    representations.
    """

    def __init__(self, num_node_features, num_classes):
        super(H2GCN, self).__init__()

        self.dense1 = torch.nn.Linear(num_node_features, 64)
        self.dense2 = torch.nn.Linear(64 * 3, num_classes) # max pooling
        # self.dense2 = torch.nn.Linear((2**3 - 1) * 64, num_classes) # concat

        self.A1_line = None
        self.A2_line = None


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        device = data.x.device

        if self.A1_line == None:
            def convert_to_format(A):
                D = np.array(A.sum(axis=0))[0].astype(np.float32) 
                D[D!=0] = D[D!=0] ** (-1/2)
                D = np.diag(D)

                D_sparse = csr_matrix(D)
                res_dense = (D_sparse @ A @ D_sparse).toarray().astype(np.float32)
                return dense_to_sparse(torch.tensor([res_dense]))

            num_nodes = data.x.shape[0]

            # work around bc of costly matrix multiplication
            adj_mat_sparse = csr_matrix(to_dense_adj(data.cpu().edge_index)[0])
            data.to(device)

            I = torch.diag(torch.Tensor(np.ones(num_nodes)))
            I_sparse = csr_matrix(I)

            A1_sparse = (adj_mat_sparse - I_sparse > 0).astype(int)
            A2_sparse = ((((adj_mat_sparse @ adj_mat_sparse) > 0).astype(int) - adj_mat_sparse > 0).astype(int) - I_sparse > 0).astype(int)

            edge_index1, values1 = convert_to_format(A1_sparse)
            edge_index2, values2 = convert_to_format(A2_sparse)

            self.A1_line = SparseTensor(row=edge_index1[0], col=edge_index1[1], value=values1, sparse_sizes=(num_nodes, num_nodes))
            self.A2_line = SparseTensor(row=edge_index2[0], col=edge_index2[1], value=values2, sparse_sizes=(num_nodes, num_nodes))

            self.A1_line = self.A1_line.to(device)
            self.A2_line = self.A2_line.to(device)

        R = self.dense1(x)
        R = F.relu(R)

        Rs = [R]
        for k in range(2):
            R1_k = torch_sparse.matmul(self.A1_line, R)
            R2_k = torch_sparse.matmul(self.A2_line, R)

            # R = torch.cat((R1_k, R2_k), 1) # concat

            R = torch.stack((R1_k, R2_k), 0) # max pooling
            R = torch.amax(R, 0) # max pooling

            Rs.append(R)
        
        R_final = torch.cat(Rs, 1)
        R_final = F.dropout(R_final, training=self.training)
        out = self.dense2(R_final)

        return F.log_softmax(out, dim=1)

    
class EarlyStopping:
    """
    From https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss