"""
This file includes helper functions for generating graphs, calculating 
statistics and training and evaluating ML models on graphs.
"""

import copy
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from community import community_louvain
from torch_geometric.data import Data
from tqdm import tqdm
from scipy.stats import chi2_contingency, ttest_rel

from .pytorch_models import GAT, GCN, H2GCN, MLP, SGC, EarlyStopping, GraphSAGE
from .graph_generation import *

torch.set_num_threads(1)


def train_eval_data(data, model_class, patience=50, num_epochs=500, verbose=False):
    """Train network on train data and use validation data for early stopping.
    The final accuracy on a test data is returned.

    Parameters
    ----------
    data : torch_geometric data
        Graph data

    model_class : torch.nn.Module
        GNN or MLP model

    patience : int, optional
        Number of epochs after which Early stopping is applied if there is no 
        improvement in accuracy on validation set, by default 50

    num_epochs : int, optional
        Number of maxmal epochs, by default 500

    verbose : bool, optional
        by default False

    Returns
    -------
    float
        accuracy on test set
    """


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class(len(data["x"][0]), len(set([t.item() for t in data["y"]]))).to(device)
    if verbose:
        print(model)
        
    data = data.clone().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # some datasets have different masking options, we always use the first
    if len(data.train_mask.shape) > 1:
        train_mask = data.train_mask[:, 0]
        test_mask = data.test_mask[:, 0]
        val_mask = data.val_mask[:, 0]
    else:
        train_mask = data.train_mask
        test_mask = data.test_mask
        val_mask = data.val_mask

    early_stopping = EarlyStopping(patience=patience, verbose=verbose)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        _, pred = model(data).max(dim=1)
        correct = int(pred[val_mask].eq(data.y[val_mask]).sum().item())
        acc = correct / int(val_mask.sum())
        early_stopping(-acc, model)
        
        if early_stopping.early_stop:
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[test_mask].eq(data.y[test_mask]).sum().item())
    acc = correct / int(test_mask.sum())

    if verbose:
        print(f"{model_class.__name__} Test Accuracy: {round(acc, 3)} after {epoch - patience} iterations")

    return round(acc, 4)


def get_homophily_edge_ratio(G):
    """Homophily measure used in https://arxiv.org/pdf/2006.11468.pdf.
    Adapted from implementation in
    https://github.com/GemsLab/H2GCN/blob/master/experiments/h2gcn/modules/graph_stats.py

    Parameters
    ----------
    G : networkx graph
        Graph object

    Returns
    -------
    float
        homophily edge ratio
    """    

    homophily_count = 0
    total_edges = 0
    for u, v in G.edges():
        if G.nodes[v]['y'] >= 0 and G.nodes[u]['y'] >= 0:
            if G.nodes[v]['y'] == G.nodes[u]['y']:
                homophily_count += 1
            total_edges += 1
    homophily_edge_ratio = float(homophily_count) / total_edges

    return homophily_edge_ratio


def get_K_hop_homophily(G, K=1):
    """Calculated Fraction of nodes with same label in K-Hop neighborhood.

    Parameters
    ----------
    G : networkx graph
        Graph object

    K : int, optional
        Neighborhood to be taken into account, by default 1

    Returns
    -------
    float
        K-Hop homophily
    """

    try:
        hs = []
        for u in G.nodes:
            L_u = G.nodes[u]["y"]

            # gives K-Hop neighborhood nodes
            N_u_dict = nx.single_source_shortest_path_length(G, u, cutoff=K)
            N_u = [key for key in N_u_dict if N_u_dict[key] == K]

            L_hop_N_u = np.zeros(len(N_u))
            for i, u in enumerate(N_u):
                L_hop_N_u[i] = G.nodes[u]["y"]

            h = sum(L_hop_N_u == L_u) / len(N_u)
            hs.append(h)

        return np.mean(hs)
    except:
        return np.nan


def calc_uncertainty(data, community=None):
    """Calculates RMI as used in https://arxiv.org/pdf/2010.16245.pdf.
    Code is adapted from https://github.com/sqrhussain/structure-in-gnn.

    Parameters
    ----------
    data : torch_geometric data
        Graph data

    community : dict, optional
        Can be used to evaluate pre-defined community. If None, the louvaine 
        algorithm is used to find the communities, by default None

    Returns
    -------
    float
        RMI value
    """

    G = data_to_nx(data)
    
    if community == None: # use louvaine to find communities
        partition = community_louvain.best_partition(G)
        community = np.zeros(len(G.nodes))
        for node, c in partition.items():
            community[node] = c

    def agg(x):
        return len(x.unique())

    df_community = pd.DataFrame({"community": community, "label": np.array(data.y), "node": np.arange(len(data.y))})
    
    communities = df_community.community.unique()
    labels = df_community.label.unique()

    mtx = df_community.pivot_table(index='community', columns='label', values='node', aggfunc=agg).fillna(0) / len(df_community)
    
    def Pmarg(c):
        return len(df_community[df_community.community == c]) / len(df_community)
    
    def Pcond(l, c):
        return mtx.loc[c, l] / Pmarg(c)
    
    H = 0
    for c in communities:
        h = 0
        for l in labels:
            if Pcond(l, c) == 0:
                continue
            h += Pcond(l, c) * np.log2(1. / Pcond(l, c))
        H += h * Pmarg(c)
    
    def Pl(l):
        return len(df_community[df_community.label == l]) / len(df_community)
    
    Hl = 0
    for l in labels:
        if Pl(l) == 0:
            continue
        Hl += Pl(l) * np.log2(1. / Pl(l))
    
    IG = Hl - H
    return IG / Hl


def get_num_comms(G, iter=4):
    """
    Get the mean number and standard deviation of found louvaine communities 
    over several iterations in a graph.

    Parameters
    ----------
    G : networkx graph
        Graph object

    iter : int, optional
        Number of iterations to perform the louvaine algorithm, by default 4

    Returns
    -------
    tuple
        mean and standard deviation of the number of communities
    """

    num_comms = []
    for i in range(iter):
        num_comms.append(len(np.unique(list(community_louvain.best_partition(G).values()))))

    return np.mean(num_comms), np.std(num_comms)


def get_nxgraph_stats(G, data):
    """Calculates various statistics of graph.

    Parameters
    ----------
    G : networkx graph
        Graph object

    data : torch_geometric data
        Graph data

    Returns
    -------
    dict
        Dictionary containing the found statistics.
    """

    if not nx.is_connected(G):
        G = connect_components(copy.deepcopy(G))

    stats_dict = {}
    stats_dict["Number of edges"] = G.number_of_edges()
    stats_dict["Number of nodes"] = G.number_of_nodes()
    stats_dict["Number of features"] = data.x.shape[1]

    stats_dict["Homophily edge ratio"] = np.round(get_homophily_edge_ratio(G), 2)
    stats_dict["Homophily 1-hop"] = np.round(get_K_hop_homophily(G, 1), 2)
    stats_dict["Homophily 2-hop"] = np.round(get_K_hop_homophily(G, 2), 2)
    stats_dict["RMI"] = np.round(calc_uncertainty(data), 2)

    stats_dict["Degree power law exponent"] = round(calc_degree_exponent(G), 2)

    degrees = np.array(list(dict(G.degree()).values()))
    stats_dict["Average degree"] = round(np.mean(degrees), 2)

    stats_dict["Number of different labels (y)"] = len(set(nx.get_node_attributes(G, 'y').values()))
    stats_dict["Label power law exponent"] = round(calc_community_exponent(G), 2)

    mean_comms, std_comms = get_num_comms(G)
    stats_dict["Number of Louvaine Communities"] = f"{mean_comms:.2f} (+- {std_comms:.2f})"

    stats_dict["Avg clustering coefficient"] = round(nx.average_clustering(G), 2)

    return stats_dict


def eval_graph_on_models(data):
    """Collect accuracies of several ML models applied on the graph data.

    Parameters
    ----------
    data : torch_geometric data
        Graph data

    Returns
    -------
    dict
        Dictionary containing the accuracies of the models on the test set
    """

    results_dict = {}
    results_dict["GCN Acc"] = train_eval_data(data, GCN)
    results_dict["GAT Acc"] = train_eval_data(data, GAT)
    results_dict["GraphSAGE Acc"] = train_eval_data(data, GraphSAGE)
    results_dict["SGC Acc"] = train_eval_data(data, SGC)
    results_dict["H2GCN Acc"] = train_eval_data(data, H2GCN)
    results_dict["MLP Acc"] = train_eval_data(data, MLP)

    return results_dict


def plot_nxgraph(G, ax, name="Unnamed"):
    """Plots networkx graph with color coded communities

    Parameters
    ----------
    G : networkx graph
        Graph object

    ax : matplotlib axes
        Axis for plot to be drawn

    name : str, optional
        Name of graph, by default "Unnamed"
    """

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=30, cmap=plt.cm.gist_rainbow, 
        node_color=[G.nodes[node]["y"] for node in G.nodes], ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    ax.set_title(f"Graph {name}")


def plot_degree_dist(G, label=""):
    """Plots degree distribution of Graph

    Parameters
    ----------
    G : networkx graph
        Graph object

    label : str, optional
        by default ""
    """

    degrees = np.array(list(dict(G.degree()).values()))
    degree_dist = pd.Series(degrees).sort_values(ascending=False)
    plt.loglog(np.arange(0, len(degrees)), degree_dist.values, label=label)


def calc_degree_exponent(G):
    """Calculate exponent of degree distribution.
    Adapted from 
    https://github.com/kunegis/konect-handbook/raw/master/konect-handbook.pdf

    Parameters
    ----------
    G : networkx graph
        Graph object

    Returns
    -------
    float
        Power-law exponent of degree distribution
    """    

    n = G.number_of_nodes()
    degrees = np.array(list(dict(G.degree()).values()))
    dmin = min(degrees)

    gamma = 1 + n * np.sum( np.log(degrees / dmin) )**(-1)
    return gamma


def calc_community_exponent(G):
    """Calculate exponent of community distribution.
    Adapted from 
    https://github.com/kunegis/konect-handbook/raw/master/konect-handbook.pdf

    Parameters
    ----------
    G : networkx graph
        Graph object

    Returns
    -------
    float
        Power-law exponent of community distribution
    """    

    n = G.number_of_nodes()
    comm_values = pd.Series(list(nx.get_node_attributes(G, 'y').values())).value_counts().values
    
    dmin = min(comm_values)

    comm_exponent = 1 + n * np.sum( np.log(comm_values / dmin) )**(-1)
    return comm_exponent


def eval_data(data, evaluate_models=True):
    """Combines ML model results and statistics for graph data.

    Parameters
    ----------
    data : torch_geometric data
        Graph data

    evaluate_models : bool, optional
        If True, ML models are executed with graph and their accuracies added to 
        the result, by default True

    Returns
    -------
    dict
        Dictionary containing the graph statistics and the accuracies of the 
        models on the test set
    """

    if evaluate_models:
        res = eval_graph_on_models(data)
    else:
        res = {}

    stats = get_nxgraph_stats(data_to_nx(data), data)

    return {**res, **stats}


def evaluate_sensitive_feature(iterations=10, sensitive_bias=0, feature_cov=1, num_communities=7):
    """Performs the evaluation of the graphs with a sensitive variable over 
    several iterations.

    Parameters
    ----------
    iterations : int, optional
        Number of graphs to evaluate, by default 10

    sensitive_bias : int, optional
        Bias to make the sub-community probabilites more extrem, by default 0

    feature_cov : float, optional
        Diagonal values of covariance matrix of feature gaussians. A 
        higher value gives more overlapping gaussians and makes predictions 
        harder, by default 1

    Returns
    -------
    list
        contains the evaluation results of each iteration
    """

    eval_res = []
    for i in tqdm(range(iterations)):

        while True:
            try:
                found_d, seed = search_graph_parallel( 
                    n=2708, 
                    mu=0.2,
                    avg_degree=4,
                    feature_cov=feature_cov
                )
                break
            except:
                pass

        reduced_found_d, reduced_data_sensitive_controlled, reduced_data_sensitive_random, labels_df = combine_subcommunities(found_d, num_communities, sensitive_bias)

        eval_res.append(
            pd.DataFrame(
                [
                    eval_data(reduced_data_sensitive_controlled, evaluate_models=True), 
                    eval_data(reduced_data_sensitive_random, evaluate_models=True)
                ], 
                index=["LFR Cora - sensitive controlled", "LFR Cora - sensitive random"]
            ).T
        )
    
    return eval_res


def compute_metrics_over_iterations(eval_res):
    """Computes the mean and standard deviation the sensitive feature evaluation.

    Parameters
    ----------
    eval_res : list
        contains the evaluation results of each iteration

    Returns
    -------
    pd.DataFrame
        Mean and standard deviation of evaluation results
    """

    controlled = pd.DataFrame( 
        [eval_res[i]["LFR Cora - sensitive controlled"] for i in range(len(eval_res))], 
        index=list(range(len(eval_res)))
    )
    randomised = pd.DataFrame( 
        [eval_res[i]["LFR Cora - sensitive random"] for i in range(len(eval_res))], 
        index=list(range(len(eval_res)))
    )

    return pd.DataFrame(
        {
        "Mean sensitive controlled": controlled.mean().round(2).astype(str).str.cat(controlled.std().round(2).astype(str), sep=' (+/- ') + ')',
        "Mean sensitive random": randomised.mean().round(2).astype(str).str.cat(randomised.std().round(2).astype(str), sep=' (+/- ') + ')',
        }    
    )


def compute_gains(eval_res):
    """Computes the gain of the accuracies of the controlled sensitive feature 
    over the zodiac features a t-test is performed.

    Parameters
    ----------
    eval_res : list
        contains the evaluation results of each iteration

    Returns
    -------
    pd.DataFrame
        gain results of evaluation
    """
        
    controlled = pd.DataFrame( 
        [eval_res[i]["LFR Cora - sensitive controlled"] for i in range(len(eval_res))], 
        index=list(range(len(eval_res)))
    )
    randomised = pd.DataFrame( 
        [eval_res[i]["LFR Cora - sensitive random"] for i in range(len(eval_res))], 
        index=list(range(len(eval_res)))
    )
    acc_cols = [col for col in controlled.columns if col.endswith(" Acc")]

    # convert accuracies to % for easier visualization
    controlled[acc_cols] = controlled[acc_cols] * 100
    randomised[acc_cols] = randomised[acc_cols] * 100

    controlled_random_df = controlled[acc_cols] - randomised[acc_cols]

    gain_df = pd.DataFrame(
        {
        "controlled gain to randomised": controlled_random_df.mean().round(2).astype(str).str.cat(controlled_random_df.std().round(2).astype(str), sep=' (+/- ') + ')',
        } 
    )

    pvalue_df = pd.DataFrame(
        {
            "controlled gain to randomised": ttest_rel(controlled[acc_cols], randomised[acc_cols]).pvalue,
        }, index=gain_df.index
    ).round(2).astype(str)

    for col in gain_df:
        gain_df[col] = gain_df[col].str.cat(pvalue_df[col], sep=", p=")

    return gain_df.loc[acc_cols]