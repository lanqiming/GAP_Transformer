import numpy as np
# import torch
import tensorflow as tf
import os
import random
import networkx as nx
from matplotlib import pyplot as plt
from networkx.generators import erdos_renyi_graph as er_graph
from networkx.generators import scale_free_graph as sfg
from networkx.linalg import adjacency_matrix
from sklearn.decomposition import PCA
from collections import Counter
from rate_keras import *

def featureless_random_graph(n_nodes, prob=0.5, kind="clusters",
                             feature_size=250,prob_extra=0.005,n_clusters=6
                            ):
    """
    Generates a random graph and then performs PCA on it.
    Stores the results as a feature matrix

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    prob : float, default=0.1
        Probability for edge creation if kind="er".
    kind : str, default="er"
        "er" (Erdős-Rényi graph), "sf" (Scale-Free graph), "connected" (), or "matlab"
    feature_size : int, default=10
        Number of components to keep in the PCA, i.e., dimension of the PCA to return.

    Returns
    -------
    adjacency_m : tf.Tensor(n_nodes,n_nodes)
        Adjacency matrix.
    node_degrees : tf.Tensor(n_nodes,2)
        Node degree matrix.
    node_features : tf.Tensor(n_nodes,feature_size)
        Node features matrix.
    """
    # Generate graph
    if kind == "sf":
        G = sfg(n_nodes)
    elif kind == "mfpt":
        #G = np.loadtxt("./Examples/pentapeptide/pentapeptide_adj.dat")
        T = np.loadtxt("./pentapeptide_transition_matrix.dat")
        adj = T + T.transpose()
        adj[adj >= 0.015] = 1.
        adj[adj < 0.015] = 0.
        G = adj
        #G = np.loadtxt("./adj_0.015.dat")
        G = nx.from_numpy_array(G)
    else:
        raise NotImplementedError(f"Graph kind {kind} is not implemented.")

    # Generate adjacency matrix
    adjacency_m = adjacency_matrix(G)  # This returns a SPARSE matrix!
    adjacency_m = adjacency_m.toarray()
    adjacency_m = tf.convert_to_tensor(adjacency_m)

    # Generate node degree matrix
    node_degrees = [*G.degree()]
    node_degrees = tf.convert_to_tensor(node_degrees, dtype=tf.float32)

    # Generate node features
    # if kind == "rate":
    #     potential = linear_pot_irregular(100)
    #     rate = create_rate_one_dim(potential, 100, 1 / 0.596, 1.)
    #     node_features = rate
    # elif kind == "clusters":
    #     node_features = gen_rate_from_adj(adjacency_m)
    if kind == "mfpt":
        # import torch.linalg as linalg
        from tensorflow import linalg 

        markov = tf.constant(np.loadtxt("./pentapeptide_transition_matrix.dat") , dtype=tf.float64)
        eigenvalues, eigenvectors = linalg.eig(tf.transpose(markov))
        eigenvalues_sorted = tf.sort(tf.math.real(eigenvalues), direction='DESCENDING')
        indices = tf.argsort(tf.math.real(eigenvalues), direction='DESCENDING')
        peq = tf.math.real(eigenvectors[:, indices[0]])
        peq = tf.expand_dims(peq, 1)
        mfpt = compute_mfpt_msm(peq, markov)
        #node_features = mfpt
        #np.savetxt("peq.dat", peq.detach().numpy())
        #np.savetxt("markov.dat", markov.detach().numpy())
        #np.savetxt("mfpt.dat", mfpt.detach().numpy())
        n_comps = min(n_nodes, feature_size)
        pca = PCA(n_components=n_comps)
        node_features = pca.fit_transform(mfpt)
        node_features = tf.convert_to_tensor(node_features)
        print("FEATS SHAPES", node_features.shape, n_nodes, feature_size)

        #node_features = pca.fit_transform(mfpt)
        #node_features = torch.Tensor(np.loadtxt("./Examples/pentapeptide/pentapeptide_adj.dat")).t().double()
    else:
        n_comps = min(n_nodes, feature_size)
        pca = PCA(n_components=n_comps)
        node_features = pca.fit_transform(adjacency_m)
        node_features = tf.convert_to_tensor(node_features)

    return adjacency_m, node_degrees, node_features


def get_feature_func(nodes, features):
    """
    Helper function to allow the node to call initial features like it would a
    layer output.
    """
    def feature_func(node):
        return features[node]

    return feature_func


def image_graph(adjacency_matrix, run_name, n_partitions, sub_run_id=1, base_path=f"./graphs/", predictions=None):
    """
    Save image of graph defined by adjacency_matrix.

    Parameters
    ----------
    adjacency_matrix : tf.Tensor(n_nodes,n_nodes)
        Adjacency matrix.
    run_name : str
        Name of the current run.
    n_partitions : int
        Number of partitions.
    sub_run_id : int
        Number of subrun for current ruNone    base_path : str, default=f"./graphs/"
        Name of the base path where the image of graph will be saved.
    predictions : tf.Tensor(n_nodes,n_parts)
        Final clustering predictions.

    Returns
    -------
        None
    """
    # Reconstruct graph from adjacency matrix
    G = nx.Graph(adjacency_matrix.numpy())
    path = base_path + "{}/".format(run_name)

    if predictions is not None:
        test_labels = gen_test_labels(predictions, n_partitions)
    else:
        test_labels = None

    # Check if dir given by path exists. If not, create it.
    if not os.path.isdir(path):
        os.makedirs(path)

    # Create fig, draw graph, save figure, and close plot
    plt.figure(figsize=(12, 12))
    nx.draw(G, node_color=test_labels, with_labels=True)
    plt.savefig(path + str(sub_run_id) + "-result.png")
    plt.close()

    return

def gen_test_labels(probabilities, n_partitions):
    """
    Generate colour labels.

    Parameters
    ----------
    probabilities : tf.Tensor
        Tensor with probabilities.

    Returns
    -------
    colour: List of str
        List containing the color to attribute to each node.
    """
    test_labels = [tf.math.argmax(p).numpy() for p in probabilities]
    colour = []
    cmap_name = "viridis"
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, n_partitions))

    for i, p in enumerate(test_labels):
        colour.append(colors[p])
    return colour

def rate_preds(adj, preds, n_states):
    """
    Rate the predictions in terms of performance.

    Parameters
    ----------
    adj : tf.Tensor(n_nodes,n_nodes)
        Adjacency matrix.
    preds : TODO: ?? lists of lists ??
        Predictions for each node.
    n_states : int
        Number of partitions. TODO:


    Returns
    -------
    edge_cut : float
        Edge cut value.
    edge_balance : float
        Balancedness value.
    """
    n_nodes = adj.shape[0]
    # Take the predicted state
    # CURRENTLY THIS IS TAKING MAX VALUE NOT INDEX!!
    states = [tf.argmax(p).numpy() for p in preds]
    # Get edge-list (i.e., indices of the elements of the adjacency matrix that are non-zero)
    edges = tf.where(tf.not_equal(adj, 0)).numpy()
    # Calculate edge cut
    edge_cut = calc_edge_cut(edges, states)
    # Calc balance
    balance = calc_balance(states, n_states)

    return edge_cut, balance

def calc_edge_cut(edges, states):
    """
    Edge cut is the ratio of the cut to the total number of edges.

    Parameters
    ----------
    edges : iterable of iterables
        Iterable of iterables of edge indices (e.g., [[1,2],[6,7]])
    states : List
        List of predicted states.

    Returns
    -------
    edge_cut : float
        Edge cut value.
    """
    n_edges = len(edges)
    states = {i: s for i, s in enumerate(states)}

    cuts = 0
    for (state1, state2) in edges:
        if states[state1] != states[state2]:
            # If two nodes connected by an edge do not belong to the same state
            cuts += 1

    return cuts / n_edges


def calc_balance(states, n_states):
    """
    Calculate the balancedness.
    Balancedness is defined as 1 minus the MSE of the number of nodes in every partition 
    and balances partition (n/g).

    Notes
    -----
    B = 1 - MSE(node_populations,n_nodes/n_parts)
    B = 1 - 1/n_nodes*sum(node_pop - n_nodes/n_parts)**2

    Parameters
    ----------
    states : List

    Returns
    -------
    balancedness : float
        Balancedness value.
    """
    n_nodes = len(states)

    # dict {partition_i: number of nodes in that partition_i)
    node_pops = Counter(states)

    # Number of nodes per partition (n/g)
    balance = 1 / n_states

    # Calculate balancedness (v iterates overall partitions)
    mse = sum([(v / n_nodes - balance) ** 2 for v in node_pops.values()])
    # print([(v/n_nodes - balance)**2 for v in node_pops.values()])
    # print(balance, n_states)
    mse *= 1 / n_states
    balancedness = 1 - mse

    return balancedness


def plot_losses(losses, run_name):
    """
    Plot evolution of the loss function as a function of the epoch number.

    Parameters
    ----------
    losses : list of float or np.array
        Array containing values of the loss function.
    run_name : str
        Name of the current run.

    Returns
    -------
    None
    """
    if isinstance(losses[0],float):
        path = f"./graphs/{run_name}/loss_kemeny.png"
        epochs = np.arange(len(losses))
        plt.plot(epochs, losses)
        plt.savefig(path)
        plt.close()

    elif len(losses[0]) == 3:
        loss_all = [l[0] for l in losses]
        loss_edge = [l[1] for l in losses]
        loss_balance = [l[2] for l in losses]

        path = f"./graphs/{run_name}/loss_both.png"
        epochs = np.arange(len(losses))
        plt.plot(epochs, loss_all)
        plt.savefig(path)
        plt.close()

        path = f"./graphs/{run_name}/loss_edge.png"
        epochs = np.arange(len(losses))
        plt.plot(epochs, loss_edge)
        plt.savefig(path)
        plt.close()

        path = f"./graphs/{run_name}/loss_balance.png"
        epochs = np.arange(len(losses))
        plt.plot(epochs, loss_balance)
        plt.savefig(path)
        plt.close()
    return


def get_neigh_list(adj_m, to_set=True):
    """
    Get list of neighbours for each node.
    Given an adjacency matrix of size (n_nodes,n_nodes), where n_nodes is the number of nodes,
    this function returns a list of n_nodes sets in which each set contains the neighbour ids of the corresponding node.

    Parameters
    ----------
    adj_m : torch.Tensor(n_nodes,n_nodes)
        Adjacency matrix
    to_set : bool, default=True
        TODO:

    Returns
    -------
    adj_list : List(n_nodes) of sets
        List of n_nodes sets containing the neighbours ids of each node.
    """
    adj_list = []

    for node in adj_m.numpy().tolist():
        neigh_list = []
        for neigh_id in range(len(node)):
            if node[neigh_id] != 0:
                if not to_set:
                    for j in range(node[neigh_id]):
                        neigh_list.append(neigh_id)
                else:
                    neigh_list.append(neigh_id)
        adj_list.append(set(neigh_list))

    return adj_list