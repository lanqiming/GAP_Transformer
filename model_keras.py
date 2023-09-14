from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from spektral.layers import graphsage_conv
import tensorflow as tf
from modules_keras import GraphEmbeddingModule, GCNGraphPartitioningModule

class GAPmodel(Model):
    """
    This is the model for GAP basing on transformer with GNN positional encoding.
    
    Input:
    -----
    - Node features of shape `(n_nodes, n_node_features)`.

    Output:
    ------
    - Probability matrix of shape `(n_nodes, n_partitions)`.

    Arguments:
    ----------
    - `aggregator`: String, aggregator for GraphSAGE convolution.
        Must be one of the following: `'sum'`, `'mean'`,`'max'`, `'min'`.
    - `adj`: SparseTensor, adjacency matrix of the graph.
    - `n_partitions`: Integer, number of partitions to cluster the graph into.
    - `node_num`: Integer, number of nodes in the graph.
    - `embed_dim`: Integer, dimensionality of the node embeddings.
    - `hidden_dim`: Integer, dimensionality of the hidden node embeddings.
    - `key_dim`: Integer, size of each attention head for query and key.
    - `num_heads`: Integer, number of attention heads.
    - `n_layers`: Integer, number of embedding layers in the graph embedding module.
    - `ff_layers`: Integer, number of decoding layers in the graph embedding module.
    - `dropout_rate`: Float, dropout rate.

    """

    def __init__(self,
            aggregator,
            adj,
            n_partitions=4,
            node_num=250,
            embed_dim=64,
            hidden_dim=256,
            key_dim=64, 
            num_heads=8,
            n_layers=4,
            ff_layers=2,
            dropout_rate = 0.1):
        
        super().__init__()

        self.embedding = GraphEmbeddingModule(aggregator = aggregator,
                                            adj = adj,
                                            embed_dim = embed_dim,
                                            hidden_dim = hidden_dim,
                                            layers = n_layers)
        

        self.CNNpartitioning = GCNGraphPartitioningModule(node_num = node_num,
                                            key_dim = key_dim,
                                            ed_dim=embed_dim,
                                            hid_dim = hidden_dim,
                                            num_heads = num_heads,
                                            n_partitions = n_partitions,
                                            adj= adj,
                                            dropout_rate = dropout_rate,
                                            depth = ff_layers)
        
        
    def call(self, nodes):
        nodes = self.embedding(nodes)
        probability = self.CNNpartitioning(nodes)

        return probability