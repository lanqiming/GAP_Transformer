from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from spektral.layers import GraphSageConv, GCNConv
from spektral.utils import gcn_filter
import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix



class GraphEmbeddingModule(Model):
    """
    This part of the network takes in a graph G = (V,E) and
    produces node embeddings of each node in the network using
    classical GraphSAGE layers.

    Note:
    -----
    Implementation of GraphSAGE layer based on https://github.com/danielegrattarola/spektral

    **This Module expects a sparse adjacency matrix.**

    Input:
    -----
    - Node features of shape `(n_nodes, n_node_features)`.

    Output:
    ------
    - Node embeddings of shape `(n_nodes, embed_dim)`.

    Arguments:
    ----------
    - `aggregator`: String, type of aggregation to use for the GraphSAGE layer.
        Must be one of the following: `'sum'`, `'mean'`,`'max'`, `'min'`
    - `adj`: Sparse binary adjacency matrix of shape `(n_nodes, n_nodes)`.
    - `embed_dim`: Integer, dimensionality of the output node embeddings.
    - `hidden_dim`: Integer, dimensionality of the hidden node embeddings.
    - `layers`: Integer, number of GraphSAGE layers to use.


    """
    def __init__(self,
                aggregator,
                adj,
                embed_dim=64,
                hidden_dim=128,
                layers=4):
        
        super().__init__()

        self.g_layer = []
        self.adj = tf.sparse.from_dense(adj)

        for i in range(layers):
            if i == 0:
                self.g_layer.append(
                    GraphSageConv(channels = hidden_dim,
                                    aggregate = aggregator,
                                    activation = tf.nn.relu
                                    ))
            elif i == layers - 1:
                self.g_layer.append(
                    GraphSageConv(channels = embed_dim,
                                    aggregate = aggregator,
                                    activation = tf.nn.relu
                                    ))             
            else:
                self.g_layer.append(
                    GraphSageConv(channels = hidden_dim,
                                    aggregate = aggregator,
                                    activation = tf.nn.relu
                                    ))
                
    def call(self, embeddings):
        for layer in self.g_layer:
            embeddings = layer([embeddings, self.adj])

        return embeddings



class PosGCN(tf.keras.layers.Layer):
    """
    This part generates the positional encoding of each node in the embedded graph.

    Note:
    -----
    **This layer expects a sparse adjacency matrix.**

    Input:
    -----
    - Node embeddings of shape `(n_nodes, embed_dim)`.

    Output:
    ------
    - embedded graph with positional information `(n_nodes, embed_dim)`.

    Arguments:
    ----------
    - `embed_dim`: Integer, dimensionality of the positional encoding.
    - `adj`: Sparse binary adjacency matrix of shape `(n_nodes, n_nodes)`.

    """
    def __init__(self, embed_dim, adj):
        super().__init__()
        self.gcn = GCNConv(embed_dim, activation='relu')
        self.graphSage = GraphSageConv(channels = embed_dim, ggregate = "mean", activation = tf.nn.relu)
        self.dense = Dense(embed_dim)
        self.adj = tf.sparse.from_dense(adj)



    def call(self, x):
        gcn_pos = self.gcn((x, self.adj))
        x = x + gcn_pos

        return x

class Residual(tf.keras.layers.Layer):
    """
    This part generates the residual connection of dropout.

    Input:
    -----
    - Node embeddings from decoding layer of shape `(n_nodes, embed_dim)`.

    Output:
    ------
    - embedded graph with dropout after residual connection `(n_nodes, embed_dim)`.

    Arguments:
    ----------
    - `dropout_rate`: Float, dropout rate.
    """

    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()


    
    def call(self, embed_layer):
        return self.add([embed_layer, self.dropout(embed_layer)])
        
# define base layers 
class BaseAttention(tf.keras.layers.Layer):
  
    """
    Base class for attention layers. Collection of all attention layers.

    Note:
    -----
    Referenced from: https://www.tensorflow.org/text/tutorials/transformer

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

# Multi-Head Attention + Add&Norm
class GlobalSelfAttention(BaseAttention):
    """
    This is a multi-head self attention layer.

    Input:
    -----
    - Node embeddings with positional information of shape `(n_nodes, embed_dim)`.

    Output:
    ------
    - result of multi-head self attention `(n_nodes, embed_dim)`.
    """
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x
            )
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

# FeedForward + Add&Norm
class FeedForward(tf.keras.layers.Layer):
    """
    This is a feed forward layer.

    Input:
    -----
    - Tensor of shape `(n_nodes, embed_dim)`.

    Output:
    ------
    - result of feed forward layer `(n_nodes, embed_dim)`.

    Arguments:
    ----------
    - `ed_dim`: Integer, dimensionality of the output, should be the same as input.
    - `hid_dim`: Integer, dimensionality of the hidden layer.
    - `dropout_rate`: Float, dropout rate.
    """

    def __init__(self, ed_dim, hid_dim, dropout_rate=0.1):
        super().__init__()

        self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(hid_dim, activation='relu'),
        tf.keras.layers.Dense(ed_dim),
        tf.keras.layers.Dropout(dropout_rate)
        ])

        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x

class DecoderLayer(tf.keras.layers.Layer):
    """
    This is a decoder layer in partioning module.

    Input:
    -----
    - Node embeddings with positional information of shape `(n_nodes, embed_dim)`.

    Output:
    ------
    - Tensor of shape `(n_nodes, embed_dim)`.

    Arguments:
    ----------
    - `key_dim`: Integer, size of each attention head for query and key.
    - `ed_dim`: Integer, dimensionality of the output in feed forward, should be the same as input.
    - `hid_dim`: Integer, dimensionality of the hidden layer in feed forward.
    - `num_heads`: Integer, number of attention heads.
    - `dropout_rate`: Float, dropout rate.
    """

    def __init__(self, key_dim, ed_dim,hid_dim, num_heads, dropout_rate):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=key_dim)

        self.ffn = FeedForward(ed_dim, hid_dim, dropout_rate)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x



class GCNGraphPartitioningModule(Model):
    """
    This is the graph partitioning module accepting node embeddings

    Input:
    -----
    - Node embeddings of shape `(n_nodes, embed_dim)`.

    Output:
    ------
    - Probability matrix of shape `(n_nodes, n_partitions)`.

    Arguments:
    ----------
    - `node_num`: Integer, number of nodes in the graph.
    - `key_dim`: Integer, size of each attention head for query and key.
    - `ed_dim`: Integer, dimensionality of the output in feed forward, should be the same as input.
    - `hid_dim`: Integer, dimensionality of the hidden layer in feed forward.
    - `num_heads`: Integer, number of attention heads.
    - `n_partitions`: Integer, number of partitions to be generated.
    - `adj`: Sparse binary adjacency matrix of shape `(n_nodes, n_nodes)`.
    - `dropout_rate`: Float, dropout rate.
    - `depth`: Integer, number of decoding layers, should be 2 or more.
    """
    def __init__(self, node_num, key_dim, ed_dim, hid_dim, num_heads, n_partitions, adj, dropout_rate, depth=2):
        super().__init__()

        self.ed = key_dim
        self.hid = hid_dim
        self.depth = depth
        self.n_parts = n_partitions
        self.decode_layers = []
        self.pos_gcn = PosGCN(ed_dim, adj)
        self.residual = Residual(dropout_rate)
        self.partitionsDense = Dense(n_partitions)

        for i in range(depth):
            self.decode_layers.append(DecoderLayer(key_dim, ed_dim, hid_dim, num_heads, dropout_rate))



    def call(self, embeddings):

        node_dim = embeddings.shape[0]
        embeddings = self.pos_gcn(embeddings) 
        embeddings = tf.reshape(embeddings, [1, node_dim ,embeddings.shape[1]])


        print(embeddings.shape)
        for i, layer in enumerate(self.decode_layers):
            embeddings = layer(embeddings)

            if i< self.depth - 1:
                embeddings = self.residual(embeddings)

        embeddings = self.partitionsDense(embeddings)
        embeddings = tf.reshape(embeddings, [node_dim, self.n_parts])


        return tf.nn.softmax(embeddings, axis=1)
    

