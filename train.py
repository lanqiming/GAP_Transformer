import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import Counter
from model_keras import GAPmodel
from rate_keras import compute_mfpt_msm
from loss_function_keras import (diff_kemeny_loss,
                           calc_assign_argmax,
                           gaussian_penalty,
                           DiffKemenyLoss)

from utils_keras import (get_feature_func,
                   image_graph,
                   rate_preds,
                   plot_losses,
                   get_neigh_list)


import argparse
from sklearn.decomposition import PCA
from tensorflow import linalg 

parser = argparse.ArgumentParser(
    prog="GAP-FrameWork",
    description="Hyper-Parameters for training",
    epilog="Happy training!"
)
parser.add_argument("-fd",
                    "--feature-dim",
                    action="store",
                    default=250,
                    type=int,
                    metavar="n_features",
                    dest="feat-dim")
parser.add_argument("-np",
                    "--n-partitions",
                    action="store",
                    default=4,
                    type=int,
                    metavar="n_partitions",
                    dest="n_part")
parser.add_argument("-ed",
                    "--embed-dim",
                    action="store",
                    default=32,
                    type=int,
                    metavar="embed_dimension",
                    dest="embedding_dim")
parser.add_argument("-hd",
                    "--hidden-dim",
                    action="store",
                    default=64,
                    type=int,
                    metavar="hidden_dimension",
                    dest="hidden_dim")

parser.add_argument("-lr",
                    "--learning_rate",
                    action="store",
                    default=1e-4,
                    type=float,
                    metavar="learning_rate",
                    dest="learning_rate")

parser.add_argument("-gl",
                    "--graph-layers",
                    action="store",
                    default=4,
                    type=int,
                    metavar="n_GraphSAGE_layers",
                    dest="n_graph_layers")

parser.add_argument("-ffl",
                    "--N-ff-layers",
                    action="store",
                    default=2,
                    type=int,
                    metavar="n_FF_layers",
                    dest="n_ff_layers")
parser.add_argument("-name",
                    "--Run-Name",
                    action="store",
                    default="testing",
                    type=str,
                    metavar="run_name",
                    dest="name")
parser.add_argument("-n-train",
                    "--N-Training-Cycles",
                    action="store",
                    default=3000,
                    type=int,
                    metavar="n",
                    dest="n_times")
parser.add_argument("-n-nodes",
                    "--Number-Nodes",
                    action="store",
                    default=250,
                    type=int,
                    metavar="n_nodes",
                    dest="n_nodes")
parser.add_argument("-cuda",
                    "--cuda",
                    action="store_true",
                    dest="cuda")
parser.add_argument("-g",
                    "--graph",
                    action="store",
                    metavar="Path to graph",
                    type=str,
                    default=None,
                    dest="path")
parser.add_argument("-bs",
                    "--batch-size",
                    action="store",
                    metavar="Nodes at once",
                    type=int,
                    default=250,
                    dest="bs"
                    )
parser.add_argument("-k",
                    "--kind",
                    action="store",
                    metavar="Nodes at once",
                    type=str,
                    default="mfpt",
                    dest="kind"
                    )
parser.add_argument("-s",
                    "--state",
                    action="store",
                    metavar="Starting Model State",
                    type=str,
                    default=None,
                    dest="state"
                    )
parser.add_argument("-save",
                    "--save-model",
                    action="store_true",
                    dest="save"
                    )
parser.add_argument("-loss",
                    "--loss-function",
                    action="store",
                    type=str,
                    default="kemeny",
                    metavar="Loss Function to use.",
                    dest="loss_function"
                    )
parser.add_argument("-feats",
                    "--features",
                    action="store",
                    type=str,
                    default="mfpt",
                    metavar="Features to use.",
                    dest="feats"
                    )
parser.add_argument("-agg",
                    "--aggregator",
                    action="store",
                    type=str,
                    default="mean",
                    metavar=" aggregation method to use.",
                    dest="aggregator"
                    )

parser.add_argument("-key",
                    "--key-dim",
                    action="store",
                    type=int,
                    default="4",
                    metavar="size of each attention head for query and key.",
                    dest="key"
                    )

parser.add_argument("-heads",
                    "--num-heads",
                    action="store",
                    type=int,
                    default="8",
                    metavar="number of attention heads.",
                    dest="num_heads"
                    )

# define training function
@tf.function
def train_step(x, model, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(x, training=True)
        prediction = tf.cast(prediction, dtype=tf.float64)
        loss_value  = diff_kemeny_loss(prediction, mfpt, peq)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value

# define training function for second training
def train_step2(x, model, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(x, training=True)
        prediction = tf.cast(prediction, dtype=tf.float64)
        loss_value  = diff_kemeny_loss(prediction, mfpt, peq)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value

if __name__ == '__main__':
    # Parse CLI args
    args = parser.parse_args()
    print(args)

    # Training variables
    n_nodes = args.n_nodes              # n_nodes in graph
    n_times = args.n_times              # n_times exposed to each graph
    n_partitions = args.n_part          # Number of partitions
    l_r = args.learning_rate            # Learning Rate
    cuda = args.cuda                    # Bool: Use cuda or not
    path = args.path                    # Path to nx.graph file
    run_name = args.name                # Name of folder
    batch_size = args.bs
    n_ff_layers = args.n_ff_layers      # Number of decoding layers
    n_g_layers = args.n_graph_layers    # Number of embedding layers
    hidden_dim = args.hidden_dim
    embed_dim = args.embedding_dim
    kind = args.kind
    state = args.state
    save = args.save
    loss_function = args.loss_function
    feats = args.feats
    agg = args.aggregator
    key = args.key
    num_heads = args.num_heads

    pretrain_accuracy = 0.1

    if not os.path.exists(f"./graphs/{run_name}"):
        os.makedirs(f"./graphs/{run_name}")
    
    # Defining the features
    feat_type = feats
    feats = feats.split("_")
    PCA_FLAG = False
    if len(feats) == 2:
        if feats[1] == "pca":
            PCA_FLAG = True
    feats = feats[0]

    # Now load all graph info
    markov_matrix = "./pentapeptide_transition_matrix.dat"
    markov = tf.constant(
        np.loadtxt(markov_matrix), dtype=tf.float64
    )
    adj = markov + tf.transpose(markov)
    adj = tf.where(adj >= 0.015, 1.0, 0.0)

    eigenvalues, eigenvectors = linalg.eig(tf.transpose(markov))
    EMBED_FLAG = False
    NN_FLAG = False

    eigenvalues, eigenvectors = linalg.eig(tf.transpose(markov))
    eigenvalues_sorted = tf.sort(tf.math.real(eigenvalues), direction='DESCENDING')
    indices = tf.argsort(tf.math.real(eigenvalues), direction='DESCENDING')
    peq = tf.math.real(eigenvectors[:, indices[0]])
    peq = tf.expand_dims(peq, 1)
    mfpt = compute_mfpt_msm(peq, markov)

    crisp =  tf.constant(
    np.loadtxt("./pentapeptide_node_probs_crisp_nstates_5.dat"),dtype=tf.float64)

    loss_initial = diff_kemeny_loss(crisp, mfpt, peq, penalty_func=None)

    print("initial loss is ", loss_initial.numpy())

    if feats == "mfpt":
        node_features = mfpt
    elif feats == "adj":
        node_features = adj
    elif feats == "markov":
        node_features = markov
    elif feats == "eigenvectors":
        node_features = eigenvectors
    elif feats == "embedding":
        node_features = markov
        EMBED_FLAG = True
    elif feats == "NN":
        node_features = tf.concat([markov, mfpt, adj], axis=1)
        NN_FLAG = True

    if PCA_FLAG:
        pca = PCA()
        node_features = pca.fit_transform(node_features)
        node_features = tf.convert_to_tensor(node_features)

    node_list = [*range(adj.shape[0])]
    input_nodes = [*range(adj.shape[0])]
    features = get_feature_func(node_list, node_features)

    print("Finsihsed pre processing graph.")

    progress_bar = tqdm(range(n_times),
                position=0,
                leave=True,
                disable=True,
                ncols=100,
                ascii=True)

    n_iters = len(input_nodes) // batch_size
   


    # load model if needed
    if state is not None:
        state_path = f"./graphs/{state}/model.h5"
        if os.path.isfile(state_path):
            model = tf.keras.models.load_model(state_path)
        else:
            print(f"Model {state} not found!")
            raise FileNotFoundError

    with open(f"./graphs/{run_name}/params.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}\t:\t{v}\n")

    
    # Train
    losses = []

    model = GAPmodel(aggregator=agg,
                adj=adj,
                n_partitions=n_partitions,
                node_num=n_nodes,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                key_dim = key,
                num_heads = num_heads,
                n_layers=n_g_layers,
                ff_layers=n_ff_layers,
                dropout_rate=0.1)
    
    model.build(input_shape=(250, 250))
    model.summary()

    optimizer_train=tf.keras.optimizers.Adam(learning_rate=l_r)

    # model(node_features)

    for k in progress_bar:
        for j in range(n_iters):
            loss = train_step(node_features, model, optimizer_train)
            losses.append(loss.numpy().item())
        
        print("epoch ", k, "  loss is ", loss.numpy().item())

        progress_bar.set_description_str(str(loss.numpy()),
                                         refresh=True)
    
    # Second training with dropout rate = 0
    model2 = GAPmodel(aggregator=agg,
            adj=adj,
            n_partitions=n_partitions,
            node_num=n_nodes,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            key_dim = key,
            num_heads = num_heads,
            n_layers=n_g_layers,
            ff_layers=n_ff_layers,
            dropout_rate=0.0)
    
    model2.build(input_shape=(250, 250))
    model2.summary()
    
    model2.set_weights(model.get_weights())

    optimizer_train2=tf.keras.optimizers.Adam(learning_rate=1e-4)

    for k in range(500):
        loss = train_step2(node_features, model2, optimizer_train2)
        losses.append(loss.numpy().item())
        print("epoch ", k, "  loss is ", loss.numpy().item())


    # get last node probs
    node_probs = model2(node_features, training=False)
    node_probs = tf.cast(node_probs, dtype=tf.float64)
    

    # print cluster result
    image_graph(adj, run_name, n_partitions, k, predictions=node_probs)
    ec, balance = rate_preds(adj, node_probs, n_partitions)
    print("EDGE CUT :", ec, "BALANCE :", balance)
    states = [tf.argmax(p).numpy() for p in node_probs]
    count = Counter(states)
    for k, v in count.items():
        print(f"State: {k} population: {v}")

    s_max = calc_assign_argmax(node_probs)
    s_max_np = s_max.numpy().astype(np.float64)
    s_max = tf.cast(s_max, dtype=tf.float64)

    np.savetxt(f"./graphs/{run_name}/s_matrix.dat", s_max_np)
    final_loss = diff_kemeny_loss(s_max, mfpt, peq, penalty_func=None).numpy().item()
    print("FINAL_LOSS", final_loss)

    # save losses graph
    plot_losses(losses, run_name)
    with open(f"./graphs/scores_{feat_type}_{agg}.txt", "a") as f:
        f.write(f"{run_name} {feats} {n_times} {final_loss}\n")

  






    

    