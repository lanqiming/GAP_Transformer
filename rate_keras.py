# import torch
import tensorflow as tf
import numpy as np
import time

def compute_mfpt_msm(peq, msm):
    """
    Calculates the mean first passage times of the transition probability matrix.

    Parameters
    ----------
    peq : tf.Tensor(n_nodes)
        Eigenvector of the stationary distribution.
    msm : torch.Tensor(n_nodes,n_nodes)
        Transition probability matrix.

    Returns
    -------
    mfpt : tf.Tensor(n_nodes,n_nodes)
        Mean first passage time matrix.
    """
    n_nodes = peq.shape[0]
    identity = tf.eye(n_nodes, dtype=tf.float64)
    onevec = tf.ones((1, n_nodes), dtype=tf.float64)
    peq = tf.reshape(peq, (peq.shape[0], 1))

    q_inv = tf.linalg.inv(identity + tf.transpose(tf.matmul(peq, onevec)) - msm).numpy()
    identity = identity.numpy()
    peq = peq.numpy()

    mfpt = np.zeros((n_nodes, n_nodes), dtype=np.float64)


    for j in range(n_nodes):
        for i in range(n_nodes):
            mfpt[i,j] = 1. / peq[j, 0]*(q_inv[j, j]-q_inv[i, j]+identity[i, j])
            # mfpt = tf.tensor_scatter_nd_update(mfpt, indices=[[i, j]], updates=[1. / peq[j, 0]*(q_inv[j, j]-q_inv[i, j]+identity[i, j])])
            # mfpt = tf.tensor_scatter_nd_update(mfpt, indices=[[i, j]], updates=[1])

    mfpt = tf.convert_to_tensor(mfpt, dtype=tf.float64)
    
    return mfpt