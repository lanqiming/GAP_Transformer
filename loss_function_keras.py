import tensorflow as tf
from tensorflow import linalg 
import numpy as np

def gaussian_penalty(node_probs, mean=0., std_dev=0.1, v=15, soft_max_factor=100.):
    """
    Gaussian penalty function.

    Parameters
    ----------
    node_probs : tf.Tensor(n_nodes, n_partitions)
        Probability matrix returned by the GAP model.
    mean : float
        Mean of the Gaussian.
    std_dev : float
        Standard deviation of the Gaussian.
    v : float
        Magnitude of the penalty potential.

    Returns
    -------
    penalty_func : tf.tensor(1)
        Value of the penalty function.
    """
    # Softmax, i.e., differentiable version of max (normalised)
    S_exp = tf.math.exp(soft_max_factor*node_probs)
    S = S_exp / tf.math.reduce_sum(S_exp, axis = 1, keepdims=True)
    nodes_per_part = tf.reduce_sum(S, axis = 0)

    # Penalty function
    penalty_func = v*tf.math.exp(-0.5*((nodes_per_part-mean) / std_dev)**2) / (np.sqrt(2*np.pi*std_dev))
    penalty_func = tf.math.reduce_sum(penalty_func)

    return penalty_func

def diff_kemeny_loss(node_probs, mfpt, peq, penalty_func=gaussian_penalty, penalty_kwargs={}):
    """
    Computes the difference between the original and the coarse-grained Kemeny constant.

    Notes
    -----
    Equation (82)
    Kells, A.; Koskin, V.; Rosta, E.; Annibale, A.
    Correlation Functions, Mean First Passage Times, and the Kemeny Constant.
    J. Chem. Phys. 2020, 152 (10), 104108.
    https://doi.org/10.1063/1.5143504.

    Parameters
    ----------
    node_probs : torch.Tensor(n_nodes, n_partitions)
        Node probabilities.
    mfpt : torch.Tensor(n_nodes, n_nodes)
        Mean first passage time matrix.
    peq : torch.Tensor(n_nodes)
        Eigenvector with stationary distribution.
    penalty_func : function
        Penalty function.
    penalty_kwargs : dict
        Kwargs of the penalty function.

    Returns
    -------
    loss_function : torch.tensor(1)
        Loss function value.
    """
    # Calculate S matrix using soft max
    S = calc_assign(node_probs)

    # Continuous S matrix
    #S = node_probs

    # Check if there are any columns with zero
    non_empty_mask = tf.greater_equal(tf.reduce_sum(tf.abs(S), axis=0), 1e-8)
    # non_empty_mask = S.abs().sum(dim=0).ge(1e-8)
    S = tf.boolean_mask(S, non_empty_mask, axis=1)

    # Calculate the normalized populations of each cluster
    # \sum_i \pi_i S_{iK}
    PP = S * peq
    PEQ = tf.math.reduce_sum(PP, axis=0)

    # \sum_{j,i} \pi_j S_{jK} t_{ji} \pi_i S_{iK}
    H = linalg.diag_part(tf.transpose(PP) @ mfpt @ PP)
    H = tf.expand_dims(H, 1)

    # Weighted sum
    # \sum_{K=1}^{m} \frac{1}{\sum_i \pi_i S_{iK}} \sum_{j,i} \pi_j S_{jK} t_{ji} \pi_i S_{iK}
    diff_kemeny_constant = (1. / tf.expand_dims(PEQ, 0)) @ H

    if penalty_func is not None:
        penalty = gaussian_penalty(node_probs)
        loss_function = diff_kemeny_constant + penalty
    else:
        loss_function = diff_kemeny_constant

    return loss_function




def calc_assign(node_probs):
    """
    Calculate the continuous cluster assignment (S) matrix.

    Parameters
    ----------
    node_probs : torch.Tensor(n_nodes, n_partitions)
        Probability matrix returned by the GAP model.

    Returns
    -------
    S : torch.Tensor(n_nodes, n_partitions)
        Cluster assignment.
    """
    exp_factor = 250
    S_exp = tf.math.exp(exp_factor*node_probs)
    S = S_exp / tf.math.reduce_sum(S_exp, axis = 1, keepdims=True)

    # Simply return the node_probs matrix
    #S = node_probs

    return S


def calc_assign_argmax(node_probs):
    """
    Calculate the discreet cluster assignment (S) matrix.

    Parameters
    ----------
    node_probs : tf.Tensor(n_nodes, n_partitions)

    Returns
    -------
    S : tf.Tensor(n_nodes, n_partitions)
        Cluster assignment.
    """
    # Returns Tensor(n_nodes) with the index of the max along the n_partitions axis
    partition_idx = tf.argmax(node_probs, axis=1)

    # Calculate cluster assignment matrix (S)
    S = tf.one_hot(partition_idx, depth=node_probs.shape[1], axis=1)


    return S


class DiffKemenyLoss(tf.keras.losses.Loss):
  
  def __init__(self, 
               mfpt, 
               peq, 
               penalty_func):
    
    super().__init__()
    self.mfpt = mfpt
    self.peq = peq
    self.penalty_func = penalty_func

    

  def call(self, y_true, y_pred):
    print(y_pred.shape)
    y_pred = tf.cast(y_pred, dtype=tf.float64)
    print(y_pred.shape)
    print(y_true.shape)
    return diff_kemeny_loss(y_pred, self.mfpt, self.peq, self.penalty_func)