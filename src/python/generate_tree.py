""" Generate graphs for embedding in Hyperbolic space """

import numpy as np
from scipy.sparse import csr_matrix
import utils


def create_adj_mat(outfolder, branching_fac, levels):
    """
    create scipy adjacency matrices for complete trees
    :param outpath: folder to write the tree to
    :param branching_fac: number of children of each vertex
    :param levels: depth of the tree
    :return:
    """
    A = generate_tree(branching_fac, levels)
    adj_mat = csr_matrix(A)
    outpath = '{}/X_z{}_l{}.p'.format(outfolder, branching_fac, levels)
    utils.pickle_sparse(adj_mat, outpath)



def generate_tree(z, l):
    """ Generate z-tree with l layers
    Should have number of nodes N=1 + z (((z-1)^L - 1) / (z-2))
    For z > 2 """
    assert z > 2
    assert l > 1
    N = 1 + z * (((z - 1) ** l) - 1) / (z - 2)
    A = np.zeros((N, N))
    i = 0
    j = 1
    while j < N:
        # add edge from i to j
        if i == 0:
            # root is special case
            num_neighbours = z
        else:
            num_neighbours = z - 1
        for p in range(num_neighbours):
            A[i, j] = 1
            j += 1
        i += 1
    return A + A.T


if __name__ == "__main__":
    outfolder = '../../local_resources/simulated_trees'
    branching_factor = 4
    levels = 5
    create_adj_mat(outfolder, branching_factor, levels)
