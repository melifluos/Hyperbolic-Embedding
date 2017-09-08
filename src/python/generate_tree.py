""" Generate graphs for embedding in Hyperbolic space """

import numpy as np
from scipy.sparse import csr_matrix
import utils
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph():
    A = generate_tree(3, 3)
    G = nx.from_numpy_matrix(A)
    pos = nx.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=False, arrows=False)
    plt.show()


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
    return outpath


def get_counts(z, l):
    """
    return the edge and vertex count
    :param z:
    :param l:
    :return:
    """
    vert_count = 1
    for layer in xrange(1, l):
        nvert = z ** layer
        vert_count += nvert
    edge_count = z * (vert_count - z ** (l - 1))
    return vert_count, edge_count


def get_vertex_count(z, l):
    sum = 1
    for layer in xrange(1, l):
        nvert = z ** layer
        sum += nvert
    return sum


def generate_tree(z, l):
    """ Generate z-tree with l layers
    Should have number of nodes N=1 + z (((z-1)^L - 1) / (z-2))
    For z > 2 """
    # assert z > 2
    # assert l > 1
    V, E = get_counts(z, l)
    A = np.zeros((V, V))
    i = 0
    j = 1
    while j < E:
        # add edge from i to j
        for p in range(z):
            A[i, j] = 1
            j += 1
        i += 1
    return A + A.T


# def generate_tree(z, l):
#     """ Generate z-tree with l layers
#     Should have number of nodes N=1 + z (((z-1)^L - 1) / (z-2))
#     For z > 2 """
#     assert z > 2
#     assert l > 1
#     nedges =
#     N = 1 + z * (((z - 1) ** l) - 1) / (z - 2)
#     A = np.zeros((N, N))
#     i = 0
#     j = 1
#     while j < N:
#         # add edge from i to j
#         if i == 0:
#             # root is special case
#             num_neighbours = z
#         else:
#             num_neighbours = z - 1
#         for p in range(num_neighbours):
#             A[i, j] = 1
#             j += 1
#         i += 1
#     return A + A.T


def generate_y(z, l):
    """
    produce the layer labels
    :param z: the branching fac
    :param l: the levels
    :return:
    """
    y = [0]
    for layer in xrange(1, l):
        nvert = z ** layer
        labels = [layer] * nvert
        y += labels
    return np.array(y)


if __name__ == "__main__":
    # print get_counts(3, 3)
    # print generate_y(3, 3)
    # plot_graph()
    outfolder = '../../local_resources/simulated_trees'
    branching_factor = 2
    levels = 2
    create_adj_mat(outfolder, branching_factor, levels)
