"""
Efficient random walk generation
"""
import numpy as np
from numpy.random import randint
from datetime import datetime
import utils
import pandas as pd
from gensim.models import Word2Vec


class Graph:
    """
    A binary graph
    """

    def __init__(self, adj, shuffle_walks=True):
        self.adj = adj.tocsr()
        self.deg = np.array(adj.sum(axis=1), dtype=int).squeeze()
        self.n_vertices = self.deg.shape[0]
        self.edges = np.zeros(shape=(self.n_vertices, max(self.deg)), dtype=int)
        self.shuffle_walks = shuffle_walks

    def build_edge_array(self):
        """
        construct an array of edges. Instead of a binary representation each row contains the index of vertices reached
        by outgoing edges padded by zeros to the right
        0 0 1
        0 0 1
        1 1 0

        becomes

        2 0
        2 0
        0 1

        :return:
        """
        for row_idx in range(self.n_vertices):
            # get the indices of the vertices that this vertex connects to
            z = self.adj[row_idx, :].nonzero()[1]
            # add these the left hand side of the edge array
            self.edges[row_idx, 0:len(z)] = z

    def sample_next_vertices(self, current_vertices, degs):
        """
        get the next set of vertices for the random walks
        :return: next_vertices np.array shape = (len(current_vertices), 1)
        """
        current_degrees = degs[current_vertices]
        # sample an index into the edges array for each walk
        next_vertex_indices = np.array(map(lambda x: randint(x), current_degrees))
        return next_vertex_indices

    def initialise_walk_array(self, num_walks, walk_length):
        """
        Build an array to store the random walks with the initial starting positions in the first column. The order of
        the nodes is randomly shuffled as this is well known to speed up SGD convergence (Deepwalk: online learning of
        social representations)
        :return: A numpy array of shape = (n_vertices * num_walks, walk_length) which is all zero except for the first
        column
        """
        initial_vertices = np.arange(self.n_vertices)
        walks = np.zeros(shape=(self.n_vertices * num_walks, walk_length), dtype=int)
        walk_starts = np.tile(initial_vertices, num_walks)
        if self.shuffle_walks:  # this improves SGD convergence, but is inconvenient for slicing the walks
            np.random.shuffle(walk_starts)
        walks[:, 0] = walk_starts
        return walks

    def generate_walks(self, num_walks, walk_length):
        """
        generate random walks
        :param num_walks the number of random walks per vertex
        :param walk_length the length of each walk
        :return:
        """
        assert self.deg.min() > 0
        # degs = np.tile(self.deg, num_walks)
        # edges = np.tile(self.edges, (num_walks, 1))
        walks = self.initialise_walk_array(num_walks, walk_length)

        for walk_idx in range(walk_length - 1):
            print 'generating walk step {}'.format(walk_idx)
            # get the degree of the vertices we're starting from
            current_vertices = walks[:, walk_idx]
            # get the indices of the next vertices. This is the random bit
            next_vertex_indices = self.sample_next_vertices(current_vertices, self.deg)
            walks[:, walk_idx + 1] = self.edges[current_vertices, next_vertex_indices]
        return walks

    def learn_embeddings(self, walks, size, outpath):
        """
        learn a word2vec embedding using the gensim library.
        :param walks: A numpy array of random walks of shape (num_walks, walk_length)
        :param size: The number of dimensions in the embedding
        :param outpath: Path to write the embedding
        :returns None
        """
        # gensim needs an object that can iterate over lists of unicode strings. Not ideal for this application really.
        walk_str = walks.astype(str)
        walk_list = walk_str.tolist()

        model = Word2Vec(walk_list, size=size, window=5, min_count=0, sg=1, workers=4,
                         iter=5)
        model.wv.save_word2vec_format(outpath)


def read_data(x_path, threshold):
    """
    reads the features and target variables
    :return:
    """
    X = utils.read_pickle(x_path)
    X1, cols = utils.remove_sparse_features(X, threshold=threshold)
    print X1.shape
    return X1


def scenario_generate_public_embeddings(size=128):
    inpaths = ['local_resources/blogcatalog/X.p', 'local_resources/flickr/X.p',
               'local_resources/youtube/X.p']
    outpaths = ['local_resources/blogcatalog/blogcatalog128.emd', 'local_resources/flickr/flickr128.emd',
                'local_resources/youtube/youtube128.emd']
    walkpaths = ['local_resources/blogcatalog/walks.csv', 'local_resources/flickr/walks.csv',
                 'local_resources/youtube/walks.csv']

    for paths in zip(inpaths, outpaths, walkpaths):
        print 'reading data'
        x = utils.read_pickle(paths[0])
        g = Graph(x)
        print 'building edges'
        g.build_edge_array()
        print 'generating walks'
        walks = g.generate_walks(10, 80)
        g.learn_embeddings(walks, size, paths[1])
        print walks.shape
        df = pd.DataFrame(walks)
        df.to_csv(paths[2], index=False, header=None)


def generate_blogcatalog_sample(size=2):
    xpath = '../../local_resources/blogcatalog_121_sample/X.p'
    ypath = '../../local_resources/blogcatalog_121_sample/y.p'
    emdpath = '../../local_resources/blogcatalog_121_sample/blogcatalog' + str(size) + '.emd'
    walkpath = '../../local_resources/blogcatalog_121_sample/walks_n1_l10.csv'
    x = utils.read_pickle(xpath)
    g = Graph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(1, 10)
    g.learn_embeddings(walks, size, emdpath)
    print walks.shape
    df = pd.DataFrame(walks)
    df.to_csv(walkpath, index=False, header=None)


def generate_blogcatalog(size=2):
    xpath = '../../local_resources/blogcatalog/X.p'
    ypath = '../../local_resources/blogcatalog/y.p'
    emdpath = '../../local_resources/blogcatalog/blogcatalog2.emd'
    walkpath = '../../local_resources/blogcatalog/walks_n1_l10.csv'
    x = utils.read_pickle(xpath)
    g = Graph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(2, 20)
    g.learn_embeddings(walks, size, emdpath)
    print walks.shape
    df = pd.DataFrame(walks)
    df.to_csv(walkpath, index=False, header=None)


def generate_political_blogs(size=2):
    xpath = '../../local_resources/political_blogs/X.p'
    ypath = '../../local_resources/political_blogs/y.p'
    emdpath = '../../local_resources/political_blogs/political_blogs2.emd'
    walkpath = '../../local_resources/political_blogs/walks_n1_l10.csv'
    x = utils.read_pickle(xpath)
    g = Graph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(1, 10)
    g.learn_embeddings(walks, size, emdpath)
    print walks.shape
    df = pd.DataFrame(walks)
    df.to_csv(walkpath, index=False, header=None)


def generate_political_blogs_deepwalk_embeddings():
    walkpath = '../../local_resources/political_blogs/walks_n1_l10.csv'
    walks = pd.read_csv(walkpath, header=None).values
    xpath = '../../local_resources/political_blogs/X.p'
    x = utils.read_pickle(xpath)
    g = Graph(x)
    sizes = [4, 8, 16, 32, 64, 128]
    path = '../../local_resources/political_blogs/political_blogs'
    for size in sizes:
        emdpath = path + str(size) + '.emd'
        g.learn_embeddings(walks, size, emdpath)


def generate_multiple_walks_and_embeddings():
    stub = '../../local_resources/'
    names = ['karate']
    for name in names:
        ypath = stub + name + '/y.p'
        xpath = stub + name + '/X.p'
        walkpath = stub + name + '/walks_n1_l10.csv'
        path = stub + name + '/' + name
        x = utils.read_pickle(xpath)
        g = Graph(x)
        print 'building edges'
        g.build_edge_array()
        print 'generating walks'
        walks = g.generate_walks(1, 10)
        print walks.shape
        df = pd.DataFrame(walks)
        df.to_csv(walkpath, index=False, header=None)
        sizes = [2, 4, 8, 16, 32, 64, 128]
        for size in sizes:
            emdpath = path + str(size) + '.emd'
            g.learn_embeddings(walks, size, emdpath)


def karate_deepwalk_gridsearch():
    """
    A function to double check the results. They seem to good and I'm worried that I messed up the DeepWalk bit somehow.
    There should be at least some configurations where DeepWalk does well.
    :return:
    """
    stub = '../../local_resources/karate'
    ypath = stub + '/y.p'
    xpath = stub + '/X.p'
    walk_lengths = [10, 40, 70]
    n_walks = [1, 5, 10]
    sizes = [2, 8, 32, 128]

    # walkpath = stub + name + '/walks_n1_l10.csv'
    x, y = utils.read_data(xpath, ypath, 0)
    g = Graph(x, shuffle_walks=False)
    n_vertices = len(y)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    all_walks = g.generate_walks(10, 70)
    print all_walks.shape

    for walk_length in walk_lengths:
        walks = all_walks[:, 0:walk_length]
        for n in n_walks:
            walks = walks[0:n_vertices * n, :]
            for size in sizes:
                emdpath = '{}/gridsearch/size_{}_nwalks_{}_walklen_{}.emd'.format(stub, size, n, walk_length)
                g.learn_embeddings(walks, size, emdpath)


def generate_simulated_tree(path, walkpath, deepwalk_path):
    x = utils.read_pickle(path)
    g = Graph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(10, 80)
    print 'generated walk of shape {}'.format(walks.shape)
    size = 2
    g.learn_embeddings(walks, size, deepwalk_path)
    df = pd.DataFrame(walks)
    df.to_csv(walkpath, index=False, header=None)


if __name__ == '__main__':
    s = datetime.now()
    path = '../../local_resources/simulated_trees/X_z4_l5.p'
    generate_simulated_tree(path)
    # karate_deepwalk_gridsearch()
    print datetime.now() - s, ' s'
