# library imports
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.neighbors import BallTree
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


class GLPE:
    """
    A nonlinear dimension reduction method with
     both distance and neighborhood preservation.
    """

    def __init__(self):
        pass

    # CONSTS #
    _DEFAULT_METRIC_NAME = "minkowski"
    # CONSTS #

    @staticmethod
    def run(data: np.ndarray,
            target_dim: int,
            alpha: float = 0.5,
            k: int = 3,
            metric=None):
        """
        :param data: a dataset as a Pandas' DataFrame
        :param target_dim: is the target dimension of the reduction
        :param alpha: is the balance factor between distance preserving and neighbor preserving
        :param k: is the neighbor parameter and control how many data points are considered neighbor of each data point
        :param metric: is a metric function to compute the distance between data points. If not provided, using the class' default value
        :return: a dimension reduction of the dataset into smaller one that preserve distance and neighbor
        """
        # step 0 - make sure the input data is valid
        data = GLPE._validate_inputs(data=data,
                                     k=k,
                                     alpha=alpha,
                                     target_dim=target_dim)
        # step 1 - get all the neighbors lists of the data points
        knn = NearestNeighbors(n_neighbors=k,
                               leaf_size=1,
                               metric=metric if metric is not None else GLPE._DEFAULT_METRIC_NAME)
        knn.fit(data)
        neighbors_list = [knn.kneighbors(row,
                                         return_distance=False)
                          for row in data]
        # step 2 - Get kernel matrix 'K' which preserves neighborhood relationships
        # Determine the neighborhood matrix 'W' which minimizes the cost function ||x_i - \sum_j w_{ij}x_j||^2
        # then get the kernel by K = largest_eigen_value_of_M * I - M where M = (1-W)^T(1-W)
        w = np.asarray([GLPE._find_neighborhood_decomposition(xi=data[row_index, :],
                                                              xj=data[neighbors, :],
                                                              n=data.shape[0])
                        for row_index, neighbors in enumerate(neighbors_list)])
        sub_m = np.subtract(np.identity(w.shape[0]))
        m = np.multiply(np.transpose(sub_m), sub_m)
        largest_eigenvalue = eigh(m,
                                  eigvals_only=True,
                                  subset_by_index=[0, 0])
        k = np.subtract(np.multiply(largest_eigenvalue, np.identity(m.shape[0])), m)
        # step 3 - Get kernel matrix 'T' which preserves global distances.
        # Compute the shortest path distances between any two points and write it in matrix D
        # then get the kernel T by T = −HSH/2, where H is the centralized matrix and
        # S is the matrix whose elements are the squared elements of matrix D0
        """
        d_method = BallTree(data, leaf_size=1, metric=metric if metric is not None else GLPE._DEFAULT_METRIC_NAME)
        d, _ = data.apply(lambda row: d_method.query(row, k=data.shape[1], return_distance=True))
        """
        # TODO: check this is the right implementation or the second option for 'd'
        d = pairwise_distances(data, metric=metric if metric is not None else GLPE._DEFAULT_METRIC_NAME)
        h = np.subtract(np.identity(d.shape[0]), np.multiply(1/d.shape[0], np.ones(d.shape[0])))
        s = np.multiply(d, d)
        t = np.multiply(np.multiply(np.multiply(-0.5, h), s), h)
        # step 4 - Compute the low dimensional output 'Y'. Making eigendecomposition of the kernel matrix
        # (1-alpha)K' + alpha T' where K' and T' are kernel matrix K and T after normalization.
        # Then the matrix 'Y' is the output of the algorithm
        k_tag = GLPE._normalize_matrix(matrix=k)
        t_tag = GLPE._normalize_matrix(matrix=t)
        kernel_matrix = np.add(np.multiply(1-alpha, k_tag), np.multiply(alpha, t_tag))
        eigen_values, eigen_vectors = np.linalg.eig(kernel_matrix)
        eigen_pairs = [(eigen_values[index], eigen_vectors[index]) for index in range(len(eigen_values))]
        eigen_pairs = sorted(eigen_pairs, key=lambda x: x[0], reverse=True)
        eigen_values = [val[0] for val in eigen_pairs]
        eigen_vectors = [val[1] for val in eigen_pairs]
        return np.transpose(np.asarray([np.sqrt(eigen_values[index]) * eigen_vectors[index] for index in range(len(eigen_values))]))[:, :target_dim]

    @staticmethod
    def _validate_inputs(data: np.ndarray,
                         k: int,
                         alpha: float,
                         target_dim: int):
        """ This method make sure the input data is valid and raise and error if not """
        if not (isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame)):
            raise ValueError("The value of argument 'data' must be a numpy.ndarray or pandas.DataFrame")
        if not (isinstance(k, int) and 0 < k < data.shape[1]):
            raise ValueError("The value of argument 'k' must be a positive integer smaller than the number of data points")
        if not (isinstance(alpha, float) and 0 <= k <= 1):
            raise ValueError("The value of argument 'alpha' must be a positive float number between 0 abd 1")
        if not (isinstance(target_dim, int) and 1 < k < data.shape[2]):
            raise ValueError("The value of argument 'target_dim' must be a positive integer smaller than the number of original number of features")

        # change format to data is needed
        if isinstance(data, pd.DataFrame):
            data = data.values
        return data

    @staticmethod
    def _find_neighborhood_decomposition(xi: np.ndarray,
                                         xj: np.ndarray,
                                         xj_index: np.ndarray,
                                         n: int):
        """ This function finds the reconstruction values of xi from a list of values xj and fill in a vector of size n"""
        # TODO: think how to implement it
        values = []

        # full in the whole size vector
        answer = np.zeros((1, n))
        for value_index, index in enumerate(xj_index):
            answer[index] = values[value_index]
        return answer

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray):
        """ L1 normalization of a matrix """
        dfmax = matrix.max()
        dfmin = matrix.max().min()
        dfdelta = dfmax - dfmin
        df = (matrix.max() - dfmin) / dfdelta
        return df
