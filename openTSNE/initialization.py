import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, SpectralEmbedding, LocallyLinearEmbedding, MDS
from sklearn.utils import check_random_state
from manifold_learning.src.python.manifold_learning.se import SchroedingerEigenmaps

from openTSNE import utils


def rescale(x, inplace=False):
    """Rescale an embedding so optimization will not have convergence issues.

    Parameters
    ----------
    x: np.ndarray
    inplace: bool

    Returns
    -------
    np.ndarray
        A scaled-down version of ``x``.

    """
    if not inplace:
        x = np.array(x, copy=True)

    x /= np.std(x[:, 0]) * 10000

    return x


def random(n_samples, n_components=2, random_state=None, verbose=False):
    """Initialize an embedding using samples from an isotropic Gaussian.

    Parameters
    ----------
    n_samples: Union[int, np.ndarray]
        The number of samples. Also accepts a data matrix.

    n_components: int
        The dimension of the embedding space.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    verbose: bool

    Returns
    -------
    initialization: np.ndarray

    """
    random_state = check_random_state(random_state)
    if isinstance(n_samples, np.ndarray):
        n_samples = n_samples.shape[0]
    embedding = random_state.normal(0, 1e-4, (n_samples, n_components))
    return np.ascontiguousarray(embedding)


def pca(X, n_components=2, svd_solver="auto", random_state=None, verbose=False):
    """Initialize an embedding using the top principal components.

    Parameters
    ----------
    X: np.ndarray
        The data matrix.

    n_components: int
        The dimension of the embedding space.

    svd_solver: str
        See sklearn.decomposition.PCA documentation.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    verbose: bool

    Returns
    -------
    initialization: np.ndarray

    """
    timer = utils.Timer("Calculating PCA-based initialization...", verbose)
    timer.__enter__()

    pca_ = PCA(
        n_components=n_components, svd_solver=svd_solver, random_state=random_state
    )
    embedding = pca_.fit_transform(X)
    rescale(embedding, inplace=True)

    timer.__exit__()

    return np.ascontiguousarray(embedding)


def lle(X, n_neighbors=100, n_components=2, method = 'standard', eig_solver="auto", random_state=None, verbose=False):
    """Initialize an embedding using the top principal components.

    Parameters
    ----------
    X: np.ndarray
        The data matrix.

    n_components: int
        The dimension of the embedding space.

    eig_solver: str
        Eigen Value solver.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    verbose: bool

    Returns
    -------
    initialization: np.ndarray

    """
    timer = utils.Timer("Calculating PCA-based initialization...", verbose)
    timer.__enter__()

    lle_ = LocallyLinearEmbedding(
        n_neighbors=n_neighbors,
        n_components=n_components,
        eigen_solver=eig_solver,
        random_state=random_state,
        method=method
    )
    embedding = lle_.fit_transform(X)
    rescale(embedding, inplace=True)

    timer.__exit__()

    return np.ascontiguousarray(embedding)


def le(X, n_components=2, eig_solver="auto", random_state=None, verbose=False):
    """Initialize an embedding using the top principal components.

    Parameters
    ----------
    X: np.ndarray
        The data matrix.

    n_components: int
        The dimension of the embedding space.

    eig_solver: str
        Eigen Value Decomposition colver.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    verbose: bool

    Returns
    -------
    initialization: np.ndarray

    """
    timer = utils.Timer("Calculating PCA-based initialization...", verbose)
    timer.__enter__()

    le_ = SpectralEmbedding(n_components = n_components,
                            random_state = random_state,
                            n_neighbors = 100)
    embedding = le_.fit_transform(X)
    rescale(embedding, inplace=True)

    timer.__exit__()

    return np.ascontiguousarray(embedding)


def mds(X, n_components=2, random_state=None, verbose=False):
    """Initialize an embedding using the top principal components.

    Parameters
    ----------
    X: np.ndarray
        The data matrix.

    n_components: int
        The dimension of the embedding space.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    verbose: bool

    Returns
    -------
    initialization: np.ndarray

    """
    timer = utils.Timer("Calculating PCA-based initialization...", verbose)
    timer.__enter__()

    mds_ = MDS(n_components = n_components,
               random_state = random_state)
    embedding = mds_.fit_transform(X)
    rescale(embedding, inplace=True)

    timer.__exit__()

    return np.ascontiguousarray(embedding)


def schroedinger(
        X,
        n_neighbors = 2,
        metric = 'euclidean',
        random_state = None,
        neighbors_algorithm='brute',
        n_jobs=4,
        weight='heat',
        affinity='heat',
        gamma=1.0,
        trees=10,
        normalization=None,  # eigenvalue tuner parameters
        norm_laplace=None,
        lap_method='sklearn',
        mu=1.0,
        potential=None,  # potential matrices parameters
        X_img=None,
        sp_neighbors=4,
        sp_affinity='heat',
        alpha=17.78,
        eta=1.0,
        beta=1.0,
        n_components=2,
        eig_solver='dense',
        eig_tol=1E-12,
        sparse=False,
        verbose=True
):
    """Initialize an embedding using the top principal components.

        Parameters
        ----------
        X: np.ndarray
            The data matrix.

        n_neighbors: int
            k-nn parameter.

        metric: str
            See sklearn.decomposition.PCA documentation.

        random_state: Union[int, RandomState]
            If the value is an int, random_state is the seed used by the random
            number generator. If the value is a RandomState instance, then it will
            be used as the random number generator. If the value is None, the random
            number generator is the RandomState instance used by `np.random`.

        verbose: bool

        Returns
        -------
        initialization: np.ndarray
        :param affinity:
        :param weight:
        :param n_jobs:
        :param random_state:
        :param metric:
        :param X_img:
        :param X:
        :param n_neighbors:
        :param neighbors_algorithm:

        """
    timer = utils.Timer("Calculating Schroedinger-based initialization...", verbose)
    timer.__enter__()

    se_ = SchroedingerEigenmaps(
        n_neighbors = n_neighbors,
        metric = metric,
        random_state = random_state,
        neighbors_algorithm='brute',
        n_jobs=4,
        weight='heat',
        affinity='heat',
        gamma=1.0,
        trees=10,
        normalization=None,  # eigenvalue tuner parameters
        norm_laplace=None,
        lap_method='sklearn',
        mu=1.0,
        potential=None,  # potential matrices parameters
        X_img=None,
        sp_neighbors=4,
        sp_affinity='heat',
        alpha=17.78,
        eta=1.0,
        beta=1.0,
        n_components=2,
        eig_solver='dense',
        eig_tol=1E-12,
        sparse=False,
    )
    embedding = se_.fit_transform(X)
    rescale(embedding, inplace=True)

    timer.__exit__()
    return np.ascontiguousarray(embedding)


def spectral(A, n_components=2, tol=1e-4, max_iter=None, random_state=None, verbose=False):
    """Initialize an embedding using the spectral embedding of the KNN graph.

    Specifically, we initialize data points by computing the diffusion map on
    the random walk transition matrix of the weighted graph given by the affiniy
    matrix.

    Parameters
    ----------
    A: Union[sp.csr_matrix, sp.csc_matrix, ...]
        The graph adjacency matrix.

    n_components: int
        The dimension of the embedding space.

    tol: float
        See scipy.sparse.linalg.eigsh documentation.

    max_iter: float
        See scipy.sparse.linalg.eigsh documentation.

    random_state: Any
        Unused, but kept for consistency between initialization schemes.

    verbose: bool

    Returns
    -------
    initialization: np.ndarray

    """
    if A.ndim != 2:
        raise ValueError("The graph adjacency matrix must be a 2-dimensional matrix.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("The graph adjacency matrix must be a square matrix.")

    timer = utils.Timer("Calculating spectral initialization...", verbose)
    timer.__enter__()

    D = sp.diags(np.ravel(np.sum(A, axis=1)))

    # Find leading eigenvectors
    k = n_components + 1
    v0 = np.ones(A.shape[0]) / np.sqrt(A.shape[0])
    eigvals, eigvecs = sp.linalg.eigsh(
        A, M=D, k=k, tol=tol, maxiter=max_iter, which="LM", v0=v0
    )
    # Sort the eigenvalues in decreasing order
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # In diffusion maps, we multiply the eigenvectors by their eigenvalues
    eigvecs *= eigvals

    # Drop the leading eigenvector
    embedding = eigvecs[:, 1:]

    rescale(embedding, inplace=True)

    timer.__exit__()

    return embedding


def weighted_mean(X, embedding, neighbors, distances, verbose=False):
    """Initialize points onto an existing embedding by placing them in the
    weighted mean position of their nearest neighbors on the reference embedding.

    Parameters
    ----------
    X: np.ndarray
    embedding: TSNEEmbedding
    neighbors: np.ndarray
    distances: np.ndarray
    verbose: bool

    Returns
    -------
    np.ndarray

    """
    n_samples = X.shape[0]
    n_components = embedding.shape[1]

    with utils.Timer("Calculating weighted-mean initialization...", verbose):
        partial_embedding = np.zeros((n_samples, n_components), order="C")
        for i in range(n_samples):
            partial_embedding[i] = np.average(
                embedding[neighbors[i]], axis=0, weights=distances[i]
            )

    return partial_embedding


def median(embedding, neighbors, verbose=False):
    """Initialize points onto an existing embedding by placing them in the
    median position of their nearest neighbors on the reference embedding.

    Parameters
    ----------
    embedding: TSNEEmbedding
    neighbors: np.ndarray
    verbose: bool

    Returns
    -------
    np.ndarray

    """
    with utils.Timer("Calculating meadian initialization...", verbose):
        embedding = np.median(embedding[neighbors], axis=1)
    return np.ascontiguousarray(embedding)
