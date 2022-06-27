# ______________________________________________________________________ #
import pandas as pd

data = pd.read_csv("Accident-Analysis/data/driver_readied.csv")
data.pop(data.columns[0])
print(data.head())
data = data.values
#
#
# from manifold_learning.src.python.manifold_learning.se import SchroedingerEigenmaps
# mapper = SchroedingerEigenmaps(n_neighbors=20, sparse = True)
#
# data = data.values
# data_se_transformed = mapper.fit_transform(data)
# print(data_se_transformed[0])


# ______________________________________________________________________ #
# from openTSNE_custom.tsne_model.tsne import TSNE
# model = TSNE(perplexity = 30,
#              n_components = 2,
#              se_neighbors = 20,
#              random_state = 84,
#              verbose = False
#              )


# ____________________________________________________________________ #
from sklearn import datasets

iris = datasets.load_iris()
x, y = iris["data"], iris["target"]
print(type(data))
print(type(x))
#
# from openTSNE import TSNE
#
# embedding = TSNE().fit(x)


# ______________________________________________________________________ #
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization

from openTSNE_custom.examples import utils

import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# import gzip
# import pickle


x_train, x_test = train_test_split(data, test_size=.33, random_state=42)

affinities_train = affinity.PerplexityBasedNN(
    x_train,
    perplexity=10,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
    k_neighbors = 30
)

init_train = initialization.schroedinger(
    X = x_train,
    n_neighbors = 8,
    metric = 'euclidean',
    random_state = 42,
    neighbors_algorithm='brute',
    n_jobs=4,
    weight='heat',
    affinity='heat',
    gamma=1.0,
    trees=10,
    normalization='identity',  # eigenvalue tuner parameters
    norm_laplace=None,
    lap_method='sklearn',
    mu=1.0,
    potential=None,  # potential matrices parameters
    X_img=None,
    sp_neighbors=4,
    sp_affinity='angle',
    alpha=17.78,
    eta=1.0,
    beta=1.0,
    n_components=2,
    eig_solver='dense',
    eig_tol=1E-12,
    sparse=False,
    verbose=True
)
# init_train = initialization.pca(x_train, random_state=42)

embedding_train = TSNEEmbedding(
    init_train,
    affinities_train,
    negative_gradient_method="bh",
    n_jobs=8,
    verbose=True,
)

embedding_train_1 = embedding_train.optimize(n_iter=50, exaggeration=1, momentum=0.9)
print("Shape of embedding train: {}".format(embedding_train.shape))
print("Done!!")

# utils.plot(embedding_train_1, y_train)
# plt.show()

# embedding_train_2 = embedding_train_1.optimize(n_iter=500, momentum=0.8)
#
# utils.plot(embedding_train_2, y_train)
# plt.show()
#
# embedding_test = embedding_train_2.prepare_partial(
#     x_test,
#     initialization="median",
#     k=25,
#     perplexity=5,
# )
#
# utils.plot(embedding_test, y_test)
# plt.show()
#
# embedding_test_1 = embedding_test.optimize(n_iter=250, learning_rate=0.1, momentum=0.8)
#
# utils.plot(embedding_test_1, y_test)
# plt.show()
#
# fig, ax = plt.subplots(figsize=(8, 8))
# utils.plot(embedding_train_2, y_train, alpha=0.25, ax=ax)
# plt.show()
#
# utils.plot(embedding_test_1, y_test, alpha=0.75, ax=ax)
# plt.show()
