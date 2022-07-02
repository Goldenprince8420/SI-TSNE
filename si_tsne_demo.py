# ______________________________________________________________________ #
import pandas as pd
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
import scipy.io
from scipy.io import loadmat
import json

from openTSNE_custom.examples import utils

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.decomposition import PCA
from manifold_learning.src.python.manifold_learning import SchroedingerEigenmaps
from sklearn.cluster import KMeans
from sklearn import metrics


import matplotlib.pyplot as plt
# ________________________________________________________________________ #

data = pd.read_csv("Accident-Analysis/data/mixed_data.csv")
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

# iris = datasets.load_iris()
# x, y = iris["data"], iris["target"]
# print(type(data))
# print(type(x))
#
# from openTSNE import TSNE
#
# embedding = TSNE().fit(x)


# ______________________________________________________________________ #


# import gzip
# import pickle


# x_train, x_test = train_test_split(data, test_size=.33, random_state=42)
#
# affinities_train = affinity.PerplexityBasedNN(
#     x_train,
#     perplexity=10,
#     metric="euclidean",
#     n_jobs=8,
#     random_state=42,
#     verbose=True,
#     k_neighbors = 30
# )
#
# init_train = initialization.schroedinger(
#     X = x_train,
#     n_neighbors = 8,
#     metric = 'euclidean',
#     random_state = 42,
#     neighbors_algorithm='brute',
#     n_jobs=4,
#     weight='heat',
#     affinity='heat',
#     gamma=1.0,
#     trees=10,
#     normalization='identity',  # eigenvalue tuner parameters
#     norm_laplace=None,
#     lap_method='sklearn',
#     mu=1.0,
#     potential=None,  # potential matrices parameters
#     X_img=None,
#     sp_neighbors=4,
#     sp_affinity='angle',
#     alpha=17.78,
#     eta=1.0,
#     beta=1.0,
#     n_components=2,
#     eig_solver='dense',
#     eig_tol=1E-12,
#     sparse=False,
#     verbose=True
# )
# # init_train = initialization.pca(x_train, random_state=42)
#
# embedding_train = TSNEEmbedding(
#     init_train,
#     affinities_train,
#     negative_gradient_method="bh",
#     n_jobs=8,
#     verbose=True,
# )
#
# embedding_train_1 = embedding_train.optimize(n_iter=50, exaggeration=1, momentum=0.9)
# print("Shape of embedding train: {}".format(embedding_train.shape))
# print("Done!!")

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

# ________________________________________________________________ #
# pca_reducer = PCA(n_components=2, svd_solver = "full", random_state = 42)
# data_pca_reduced = pca_reducer.fit_transform(data)
# print(data_pca_reduced.shape)
#
#
# se_reducer = SchroedingerEigenmaps()
# data_se_reduced = se_reducer.fit_transform(data)
# print(data_se_reduced.shape)
#
# # x_train, x_test = train_test_split(data, test_size=.33, random_state=42)
# x_train = data
#
# affinities_train = affinity.PerplexityBasedNN(
#     x_train,
#     perplexity=2500,
#     metric="euclidean",
#     n_jobs=8,
#     random_state=42,
#     verbose=True,
#     k_neighbors = 5000
# )
#
# init_train_se = initialization.schroedinger(x_train, n_neighbors=100, random_state=42)
# init_train = initialization.pca(init_train_se, random_state = 42, svd_solver = "full")
#
#
# embedding_train = TSNEEmbedding(
#     init_train,
#     affinities_train,
#     negative_gradient_method="bh",
#     n_jobs=8,
#     verbose=True
# )
#
# embedding_train_1 = embedding_train.optimize(n_iter=1000, exaggeration=1.0, momentum=0.9)
# print("Shape of embedding train: {}".format(embedding_train.shape))
# print("Done!!")
#
# plt.scatter(embedding_train_1.T[0], embedding_train_1.T[1])
# plt.show()


# init_test = initialization.schroedinger(x_test, n_neighbors=100, random_state=42)
# init_test = initialization.pca(init_test, random_state = 42, svd_solver = "full")
# embedding_test = embedding_train_1.prepare_partial(
#     init_test.T,
#     initialization="median",
#     k=1500,
#     perplexity=570
# )
#
# plt.scatter(embedding_test.T[0], embedding_test.T[1])
# plt.show()


# kmeans_cluster_train = KMeans(n_clusters = 3, random_state = 42)
# cluster_res_ = kmeans_cluster_train.fit(embedding_train_1)
# labels_true = cluster_res_.labels_
# cluster_center = cluster_res_.cluster_centers_

print("______________________________________________________________________________________________")
# print("Validation:")
# kmeans_cluster_test = KMeans(n_clusters = 3, random_state = 42)
# cluster_res_test = kmeans_cluster_test.fit(embedding_train_1)
# labels_pred = cluster_res_test.labels_
# cluster_center_pred = cluster_res_test.cluster_centers_

print("______________________________________________________________________________________________")
# print("Silhoutte Index: {}".format(metrics.silhouette_score(embedding_train_1, labels_true)))
# print("Davies Bouldin Score: {}".format(metrics.davies_bouldin_score(embedding_train_1, labels_true)))
# print("Silhoutte Index Test: {}".format(metrics.silhouette_score(embedding_test, labels_pred)))
# print("Davies Bouldin Score Test: {}".format(metrics.davies_bouldin_score(embedding_test, labels_pred)))

# print("___________________________________________________________________________________________________")
# print("Homogeneity Score: {}".format(metrics.homogeneity_score(labels_true, labels_pred)))
# print("Completeness Score: {}".format(metrics.completeness_score(labels_true, labels_pred)))
# print("V Measure Score: {}".format(metrics.v_measure_score(labels_true, labels_pred)))
# print("Adjusted Mutual Info Score: {}".format(metrics.adjusted_mutual_info_score(labels_true, labels_pred)))
# print("Adjusted Rand Score: {}".format(metrics.adjusted_rand_score(labels_true, labels_pred)))
# print("____________________________________________________________________________________________________")


# data_new = pd.read_csv("Accident-Analysis/data/mixed_data.csv")
# data_new['cluster_labels'] = labels_true
#
# data_dic = {
#     "data_initial": data,
#     "init_train_se": init_train_se,
#     "init_train": init_train,
#     "embedding_initial": embedding_train,
#     "embedded_final": embedding_train_1,
#     "cluster_labels": cluster_res_,
#     "cluster_centroids": cluster_center,
#     "final_data": data_new.values
# }
#
# scipy.io.savemat("data/results.mat", data_dic)
# data_new.to_csv("data/mixed_data_labeled.csv")


# ___________________________________________________________________________________ #
from scipy.io import loadmat
# print("Performance Analysis...")
# workspace_path = 'Accident-Analysis/data/results.mat'
# workspace = loadmat(workspace_path)
# print(workspace.keys())
# embedding_initial = workspace['embedding_initial']
# embedding_final = workspace['embedded_final']
#
# plt.scatter(embedding_final.T[0], embedding_final.T[1])
# plt.show()
#
#
# def visualize_clusters():
#     reduced_data = embedding_final
#     kmeans_cluster_train = KMeans(n_clusters=4, random_state=42)
#     cluster_res_ = kmeans_cluster_train.fit(embedding_final)
#     labels_true = cluster_res_.labels_
#     utils.plot(reduced_data, labels_true, draw_centers = True)
#     plt.show()
#
#
# visualize_clusters()
#
#
# def k_means_performance():
#     performance_dict = {}
#     for n_clusters in range(2, 7):
#         performance_dict[str(n_clusters)] = {}
#         kmeans_cluster_train = KMeans(n_clusters = n_clusters, random_state = 42)
#         cluster_res_ = kmeans_cluster_train.fit(embedding_final)
#         labels_true = cluster_res_.labels_
#
#         print("Silhoutte Index: {}".format(metrics.silhouette_score(embedding_final, labels_true)))
#         print("Davies Bouldin Score: {}".format(metrics.davies_bouldin_score(embedding_final, labels_true)))
#
#         performance_dict[str(n_clusters)]['SI_index'] = metrics.silhouette_score(embedding_final, labels_true)
#         performance_dict[str(n_clusters)]['DBI_Score'] = metrics.davies_bouldin_score(embedding_final, labels_true)
#
#     performance_space = json.dumps(performance_dict, indent = 4)
#     print(performance_space)
#
#     with open('Accident-Analysis/.idea/kmeans_performance.json', 'w') as f:
#         f.write(performance_space)
#
#
# k_means_performance()




# _________________________________________________________________________________________________________________ #
## Initialization Comparison
# 1. PCA

x_train = data.copy()
affinities_train = affinity.PerplexityBasedNN(
    x_train,
    perplexity=2500,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
    k_neighbors = 5000
)

init_train_pca = initialization.pca(x_train, random_state=42)
# init_train = initialization.pca(init_train_se, random_state = 42, svd_solver = "full")
init_train_le = initialization.le(x_train, random_state = 42)

init_train_se = initialization.schroedinger(x_train, n_neighbors=100, random_state=42)
init_train_se = initialization.pca(init_train_se, random_state = 42, svd_solver = "full")

init_train_mds = initialization.mds(x_train, random_state = 42)

init_train_ltsa = initialization.lle(x_train, method = 'ltsa', random_state = 42)

init_train_hlle = initialization.lle(x_train, method = 'hessian', random_state = 42)

init_train_lle = initialization.lle(x_train, random_state = 42)

init_embed = {
    "pca": init_train_pca,
    "le": init_train_le,
    "se": init_train_se,
    "hlle": init_train_hlle,
    "lle": init_train_lle,
    "mds": init_train_mds,
}

scipy.io.savemat("Accident-Analysis/data/initial_embed.mat", init_embed)



# embedding_train = TSNEEmbedding(
#     init_train_pca,
#     affinities_train,
#     negative_gradient_method="bh",
#     n_jobs=8,
#     verbose=True
# )