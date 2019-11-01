import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from sklearn.decomposition import PCA
from copy import deepcopy

# fileName = "cho.txt"
fileName = sys.argv[1]
K = int(sys.argv[2])

df = pds.read_csv(
    filepath_or_buffer=fileName,
    header=None,
    sep="\t")
dim = df.shape[1]

# the amount of attributes of data
num_dim = dim - 2

#  the amount of data
num_data = df.shape[0]

# dataframe only contains the value of attributes of data
x_df = df.iloc[:, 2:dim]
x_arr = x_df.values

# groundTruth of dataset
groundTruth_arr = df.iloc[:, 1].values

# Initialize the number of clusters
uniq_classes = np.unique(groundTruth_arr)
# K = len(uniq_classes)

# pick 5 random points as centroids
Centroids = (x_df.sample(n=K)).values
# print(Centroids)

# Initialize the clusters array
clusters = np.zeros(len(x_arr))
# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# store the old  To store the value of centroids when it updates
Centroids_old = np.zeros(Centroids.shape)

# the different between old Centroids and new Centroids
diff = dist(Centroids, Centroids_old, None)


while diff != 0:
    # Assigning each value to its closest cluster
    for i in range(len(x_arr)):
        distances = dist(x_arr[i],Centroids)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    Centroids_old = deepcopy(Centroids)
    # upgrade Centroids
    for i in range(K):
        points = [x_arr[j] for j in range(len(x_arr)) if clusters[j] == i]
        Centroids[i] = np.mean(points, axis=0)
    diff = dist(Centroids, Centroids_old, None)

clusters = clusters+1
# print(Centroids)
# print(clusters)


def externalIndex(results, groundtruth):
    result = results.reshape(1, -1)
    gt = groundtruth.reshape(1, -1)
    res_mat = result == result.T
    gt_mat = gt == gt.T
    rand_mat = (res_mat == gt_mat).astype(int)
    rand_index = rand_mat.sum()*1.0/rand_mat.size
    jaccard_coef = (res_mat & gt_mat).astype(int).sum()*1.0/(res_mat | gt_mat).astype(int).sum()
    return rand_index, jaccard_coef


randIndex, jaccardCoef = externalIndex(clusters, groundTruth_arr)
print("Rand Index: " + str(randIndex))
print("Jaccard Coefficient: " + str(jaccardCoef))


# plot data by pca
def pca_plot(data, lbl):
    if lbl == "groundTruth":
        label = groundTruth_arr
    elif lbl == "clusters":
        label = clusters
    else:
        print("Invalid label input.")
    pca_data = PCA(n_components=2)
    principalComponents_data = pca_data.fit_transform(data)
    principal_data_Df = pds.DataFrame(data=principalComponents_data, columns=['pc_1', 'pc_2'])

    #   visualied the data
    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('1st Principal Component', fontsize=20)
    plt.ylabel('2nd Principal Component', fontsize=20)
    plt.title("Visualization with PCA ", fontsize=20)

    targets = list(range(1, K+1))
    # colors = ['r', 'g', 'b', 'y', 'c']
    # for target, color in zip(targets, colors):
    #     indicesToKeep = label == target
    #     plt.scatter(principal_data_Df.loc[indicesToKeep, 'pc_1']
    #                 , principal_data_Df.loc[indicesToKeep, 'pc_2'], c=color, s=50)

    for target in targets:
        indicesToKeep = [idx for idx in range(len(label)) if label[idx] == target]
        plt.scatter(principal_data_Df.loc[indicesToKeep, 'pc_1'],
                    principal_data_Df.loc[indicesToKeep, 'pc_2'], s=50)

    plt.legend(targets, prop={'size': 15})
    plt.show()
    # plt.savefig("plot/"+lbl+".png")


pca_plot(x_arr, "groundTruth")
pca_plot(x_arr, "clusters")
