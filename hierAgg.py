import pandas as pds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
from sklearn.decomposition import PCA
from scipy.spatial import distance
from copy import deepcopy

# fileName = "iyer.txt"
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
K = len(uniq_classes)


# --------------------------------------------#
#              Helper Functions
# --------------------------------------------#

# def dist(a, b, ax=None):
#     return np.linalg.norm(a - b, axis=ax)
def dist(a, b):
    return distance.euclidean(a, b)


def dist_cluster(cluster_a, cluster_b):
    c_dist = np.inf
    pts_a = C_dict[cluster_a]   # the data points in cluster a
    pts_b = C_dict[cluster_b]   # the data points in cluster b
    for pt_a in pts_a:
        for pt_b in pts_b:
            dist_ab = D_mat_pts[pt_a][pt_b]
            if dist_ab < c_dist:
                c_dist = dist_ab
    return c_dist


def find_min_dist():
    # for c_i in range(num_data):
    #     for c_j in range(num_data):
    #         if c_i!=c_j & cluster_state[c_i] & cluster_state[c_j]:
    #             dist_ij =
    min_row_idxes = D_mat_cls.idxmin()
    min_cols = D_mat_cls.min()
    min_col_idx = min_cols.idxmin()
    min_row_idx = min_row_idxes[min_col_idx]
    c_small = min([min_row_idx, min_col_idx])
    c_large = max([min_row_idx, min_col_idx])
    return c_small, c_large


# Init dist_mat among points
D_mat_pts = np.zeros(shape=(num_data, num_data))
# Init cluster dict
C_dict = dict()
# cluster_state = [True] * num_data

for i in range(num_data):
    C_dict[i] = [i]
    for j in range(i):
        D_mat_pts[i][j] = dist(x_arr[i], x_arr[j])
        D_mat_pts[j][i] = D_mat_pts[i][j]

D_mat_pts[D_mat_pts == 0] = np.inf
D_mat_cls = deepcopy(D_mat_pts)
D_mat_cls = pd.DataFrame(D_mat_cls, index=list(range(num_data)), columns=list(range(num_data)))

for step in range(num_data-K):
    print("step: " + str(step))
    m, n = find_min_dist()  # m < n
    C_dict[m].extend(C_dict[n])
    C_dict.pop(n)
    # cluster_state[n] = False
    # Update distance mat D_mat_cls
    for i in C_dict.keys():
        if m != i:
            D_mat_cls.loc[m, i] = dist_cluster(m, i)
            D_mat_cls.loc[i, m] = D_mat_cls.loc[m, i]
    D_mat_cls.drop([n], axis=0, inplace=True)
    D_mat_cls.drop([n], axis=1, inplace=True)

clusters = np.zeros(len(x_arr))
cluster_idx = 1
for key in C_dict.keys():
    for pt in C_dict[key]:
        clusters[pt] = cluster_idx
    cluster_idx += 1


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
    plt.title("Visualization with PCA", fontsize=20)
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
    # plt.savefig("plot/" + lbl + ".png")
    plt.show()


pca_plot(x_arr, "groundTruth")
pca_plot(x_arr, "clusters")
