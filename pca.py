
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import argparse
import sys
from sklearn.preprocessing import StandardScaler







# # parser = argparse.ArgumentParser(description='Process some dataset.')
# # parser.add_argument('fileName', metavar='File Name', help='A file name for the program')
# # args = parser.parse_args()
# # fileName = args.fileName
# fileName = "cho.txt"
# df = pds.read_csv(
#     filepath_or_buffer=fileName,
#     header=None,
#     sep="\t")
# dim = df.shape[1]
# num_data = df.shape[0]
#
# # df.columns=['0', '1', '2', '3', 'Disease']
# # print(df)
#
# x = df.iloc[:, 0:dim - 1].values
# y = df.iloc[:, -1].values
# # print(x)
# # X_std = StandardScaler().fit_transform(x)
#
# mean_vec = np.mean(x, axis=0)
# # test = np.mean(x-mean_vec,axis=0)
# cov_mat = (x - mean_vec).T.dot((x - mean_vec)) / (num_data - 1)
# # cov_mat_1 = np.cov((x - mean_vec).T)
# eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
# p_vecs = np.fliplr(eig_vecs[:, -2:])
#
# pc = x.dot(p_vecs)
#
# # print('Covariance matrix \n%s' % cov_mat)
# # print('NumPy covariance matrix: \n%s' % np.cov(X_std.T))
#
# # Plot
# # Y = np.concatenate((pc, y.reshape(y.shape[0], 1)), axis=1)
# # hasharray = np.zeros([num_data, 1])
# #
# # for i in range(num_data):
# #     hasharray[i, 0] = hash(y[i])
# #
# # plt.scatter(pc[:, 0], pc[:, 1], c=hasharray[:, 0])
# # plt.title('Scatter plot PCA on ' + fileName.split('.')[0])
# # plt.xlabel('x')
# # plt.ylabel('y')
# # plt.show()
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
#
# targets = ['1', '2', '3', '4', '5']
# ax.legend(targets)
# ax.grid()



import pandas as pd
inp = [{'c1':10, 'c2':100}, {'c1':11,'c2':110}, {'c1':12,'c2':120}]
df = pd.DataFrame(inp)
print(df)


for row in df.iterrows():

   print(row["c1"], row["c2"])