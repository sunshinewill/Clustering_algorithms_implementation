import sys
import pandas as pd
import numpy as np
import math
from copy import deepcopy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class spectralClustering(object):

    def __init__(self, sigma,data):
        self.sigma = sigma
        self.data = data

    def matrix(self):
        row = self.data.shape[0]
        self.w_matrix = np.zeros((row, row))

        for i in range(row):
            for j in range(i+1,row):
                self.w_matrix[i][j] = math.exp(-1*np.sum((data[i]-data[j])**2)/sigma**2)

        self.w_matrix += self.w_matrix.T
        self.d_matrix = np.zeros((row, row))
        for i in range(row):
            self.d_matrix[i][i] = np.sum(self.w_matrix[i])
        self.l_matrix = self.d_matrix-self.w_matrix

    def decomposition(self):
        eig_vals, eig_vecs = np.linalg.eigh(self.l_matrix)
        max = 0
        k = 0
        for i in range(1,eig_vals.shape[0]):
            if eig_vals[i]-eig_vals[i-1]>max:
                max = eig_vals[i]-eig_vals[i-1]
                k = i-1
        self.kspace = eig_vecs[:,:k]
        return self.kspace


class kmeans(object):
    def __init__(self,df,K):
        self.df = df
        self.K = K

    def clustering(self):
        # dataframe only contains the value of attributes of data
        x_df = self.df
        x_arr = x_df.values

        # pick 5 points as centroids
        CentroidsList = [x_arr[i] for i in range(x_df.shape[0]) if i in (60,90,220,290,340)]
        Centroids = CentroidsList[0].reshape(1,len(CentroidsList[0]))

        for i in range(1,len(CentroidsList)):
            newone = CentroidsList[i]
            newone = newone.reshape(1,len(newone))
            Centroids = np.concatenate((Centroids,newone))

        # Initialize the clusters array
        clusters = np.zeros(x_df.shape[0])

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
                distances = dist(x_arr[i], Centroids)
                cluster = np.argmin(distances)
                clusters[i] = cluster
            Centroids_old = deepcopy(Centroids)
            # upgrade Centroids
            for i in range(self.K):
                points = [x_arr[j] for j in range(len(x_arr)) if clusters[j] == i]
                Centroids[i] = np.mean(points, axis=0)
            diff = dist(Centroids, Centroids_old, None)

        clusters = clusters + 1
        return clusters


class externalIndex(object):

    def __init__(self, result, groundTruth):
        self.resMatrix = result == result.T
        self.gtMatrix = groundTruth == groundTruth.T

    def randIndex(self):
        randMatrix = (self.resMatrix == self.gtMatrix).astype(int)
        return randMatrix.sum()*1.0/randMatrix.size

    def jaccardCoefficient(self):
        return (self.resMatrix & self.gtMatrix).astype(int).sum()*1.0/(self.resMatrix | self.gtMatrix).astype(int).sum()




filename = sys.argv[1]
sigma = float(sys.argv[2])

df = pd.read_csv(
            filepath_or_buffer=filename,
            header=None,
            sep='\t')

groundTruth = df.iloc[:,1].values
groundTruth = groundTruth.reshape(1,len(groundTruth))

data = df.iloc[:,2:].values

spectral = spectralClustering(sigma,data)
spectral.matrix()
kspace = spectral.decomposition()
kspaceDF = pd.DataFrame(data=kspace)

kmeans = kmeans(kspaceDF,5)
res = kmeans.clustering().astype(int)
#print(res)
res = res.reshape(1,len(res))

index = externalIndex(res,groundTruth)
randIndex = index.randIndex()
jaccardCoefficient = index.jaccardCoefficient()

print(randIndex)
print(jaccardCoefficient)

pca = PCA(n_components=2)
pcaData = pca.fit_transform(data)

res = res.reshape(res.shape[1])
df_pca = pd.DataFrame(dict(x=list(pcaData[:,0]),y=list(pcaData[:,1]), labels= res))

lb = set(res)

for i in lb:
    plt.scatter(df_pca[df_pca['labels'] == i]['x'], df_pca[df_pca['labels'] == i]['y'],label = i)


plt.title('Scatter plot by PCA on ' + filename.split('.')[0])
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()