import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class DBSCAN(object):

    def __init__(self, data, MinPts, Eps):
        self.data = data
        self.MinPts = MinPts
        self.Eps = Eps

    def clustering(self):
        self.clusterResult = np.zeros(self.data.shape[0])

        C=0
        for i in range(self.data.shape[0]):
            if self.clusterResult[i]!=0:
                continue
            self.clusterResult[i] = -1
            NeighborPts = self.regionQuery(i)
            if(len(NeighborPts)>=self.MinPts):
                C += 1
                self.clusterResult[i] = C
                self.expandCluster(NeighborPts,C)

        return self.clusterResult

    def expandCluster(self,NeighborPts,C):
        for j in NeighborPts:
            if self.clusterResult[j] == 0:
                self.clusterResult[j] = C
                NeighborPtsPrime = self.regionQuery(j)
                if(len(NeighborPtsPrime)>=self.MinPts):
                    self.expandCluster(NeighborPtsPrime,C)
            if self.clusterResult[j] == -1:
                self.clusterResult[j] = C

    def regionQuery(self, i):
        NeighborPts = set()
        for j in range(self.data.shape[0]):
            if(np.linalg.norm(self.data[j]-self.data[i])<=self.Eps):
                NeighborPts.add(j)
        return NeighborPts


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
MinPts = int(sys.argv[2])
Eps = float(sys.argv[3])

df = pd.read_csv(
            filepath_or_buffer=filename,
            header=None,
            sep='\t')

groundTruth = df.iloc[:,1].values
groundTruth = groundTruth.reshape(1,len(groundTruth))

data = df.iloc[:,2:].values

db = DBSCAN(data, MinPts,Eps)
res = db.clustering().astype(int)
print(res)
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