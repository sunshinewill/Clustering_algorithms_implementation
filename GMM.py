import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys

fileName = sys.argv[1]
data = pd.read_csv(filepath_or_buffer=fileName, header=None, sep='\t')
# starts from column 2 to the end, representing gene's expression values
X = data.iloc[:, 2:].values


# ground truth is column 1
y = data.iloc[:, 1].values
y = y.reshape(1, len(y))
class GMM:
    def __init__(self, X, number_of_cluster, iterations, Y):
        self.iterations = iterations
        self.number_of_cluster = number_of_cluster
        self.X = X
        self.Y = Y
        self.mu = None
        self.pi = None
        self.cov = None
        self.smoothing_value = 1e-9 * np.ones(X.shape[1])
        self.convergence_threshold = 1e-9

    def run(self):
        # Initialize mu, covariance and pi
        # k x d
        # self.mu = np.array([[0, 0], [1, 1], [2, 2]])
        self.mu = np.random.randint(min(self.X[:, 1]), max(self.X[:, 1]), size=(self.number_of_cluster, len(self.X[0])))

        # n x m matrix (n clusters and m dimensions)
        # self.cov = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]])*1.0
        self.cov = np.ones((self.number_of_cluster, len(X[0]), len(X[0])))
        # self.pi = np.array([0.3, 0.3, 0.4])
        self.pi = np.ones(self.number_of_cluster) / self.number_of_cluster

        log_likelihoods = [0]
        # Store the log likelihoods for each iteration and check if it has converged

        # E-step
        for i in range(self.iterations):
            # r_ic is n x k matrix storing the possibilities of all data belonging to each clusters
            r_ic = np.zeros((self.X.shape[0], self.number_of_cluster))
            for m, co, p, r in zip(self.mu, self.cov, self.pi, range(self.number_of_cluster)):
                co += self.smoothing_value
                mn = multivariate_normal(mean=m, cov=co, allow_singular=True)
                r_ic[:, r] = p * mn.pdf(self.X) / np.sum([pi_c * multivariate_normal(mean=mu_c, cov=cov_c, allow_singular=True).pdf(self.X) for
                                                          pi_c, mu_c, cov_c in zip(self.pi, self.mu, self.cov + self.smoothing_value)], axis=0)
            # M-step
            self.mu = []
            self.cov = []
            self.pi = []
            for c in range(self.number_of_cluster):
                m_c = np.sum(r_ic[:, c], axis=0)
                mu_c = (1 / m_c) * np.sum(self.X * r_ic[:, c].reshape(self.X.shape[0], 1), axis=0)
                self.mu.append(mu_c)
                # Calculate the covariance matrix per cluster based on the new mean
                self.cov.append((1/m_c)*np.dot((np.array(r_ic[:, c]).reshape(self.X.shape[0], 1)*(self.X-mu_c)).T, (self.X-mu_c)))
                # Calculate new pi values
                self.pi.append(m_c / self.X.shape[0])

            log_likelihoods.append(np.log(np.sum([k * multivariate_normal(self.mu[ii], self.cov[j], allow_singular=True).pdf(self.X)
                                                  for k, ii, j in zip(self.pi, range(len(self.mu)), range(len(self.cov)))])))

            if abs(log_likelihoods[-1]-log_likelihoods[-2]) < self.convergence_threshold:
                break

        # print (self.mu)
        # print (self.cov)
        # print (self.pi)

        # Assign each data point to the cluster with the maximum possibility

        r_pred = []
        for i in range(r_ic.shape[0]):
            r_pred.append(np.argmax(r_ic[i])+1)

        r_pred = np.array(r_pred).reshape(1, len(r_pred))
        self.resMatrix = r_pred == r_pred.T
        self.gtMatrix = self.Y == self.Y.T

        return r_pred

    def randIndex(self):
        randMatrix = (self.resMatrix == self.gtMatrix).astype(int)
        return randMatrix.sum()*1.0/randMatrix.size

    def jaccardCoefficient(self):
        return (self.resMatrix & self.gtMatrix).astype(int).sum()*1.0/(self.resMatrix | self.gtMatrix).astype(int).sum()

GMM = GMM(X, 10, 100, y)
res = GMM.run()
print("randIndex:")
print(GMM.randIndex())
print("jaccardCoefficient:")
print(GMM.jaccardCoefficient())

pcaData = X
if X.shape[1]>2:
    # print("y")
    pca = PCA(n_components=2)
    pcaData = pca.fit_transform(X)

res = res.reshape(res.shape[1])
df_pca = pd.DataFrame(dict(x=list(pcaData[:,0]),y=list(pcaData[:,1]), labels= res))

lb = set(res)

for i in lb:
    plt.scatter(df_pca[df_pca['labels'] == i]['x'], df_pca[df_pca['labels'] == i]['y'], label=i)

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
