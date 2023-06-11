import numpy as np # linear algebra
import matplotlib.pyplot as plt # for plotting
from scipy.spatial.distance import cdist # for distances calculation

class KMeans(object):

    def __init__(self, n_clusters, init='random', max_iter=300, center_type='centroids'):

        initializations = ['random', 'kmeans++']
        center_types = ['centroids', 'medoids']

        if init not in initializations:
            raise ValueError(f'possible initializations include {initializations}')

        if center_type not in center_types:
            raise ValueError(f'possible center types include {center_types}')

        self.init = init
        self.center_type = center_type

        if isinstance(n_clusters, int):
            self.n_clusters = n_clusters
        else:
            ValueError('expected int for n_clusters')

        self.max_iter = max_iter


    def fit(self, X):
        
        # save data
        self._X_fit = X

        # extract number of features and dimensions
        n_feats, dims = X.shape

        # intialize kmeans
        centers = self._initialize(X)

        # epochs + old_labels
        epoch = 0
        old_labels = np.zeros(n_feats, )

        while True:

            # find the distances
            dists = cdist(X, centers)

            # find the labels
            labels = np.argmin(dists, axis=-1)

            # update centroids
            centers = self._calculate_center(X, labels)

            # breaking if no change in the labels:
            if np.allclose(labels, old_labels):
                break

            # breaking at maximum iteration
            if epoch == self.max_iter:
                break
            
            # update
            old_labels = labels
            epoch += 1


        # saving variables
        self.centers = centers
        self.labels = labels
        self.set_labels = set(labels)
        self.loss = np.sum(dists.min(axis=-1))
        self.final_epoch = epoch

    def _initialize(self, X):
        """ Initialization routine """
        if self.init == 'random':
            idxs = np.random.choice(len(X), size=self.n_clusters, replace=False)
            return X[idxs]
        
        elif self.init == 'kmeans++':
            centers = []

            # first chosen randomly
            rand_idx = np.random.choice(len(X), size=1, replace=False)
            centers.append(X[rand_idx].ravel())

            # the other one are chosen accordingly to a prob distr
            for idx in range(1, self.n_clusters):

                # calculate distances
                dists = cdist(X, np.array(centers))

                # get minimum and square
                min_dists = np.min(dists, axis=-1) ** 2

                # chose new one accordingly to probability distribution
                probs = min_dists / min_dists.sum()
                rand_idx = np.random.choice(len(X), size=1, replace=False, p=probs)
                centers.append(X[rand_idx].ravel())

            return np.array(centers)

        else:
            raise NotImplementedError()

    def _calculate_center(self, X, labels):
        """ Calculate centers routine """

        if self.center_type == 'centroids':
            return np.array([np.mean(X[labels==k], axis=0) for k in set(labels)])

        if self.center_type == 'medoids':
            centers = []
            for k in range(self.n_clusters):
                # extract points in cluster k
                pts = X[labels==k]
                # compute pairwise distances
                dists = cdist(pts, pts)
                # compute sum of distances
                sum_dists = dists.sum(axis=-1)
                # find the one with lower distance
                idx = np.argmin(sum_dists)
                # put the medoid in the list
                centers.append(pts[idx])
            return np.array(centers)
            
        else:
            raise NotImplementedError()

    def plot_kmeans(self):

        plt.figure(figsize=(10,8))
        plt.scatter(self._X_fit[:,0], self._X_fit[:,1], c=self.labels)
        plt.plot(self.centers[:,0], self.centers[:,1], 'ro')
        plt.title(f"Final loss {self.loss:.2f}")
        plt.show()


