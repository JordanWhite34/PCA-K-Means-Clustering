#### PROBLEM 2 ####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist

# to calculate accuracy
def accuracy_score(y_labels, predicted_labels):
    return np.sum(y_labels == predicted_labels)/len(y_labels)

# implementation of the pca class
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        covariance_matrix = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        self.components = np.array([eig_pairs[i][1] for i in range(self.n_components)]).T

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# implementation of KMeans
class CustomKMeans:
    def __init__(self, n_clusters, n_init=1, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.init = 'fps'
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None

    @staticmethod
    def _fps_sampling(X, k): # this is our implementation of furthest point sampling
        centroids = [X[0]]
        for _ in range(k - 1):
            dist = cdist(X, centroids).min(axis=1)
            idx = np.argmax(dist)
            centroids.append(X[idx])
        return np.array(centroids)

    def _initialize_centroids(self, X): # this initializes the centroids
        if self.init == 'fps':
            return self._fps_sampling(X, self.n_clusters)
        else:
            raise ValueError(f"Unknown initialization method '{self.init}'")

    def _assign_points(self, X, centroids): # correct assignment of points
        return np.argmin(cdist(X, centroids), axis=1)

    def _update_centroids(self, X, assignments): # updating the centroids
        return np.array([X[assignments == i].mean(axis=0) for i in range(self.n_clusters)])

    def _k_means_single_run(self, X): # singly running k means
        centroids = self._initialize_centroids(X)
        assignments = self._assign_points(X, centroids)

        for _ in range(self.max_iter):
            new_centroids = self._update_centroids(X, assignments)
            new_assignments = self._assign_points(X, new_centroids)

            if np.linalg.norm(new_centroids - centroids) < self.tol:
                break

            centroids = new_centroids
            assignments = new_assignments

        return centroids, assignments

    def fit(self, X): # fitting data
        best_inertia = None
        best_centroids = None
        best_assignments = None

        for _ in range(self.n_init):
            centroids, assignments = self._k_means_single_run(X)
            inertia = np.sum([np.linalg.norm(X[assignments == i] - centroids[i])**2 for i in range(self.n_clusters)])

            if best_inertia is None or inertia < best_inertia: # placeholder
                best_inertia = inertia
                best_centroids = centroids
                best_assignments = assignments

        self.cluster_centers_ = best_centroids
        self.labels_ = best_assignments
        return self

    def predict(self, X): # predictions
        return self._assign_points(X, self.cluster_centers_)




# Load data
train_data = pd.read_csv("HW1_2/mnist_train.csv")
test_data = pd.read_csv("HW1_2/mnist_test.csv")

X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# furthest point sampling
def fps_sampling(X, k):
    centroids = [X[0]]
    for q in range(k - 1):
        dist = cdist(X, centroids).min(axis=1)
        idx = np.argmax(dist)
        centroids.append(X[idx])
    return np.array(centroids)


# kmeans function, for returning centroids and assignments
def k_means(X, k):
    centroids = fps_sampling(X, k)
    assignments = np.argmin(cdist(X, centroids), axis=1)
    updated = True

    while updated:
        updated = False
        new_centroids = np.array([X[assignments == i].mean(axis=0) for i in range(k)])
        new_assignments = np.argmin(cdist(X, new_centroids), axis=1)

        if not np.array_equal(assignments, new_assignments):
            updated = True
            centroids = new_centroids
            assignments = new_assignments

    return centroids, assignments

# 2b
def plot_k_means(X, centroids, assignments, title):
    plt.scatter(X[:, 0], X[:, 1], c=assignments, cmap="tab10")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="x")
    plt.title(title)
    plt.show()

def k_means_with_visualization(X, k):
    # Initialization
    centroids = fps_sampling(X, k)
    assignments = np.argmin(cdist(X, centroids), axis=1)
    plot_k_means(X, centroids, assignments, "Initial k-means clustering (k = 10)")

    # First step
    centroids = np.array([X[assignments == i].mean(axis=0) for i in range(k)])
    assignments = np.argmin(cdist(X, centroids), axis=1)
    plot_k_means(X, centroids, assignments, "First step of k-means clustering (k = 10)")

    # Remaining steps
    updated = True
    while updated:
        updated = False
        new_centroids = np.array([X[assignments == i].mean(axis=0) for i in range(k)])
        new_assignments = np.argmin(cdist(X, new_centroids), axis=1)

        if not np.array_equal(assignments, new_assignments):
            updated = True
            centroids = new_centroids
            assignments = new_assignments

    return centroids, assignments

k = 10
X_train_100 = X_train_pca[:100]

centroids, assignments = k_means_with_visualization(X_train_100, k)
plot_k_means(X_train_100, centroids, assignments, "Final k-means clustering (k = 10)")


# 2c
def evaluate_accuracy(assignments, y_train, k):
    label_map = {}
    for i in range(k):
        label_map[i] = np.argmax(np.bincount(y_train[assignments == i]))
    predicted_labels = np.array([label_map[assignment] for assignment in assignments])
    return accuracy_score(y_train, predicted_labels) # HERE

accuracy = evaluate_accuracy(assignments, y_train[:100], k)
print(f"Classification accuracy: {accuracy:.4f}")

# 2d
def test_k_and_pca(k_values, pca_dim, X_train, y_train, X_test, y_test):
    for k in k_values:
        pca = PCA(n_components=pca_dim)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        centroids, assignments = k_means(X_train_pca, k)
        train_accuracy = evaluate_accuracy(assignments, y_train, k)

        kmeans = CustomKMeans(n_clusters=k, n_init=1)
        kmeans.fit(X_train_pca)
        test_assignments = kmeans.predict(X_test_pca)
        test_accuracy = evaluate_accuracy(test_assignments, y_test, k)

        print(f"k = {k}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

k_values = [5, 10, 20, 40]
pca_dim = 30
test_k_and_pca(k_values, pca_dim, X_train, y_train, X_test, y_test)




# 2e
def gmm_clustering(X_train, y_train, X_test, y_test, n_components):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(X_train)
    train_assignments = gmm.predict(X_train)
    test_assignments = gmm.predict(X_test)

    label_map = {}
    for i in range(n_components):
        label_map[i] = np.argmax(np.bincount(y_train[train_assignments == i]))

    predicted_train_labels = np.array([label_map[assignment] for assignment in train_assignments])
    predicted_test_labels = np.array([label_map[assignment] for assignment in test_assignments])

    train_accuracy = accuracy_score(y_train, predicted_train_labels)  # HERE
    test_accuracy = accuracy_score(y_test, predicted_test_labels)  # HERE

    return train_accuracy, test_accuracy

n_components = 10
train_accuracy, test_accuracy = gmm_clustering(X_train_pca, y_train, X_test_pca, y_test, n_components)
print(f"GMM - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
