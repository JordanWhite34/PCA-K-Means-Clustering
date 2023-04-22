import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# opening data, assignment to variables
X_train, Y_train, X_test, Y_test = [], [], [], []

with open('HW1_1/X_train.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    train_column_name = next(csv_reader)
    for row in csv_reader:
        row = row[0].split(",")
        row = [float(i) for i in row]
        X_train.append(row)

with open('HW1_1/Y_train.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    train_column_name = next(csv_reader)
    for row in csv_reader:
        row = row[0].split(",")
        row = [float(i) for i in row]
        Y_train.append(row)

with open('HW1_1/X_test.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    train_column_name = next(csv_reader)
    for row in csv_reader:
        row = row[0].split(",")
        row = [float(i) for i in row]
        X_test.append(row)

with open('HW1_1/Y_test.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    train_column_name = next(csv_reader)
    for row in csv_reader:
        row = row[0].split(",")
        row = [float(i) for i in row]
        Y_test.append(row)


# 1Part 1a - Question One
# Data standardization
X_STD = X_train.copy()
X_mean = np.mean(X_train, axis=0)
X_STD -= X_mean


# Part 1a - Question Two
# computing covariance matrix
covariant_matrix = np.cov(np.transpose(X_STD))

# Part 1a - Question Three
# Eigen-decomposition
eigen_values, eigen_vectors = np.linalg.eigh(covariant_matrix) # this is 2 arrays, index 0 is eigenvectors, 1 is eigenvalues
eigen_vectors = eigen_vectors[:, ::-1]

# Part 1a - Question Four
# selecting principal components
def top_k_eigen_vectors(k, vectors):
    new_vectors = np.transpose(vectors)
    new_vectors = -1 * new_vectors[:k]
    return new_vectors

# Part 1a - Question Five
# Dimension reduction
eigen_vectors_250 = top_k_eigen_vectors(250, eigen_vectors)

# Part 1b
# Reporting the 5 largest eigenvalues from 1a
top_five_eigen_vector = eigen_vectors_250[:5]

print("5 largest eigenvalues are", top_five_eigen_vector)

# Part 1c
# computing mean face
mean_face = np.mean(X_train, axis=0).reshape(56, 46)

# plotting mean face and the eigenfaces corresponding to largest 5
def construct_eigen_face(n_face, eigen_v, face_type):
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=n_face + 1, figsize=(10, 5))
    axes[0].imshow(mean_face, cmap='gray')
    axes[0].set_title('Mean Face')
    for i in range(n_face):
        temp_face = eigen_v[i].reshape(56, 46)
        axes[i + 1].imshow(temp_face, cmap='gray')
        axes[i + 1].set_title(f'{face_type} {i + 1}')
    plt.tight_layout()
    plt.show()


construct_eigen_face(5, eigen_vectors_250, "Eigen Face")

# 1d
# reconstructing face with first n=10,20,100,200 eigenfaces
first_row_X = X_train[0]

X_mean = np.mean(X_train, axis=0)
new_X_STD = np.subtract(first_row_X, X_mean)

n = [10, 20, 100, 200]
V_k = np.array([])
X_hat_vectors = []
for i in n:  # changed loop variable to use range(len(n))
     V_k = eigen_vectors_250  # store result of pca in V_k array
     X_hat = X_mean + np.dot(np.dot(np.transpose(V_k), V_k), new_X_STD)  # use the i-th pca result from V_k
     X_hat_vectors.append(X_hat)
     mse = mean_squared_error(first_row_X, X_hat)
     print("MSE:", mse)
     # Plot reconstructed image for current k value


# 1e
# outputting the 5 closest faces in the training dataset to the first 3 entries in the testing dataset
def reconstruct_eigen_face(top_n, vectors, presentation_type, face):
    original_face = np.reshape(X_train[face], (56, 46))
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=len(top_n) + 1, figsize=(10, 5))
    axes[0].imshow(original_face, cmap='gray')
    axes[0].set_title('Original Face')
    for i in range(len(top_n)):
        temp_face = vectors[i].reshape(56, 46)
        axes[i + 1].imshow(temp_face, cmap='gray')
        axes[i + 1].set_title(f'{presentation_type}={top_n[i]}')
    plt.tight_layout()

reconstruct_eigen_face(n, X_hat_vectors, "k", 0)
plt.show()
distances = np.linalg.norm(np.array(X_test)[:, np.newaxis, :] - np.array(X_train), axis=2)

# Find the indices of the 5 closest training images for each testing image
face_indices = np.argsort(distances, axis=1)[:, :5]

# Determine the number of matching identities among the 5 nearest training images for each testing image
num_matches = 0
for i in range(3): # first three testing images
    match_id = i // 2 # identity of the testing image (0 for A, 1 for B, 2 for C)
    match_indices = face_indices[i]
    match_ids = match_indices // 2 # identities of the nearest training images
    num_matches += np.sum(match_ids == match_id)



# Output the number of matching identities among the 5 nearest training images for the first three testing images
print("Number of matching identities among the 5 nearest training images for the first three testing images:", num_matches)
def reconstruct_eigen_face_two(indices, presentation_type, face):
    original_face = np.reshape(X_test[face], (56, 46))
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=len(indices) + 1, figsize=(10, 5))
    axes[0].imshow(original_face, cmap='gray')
    axes[0].set_title('Original Face')
    for i in range(len(indices)):
        temp_face = np.reshape(X_train[indices[i]], (56, 46))
        axes[i + 1].imshow(temp_face, cmap='gray')
        axes[i + 1].set_title(f'{presentation_type}={indices[i]}')
    plt.tight_layout()
    plt.show()

reconstruct_eigen_face_two(face_indices[0, :], "ID", 0)
reconstruct_eigen_face_two(face_indices[1, :], "ID", 1)
reconstruct_eigen_face_two(face_indices[2, :], "ID", 2)


