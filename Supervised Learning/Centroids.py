# Create a function to find the closest centroids.
def findClosestCentroids(X, centroids):
    """
    Returns the closest centroids in idx for a dataset X
    where each row is a single example. idx = m x 1 vector
    of centroid assignments (i.e. each entry in range [1..K])
    Args:
        X        : array(# training examples, n)
        centroids: array(K, n)
    Returns:
        idx      : array(# training examples, 1)
    """
    # Set K size.
    K = centroids.shape[0]

    # Initialise idx.
    idx = np.zeros((X.shape[0], 1), dtype=np.int8)

    # Iterate over every example, find its closest centroid, and store
    # the index inside idx at the appropriate location. Concretely,
    # idx[i] should contain the index of the centroid closest to
    # example i. Hence, it should be a value in the range 1..K.

    #     # Iterate over training examples.
    #     for i in range(X.shape[0]):
    #         # Set norm distance to infinity.
    #         min_dst = math.inf
    #         # Iterate over centroids.
    #         for k in range(K):
    #             # Compute the norm distance.
    #             dst = np.linalg.norm(X[i,:] - centroids[k,:], axis=0)
    #             if dst < min_dst:
    #                 min_dst = dst
    #                 idx[i] = k

    # Alternative partial vectorized solution.
    # Iterate over training examples.
    for i in range(X.shape[0]):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        # argmin returns the indices of the minimum values along an axis,
        # replacing the need for a for-loop and if statement.
        min_dst = np.argmin(distances)
        idx[i] = min_dst

    return idx


# Find the closest centroids for the examples.
idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: \n')
print(idx[:3])
print('\n(The closest centroids should be 0, 2, 1 respectively)')