import numpy as np

train_indices = [3, 6, 24, 32, 19, 17, 13, 42, 15, 9, 16, 37, 31, 27, 0, 30, 29, 5, 11, 33, 1, 40, 21, 2, 34, 23, 36,
                 10, 22, 18, 44, 20, 7, 14, 28, 38]
test_indices = [39, 25, 26, 43, 35, 41, 4, 12, 8]

fin_indices = [4, 8, 41]

fin_test = True


# train_indices = [ 28, 18, 34, 29, 24, 2, 5, 13, 30, 38, 23, 3, 35, 33, 26, 14, 15, 20, 8, 32, 21, 17, 0, 9, 43, 41, 10, 31, 19, 42, 12, 1, 37, 7, 27, 6 ]
# test_indices = [ 25, 22, 36, 40, 4, 44, 39, 16, 11 ]

def train_test_split_indices(size, test_ratio, seed=None):
    """
    Split indices into training and testing sets based on the given test ratio.

    Parameters:
    size (int): Total number of elements.
    test_ratio (float): Ratio of the dataset to be used as the test set.

    Returns:
    tuple: Two lists, the first containing the training indices and the second containing the testing indices.
    """
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    # Generate a list of indices from 0 to size-1
    indices = np.arange(size)
    # Shuffle the indices
    np.random.shuffle(indices)

    # Calculate the number of test samples
    test_size = int(size * test_ratio)

    # Split the indices into test and train
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return train_indices, test_indices


if __name__ == '__main__':
    pass
    # size = 45
    # test_ratio = 0.2
    # train_indices, test_indices = train_test_split_indices(size, test_ratio, seed=42)
    # print("train_indices = [", ', '.join([str(i) for i in train_indices]), "]")
    # print("test_indices = [", ', '.join([str(i) for i in test_indices]), "]")
