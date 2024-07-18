import numpy as np

# 2-class
# train_indices = [3, 6, 24, 32, 19, 17, 13, 42, 15, 9, 16, 37, 31, 27, 0, 30, 29, 5, 11, 33, 1, 40, 21, 2, 34, 23, 36,
#                  10, 22, 18, 44, 20, 7, 14, 28, 38]
# test_indices = [39, 25, 26, 43, 35, 41, 4, 12, 8]
#
# fin_indices = [4, 8, 41]

# 5-class
train_indices = [ 26, 141, 117, 93, 133, 36, 82, 22, 126, 67, 97, 11, 65, 86, 6, 27, 76, 142, 38, 41, 4, 138, 32, 144, 109, 68, 10, 96, 111, 0, 122, 123, 64, 44, 146, 28, 40, 114, 25, 23, 119, 81, 79, 39, 90, 108, 158, 137, 47, 124, 61, 73, 33, 112, 120, 128, 62, 161, 100, 104, 53, 5, 118, 127, 150, 49, 35, 80, 77, 34, 46, 7, 43, 70, 125, 110, 91, 83, 147, 148, 89, 8, 155, 113, 13, 59, 140, 3, 17, 72, 143, 136, 149, 63, 54, 107, 50, 160, 58, 48, 88, 21, 57, 157, 129, 37, 153, 1, 52, 130, 103, 99, 116, 87, 74, 121, 162, 20, 71, 106, 14, 92, 102 ]
test_indices = [ 135, 115, 131, 55, 95, 29, 156, 51, 101, 145, 19, 85, 15, 66, 24, 30, 132, 105, 151, 16, 75, 18, 12, 9 ]
fin_indices = [ 31, 154, 98, 56, 134, 159, 139, 78, 60, 84, 2, 94, 45, 42, 69, 152 ]

# total
# train_indices = [195, 29, 19, 55, 93, 181, 158, 5, 132, 56, 127, 115, 146, 108, 177, 31, 12, 35, 28, 42, 79, 97, 142,
#                  51, 117, 168, 76, 41, 128, 78, 176, 26, 139, 109, 198, 2, 77, 46, 126, 111, 90, 85, 140, 36, 118, 61,
#                  22, 137, 145, 33, 11, 202, 165, 6, 27, 156, 199, 143, 4, 32, 112, 192, 119, 150, 114, 10, 62, 122, 152,
#                  172, 0, 187, 159, 70, 162, 64, 44, 136, 40, 123, 23, 186, 153, 81, 39, 175, 47, 94, 182, 43, 144, 3,
#                  105, 53, 133, 201, 163, 193, 49, 80, 34, 7, 110, 91, 83, 184, 189, 89, 8, 13, 59, 206, 131, 17, 72,
#                  183, 134, 173, 191, 63, 54, 107, 50, 174, 204, 169, 58, 48, 88, 21, 57, 160, 200, 129, 37, 157, 196, 1,
#                  52, 149, 130, 151, 103, 99, 116, 87, 74, 121, 207, 20, 188, 71, 106, 14, 92, 179, 102]
# test_indices = [161, 15, 73, 96, 166, 9, 100, 135, 18, 148, 171, 30, 155, 180, 125, 197, 164, 190, 84, 75, 124, 170,
#                 104, 101, 69, 25, 95, 16, 141, 185, 154]
# fin_indices = [68, 66, 120, 147, 98, 138, 167, 45, 113, 65, 178, 86, 203, 67, 82, 205, 194, 38, 24, 60]

fin_test = True


def train_test_fin_split_indices(size, test_ratio, fin_ratio, seed=None):
    """
    Split indices into training, validation, and testing sets based on the given test and validation ratios.

    Parameters:
    size (int): Total number of elements.
    test_ratio (float): Ratio of the dataset to be used as the test set.
    val_ratio (float): Ratio of the dataset to be used as the validation set.

    Returns:
    tuple: Three lists, the first containing the training indices, the second containing the validation indices,
           and the third containing the testing indices.
    """
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Generate a list of indices from 0 to size-1
    indices = np.arange(size)
    # Shuffle the indices
    np.random.shuffle(indices)

    # Calculate the number of test and validation samples
    test_size = int(size * test_ratio)
    fin_size = int(size * fin_ratio)

    # Split the indices into test, validation, and train
    test_indices = indices[:test_size]
    fin_indices = indices[test_size:test_size + fin_size]
    train_indices = indices[test_size + fin_size:]

    return train_indices, test_indices, fin_indices


if __name__ == '__main__':
    size = 163
    test_ratio = 0.15
    fin_ratio = 0.1
    train_indices, test_indices, fin_indices = train_test_fin_split_indices(size, test_ratio, fin_ratio, seed=42)
    print("train_indices = [", ', '.join([str(i) for i in train_indices]), "]")
    print("test_indices = [", ', '.join([str(i) for i in test_indices]), "]")
    print("fin_indices = [", ', '.join([str(i) for i in fin_indices]), "]")

    pass
