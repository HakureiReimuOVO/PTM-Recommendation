import numpy as np
from slice_dataset import get_all_datasets_and_idx

# 2-class
# train_indices = [ 6, 24, 32, 19, 17, 13, 42, 15, 9, 16, 37, 31, 27, 0, 30, 29, 5, 11, 33, 1, 40, 21, 2, 34, 23, 36, 10, 22, 18, 44, 20, 7, 14, 28, 38 ]
# test_indices = [ 39, 25, 26, 43, 35, 41 ]
# fin_indices = [ 4, 12, 8, 3 ]

# 5-class
# train_indices = [ 85, 117, 38, 118, 150, 75, 161, 2, 108, 46, 162, 153, 109, 148, 26, 156, 100, 113, 125, 36, 101, 22, 177, 76, 112, 11, 104, 6, 27, 132, 4, 32, 78, 145, 133, 128, 10, 170, 147, 0, 137, 70, 64, 44, 120, 28, 40, 139, 140, 25, 23, 135, 81, 79, 84, 39, 141, 86, 77, 176, 152, 47, 94, 138, 169, 61, 73, 33, 43, 136, 142, 62, 180, 173, 124, 105, 53, 5, 163, 3, 168, 49, 35, 80, 34, 7, 110, 91, 83, 159, 164, 89, 8, 13, 59, 155, 131, 17, 72, 158, 134, 167, 166, 63, 54, 107, 50, 178, 58, 48, 88, 21, 57, 175, 129, 37, 157, 171, 1, 52, 149, 130, 151, 103, 99, 116, 87, 74, 121, 181, 20, 71, 106, 14, 92, 179, 102 ]
# test_indices = [ 19, 42, 154, 98, 146, 15, 24, 68, 115, 96, 95, 160, 69, 111, 45, 16, 51, 127, 97, 56, 174, 122, 144, 30, 9, 123, 60 ]
# fin_indices = [ 18, 165, 143, 172, 55, 90, 82, 66, 29, 119, 65, 67, 31, 12, 41, 126, 93, 114 ]

# total: seed 39
train_indices = [ 103, 60, 175, 183, 95, 162, 190, 90, 26, 9, 209, 125, 88, 38, 34, 211, 134, 63, 24, 114, 28, 124, 51, 129, 130, 74, 37, 188, 165, 192, 18, 210, 200, 14, 226, 118, 83, 111, 116, 108, 53, 150, 218, 107, 65, 58, 49, 154, 48, 43, 163, 177, 79, 52, 216, 27, 109, 161, 220, 82, 155, 46, 203, 180, 7, 59, 81, 181, 57, 32, 101, 202, 77, 4, 66, 112, 204, 135, 179, 212, 92, 139, 15, 167, 156, 173, 215, 71, 47, 128, 225, 178, 42, 223, 67, 110, 160, 146, 193, 11, 39, 164, 98, 102, 68, 106, 166, 199, 10, 208, 1, 16, 97, 115, 147, 35, 25, 0, 224, 76, 185, 195, 40, 168, 104, 33, 3, 56, 221, 120, 31, 100, 73, 62, 94, 54, 45, 151, 87, 214, 72, 8, 69, 198, 121, 13, 217, 84, 36, 142, 89, 222, 145, 158, 172, 12, 205, 30, 141, 159, 75, 187, 132, 176, 123, 70, 148, 64, 41, 105, 137 ]
test_indices = [ 23, 140, 5, 207, 96, 44, 138, 189, 17, 201, 219, 197, 157, 61, 174, 206, 196, 126, 85, 191, 78, 194, 153, 55, 152, 213, 144, 19, 171, 6, 2, 143, 184, 149 ]
fin_indices = [ 169, 20, 170, 93, 80, 133, 186, 91, 113, 50, 136, 29, 117, 122, 99, 131, 86, 119, 21, 22, 182, 127 ]

# Seed 42: No 2-class test sample
# train_indices = [ 211, 38, 24, 66, 184, 195, 10, 136, 29, 109, 67, 215, 5, 56, 113, 138, 222, 65, 132, 199, 135, 31, 12, 35, 28, 42, 177, 114, 154, 51, 159, 181, 76, 41, 96, 139, 78, 180, 26, 150, 170, 127, 0, 2, 77, 46, 100, 144, 137, 163, 90, 85, 196, 208, 98, 36, 213, 61, 22, 148, 168, 33, 11, 221, 178, 6, 27, 140, 217, 218, 155, 4, 122, 32, 162, 62, 128, 203, 171, 70, 189, 64, 44, 147, 40, 123, 23, 190, 165, 81, 39, 220, 47, 94, 172, 43, 145, 156, 3, 105, 53, 133, 204, 175, 209, 49, 80, 34, 7, 110, 91, 83, 200, 205, 89, 8, 13, 59, 193, 131, 17, 166, 72, 197, 134, 186, 207, 63, 54, 107, 50, 174, 223, 198, 169, 58, 48, 88, 21, 57, 160, 219, 187, 191, 129, 37, 157, 212, 1, 52, 149, 130, 151, 103, 99, 116, 87, 202, 74, 214, 210, 121, 226, 20, 188, 71, 106, 14, 92, 179, 102 ]
# test_indices = [ 9, 143, 15, 124, 153, 75, 126, 95, 45, 25, 104, 176, 201, 19, 30, 115, 182, 117, 86, 84, 192, 119, 225, 161, 18, 101, 55, 97, 167, 194, 173, 73, 79, 16 ]
# fin_indices = [ 112, 158, 118, 152, 146, 125, 82, 142, 120, 69, 111, 93, 60, 164, 68, 183, 216, 141, 224, 185, 206, 108 ]

fin_test = True
# fin_test = False


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
    size = 227
    test_ratio = 0.15
    fin_ratio = 0.1
    train_indices, test_indices, fin_indices = train_test_fin_split_indices(size, test_ratio, fin_ratio, seed=42)
    print("train_indices = [", ', '.join([str(i) for i in train_indices]), "]")
    print("test_indices = [", ', '.join([str(i) for i in test_indices]), "]")
    print("fin_indices = [", ', '.join([str(i) for i in fin_indices]), "]")

    pass
