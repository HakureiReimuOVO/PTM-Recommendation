import numpy as np

SGR_PATH = 'meta_features/AutoSGR/'
MRM_PATH = 'meta_features/AutoMRM/'
IMG_PATH = 'meta_features/ImageDataset2Vec/'


def get_sgr_features(dataset_name):
    return np.load(SGR_PATH + dataset_name + '.npy')


def get_mrm_features(dataset_name):
    return np.load(MRM_PATH + dataset_name + '.npy')


def get_imgdataset2vec_features(dataset_name):
    return np.load(IMG_PATH + dataset_name + '.npy')
