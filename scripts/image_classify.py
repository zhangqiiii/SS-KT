import os

import cv2
import numpy as np
import pickle

from sklearn.cluster import KMeans


def cls_cluster(img, name, n_cluster=5, filter_size=7):
    """
    Classify an image using KMeans clustering. Use median filtering to filter cluster map.
    Args:
        img: ndarry (C, H, W)
        name: string, dataset name
        n_cluster: int, the number of clusters
        filter_size: int, the filter kernel size
    Returns:
        labels: ndarry (H, W)
        centers: cluster centers
    """
    km = KMeans(n_cluster)
    if img.ndim == 2:
        h, w = img.shape
        img = img.reshape((-1, 1))
    else:
        c, h, w = img.shape
        img = img.transpose((1, 2, 0)).reshape((-1, img.shape[0]))
    label = km.fit_predict(img)
    centers = km.cluster_centers_
    label = label.reshape((h, w))
    # label = (label - np.min(label)) / (np.max(label) - np.min(label)) * 255
    label = label.astype(np.uint8)
    label = cv2.medianBlur(label, filter_size)
    cv2.imwrite(f'cluster_{n_cluster}_filter_{filter_size}/{name}.jpg', label * (255 / (n_cluster - 1)))
    os.makedirs(f"cluster_{n_cluster}_filter_{filter_size}", exist_ok=True)
    a_file = open(f"cluster_{n_cluster}_filter_{filter_size}/{name}.pkl", "wb")
    pickle.dump((label, centers), a_file)
    a_file.close()
    return label, centers


if __name__ == "__main__":
    data_name_list = ['Italy_1.bmp', 'Italy_2.bmp', 'yellow_1.bmp', 'yellow_2.bmp',
                      'California_1.bmp', 'California_2.bmp', 'Texas_1.bmp', 'Texas_2.bmp',
                      'shuguang_1.bmp', 'shuguang_2.bmp']
    data_name_list += ['Gloucester1_1.bmp', 'Gloucester1_2.bmp',
                      'Gloucester2_1.bmp', 'Gloucester2_2.bmp',
                      'wuhan_1.png', 'wuhan_2.jpg']
    # data_name_list = ['Italy_1.bmp', 'Italy_2.bmp']
    for data_name in data_name_list:
        img_ = cv2.imread(f"../data/optsar/{data_name[:-6]}/{data_name}").transpose((2, 0, 1))
        # img_ = cv2.imread("../data/optsar/yellow/yellow_2.bmp").transpose((2, 0, 1))
        # img_ = cv2.imread("../data/optsar/California/California_2.bmp").transpose((2, 0, 1))
        # img_ = cv2.imread("../data/optsar/Texas/Texas_2.bmp").transpose((2, 0, 1))
        # img_ = cv2.imread("../data/optsar/shuguang/shuguang_2.bmp").transpose((2, 0, 1))
        # cls_cluster(img_, data_name[:-4], 5)
        # cls_cluster(img_, data_name[:-4], 5, 1)
        cls_cluster(img_, data_name[:-4], 9, 7)
