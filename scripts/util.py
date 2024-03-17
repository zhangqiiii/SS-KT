import pickle

import cv2
import numpy as np
import torch


def set_requires_grad(models, requires_grad=False):
    if not isinstance(models, list):
        models = [models, ]

    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad


def cos_smi(x, y):
    x = torch.tensor(x)
    y = torch.tensor(y)
    img = torch.cosine_similarity(x, y, 0)
    img = img.numpy()
    return img


def _norm_one(img):
    """
    normalize to [0,1]
    Args:
        img: img with (c, h, w)
    Returns:

    """
    if img.ndim == 2:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    else:
        for i in range(img.shape[0]):
            img[i] = (img[i] - np.min(img[i])) / (np.max(img[i]) - np.min(img[i]))
    return img


def save_image(path, img, pkl=False):
    """
    Save ndarray as image. You can choose whether to save the raw data as a pkl file.
    Args:
        path: path to save
        img: ndarray image, shape is (h, w) or ([1|3], h, w)
        pkl: whether to save raw data
    """
    if pkl:
        pickle.dump(img, open(path+'.pkl', 'wb'))

    img = img.copy()
    channel = 1 if img.ndim == 2 else img.shape[0]

    if img.ndim == 2:
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    else:
        if channel != 3 and channel != 1:
            return
        for i in range(channel):
            img[i] = (img[i] - np.min(img[i])) / (np.max(img[i]) - np.min(img[i])) * 255
        img = img.transpose(1, 2, 0)
        img = img.squeeze()

    img = img.astype(np.uint8)
    cv2.imwrite(path, img)


def _normalize(img: np.ndarray):
    """
    Normalize ndarray to [0, 1] channel by channel.
    """
    channel = 1 if img.ndim == 2 else img.shape[0]

    for i in range(channel):
        ch_min = np.min(img[i])
        ch_max = np.max(img[i])
        if ch_max - ch_min == 0:
            img[i] = 0
        else:
            img[i] = (img[i] - ch_min) / (np.max(img[i]) - np.min(img[i]))
    return img


def _to_grey(img):
    """
    Convert RGB image to grayscale image.
    Args:
        img : ndarray (3, h, w)
    """
    weight = [0.3, 0.3, 0.4]
    assert img.shape[0] == 3
    tmp_img = None
    for i in range(len(weight)):
        if tmp_img is None:
            tmp_img = img[i] * weight[i]
        else:
            tmp_img += img[i] * weight[i]
    return tmp_img

