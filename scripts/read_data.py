import os
from glob import glob

import cv2
import gdal
import numpy as np
from scipy.io import loadmat
from tifffile import tifffile


def read_RGB(dataset_path, reduce=None):
    img_path_1 = glob(os.path.join(dataset_path, "*_1.*"))[0]
    img_path_2 = glob(os.path.join(dataset_path, "*_2.*"))[0]
    img_path_gt = glob(os.path.join(dataset_path, "*_gt.*"))[0]

    img_1 = cv2.imread(img_path_1).astype(np.float)
    if img_1.ndim == 2:
        img_1 = img_1[np.newaxis, ...]
    img_2 = cv2.imread(img_path_2).astype(np.float)
    img_gt = cv2.imread(img_path_gt).astype(np.float)//255

    if reduce is not None:
        h, w = img_1.shape[:2]
        new_shape = (int(w * reduce), int(h * reduce))
        img_1 = cv2.resize(img_1, new_shape)
        img_2 = cv2.resize(img_2, new_shape)
        img_gt = cv2.resize(img_gt, new_shape)

    # 保留成单通道
    if img_gt.ndim == 3:
        img_gt = img_gt[..., 0][..., np.newaxis]
    img_1, img_2 = _normalize(img_1), _normalize(img_2)

    return img_1, img_2, img_gt


def read_mat_california(dataset_path, reduce=None, select_channel=None):
    """
    img_1 11 channels, img_2 3 channels
    """
    img_path = glob(os.path.join(dataset_path, "*.mat"))[0]
    mat = loadmat(img_path)
    img_1 = np.array(mat["t1_L8_clipped"], dtype=np.float)
    img_2 = np.array(mat["logt2_clipped"], dtype=np.float)
    img_gt = np.array(mat["ROI"], dtype=np.float)[..., np.newaxis]
    if reduce is not None:
        h, w = img_1.shape[:2]
        new_shape = (int(w * reduce), int(h * reduce))
        img_1 = cv2.resize(img_1, new_shape)
        img_2 = cv2.resize(img_2, new_shape)
        img_gt = cv2.resize(img_gt, new_shape)[..., np.newaxis]
    if select_channel is not None:
        img_1 = img_1[..., select_channel[0]]
        img_2 = img_2[..., select_channel[1]]
    return img_1, img_2, img_gt


def read_mat_texas(dataset_path, select_channel=None):
    """
    img_1 7 channels, img_2 10 channels
    """
    img_path = glob(os.path.join(dataset_path, "*.mat"))[0]
    mat = loadmat(img_path)
    img_1 = np.array(mat["t1_L5"], dtype=np.float)
    img_2 = np.array(mat["t2_ALI"], dtype=np.float)

    if select_channel is not None:
        img_1 = img_1[..., select_channel[0]]
        img_2 = img_2[..., select_channel[1]]
    img_gt = np.array(mat["ROI_1"], dtype=np.float)[..., np.newaxis]
    img_1, img_2 = _clip(img_1), _clip(img_2)
    return img_1, img_2, img_gt


def read_tiff(dataset_path: str, city, data_source='rgb', select_channel=None):
    opt_path = r"OSCD/Onera Satellite Change Detection dataset - Images/"
    sar_path = r'multisensor_fusion_CD/S1/'
    label_path = r'OSCD/Onera Satellite Change Detection dataset - Train Labels/'

    if data_source == "tiff":
        t1 = None
        for _file in os.listdir(os.path.join(dataset_path, opt_path, city, 'imgs_1_rect')):
            band = tifffile.imread(os.path.join(dataset_path, opt_path, city, 'imgs_1_rect', _file))[..., np.newaxis]
            if t1 is None:
                t1 = band
            else:
                t1 = np.concatenate((t1, band), axis=2)
        img_1 = t1.astype(np.float)

        sar_file = glob(os.path.join(dataset_path, sar_path, city, 'imgs_2', 'transformed', '*.tif'))
        sar_img = gdal.Open(sar_file[0], gdal.GA_ReadOnly)
        t2 = None
        for band_iter in range(1, sar_img.RasterCount + 1):
            band = sar_img.GetRasterBand(band_iter)
            band_array = band.ReadAsArray()[..., np.newaxis]
            if t2 is None:
                t2 = band_array
            else:
                t2 = np.concatenate((t2, band_array), axis=2)
        img_2 = t2.astype(np.float)

        if select_channel is not None:
            img_1 = img_1[..., select_channel[0]]
            img_2 = img_2[..., select_channel[1]]

        # SAR 的强度数据数据比较特殊,有正有负,全部平移为正值
        min_tmp = np.min(img_2, axis=0)
        min_tmp = np.min(min_tmp, axis=0)
        for i, v in enumerate(min_tmp):
            img_2[..., i] = img_2[..., i] - v

        # img_gt的像素语义:1表示未改变,2表示改变
        img_gt = tifffile.imread(os.path.join(dataset_path, label_path, city, 'cm', city + '-cm.tif'))[..., np.newaxis]
        img_gt = img_gt.astype(np.float) - 1
        img_1, img_2 = _clip(img_1), _clip(img_2)

    elif data_source == "rgb":
        img_2 = cv2.imread(os.path.join(dataset_path, opt_path, city, 'pair', 'img1.png')).astype(np.float)
        img_1 = cv2.imread(os.path.join(dataset_path, sar_path, city, 'imgs_2', 'preview', '0.png')).astype(np.float)
        img_gt = cv2.imread(os.path.join(dataset_path, label_path, city, 'cm', 'cm.png')).astype(np.float) // 255

        # 保留成单通道
        if img_gt.ndim == 3:
            img_gt = img_gt[..., 0][..., np.newaxis]
        img_1, img_2 = _normalize(img_1), _normalize(img_2)
    else:
        raise NotImplementedError
    return img_1, img_2, img_gt


def _clip(image):
    """
        Normalize image from R_+ to [-1, 1].

        For each channel, clip any value larger than mu + 3sigma,
        where mu and sigma are the channel mean and standard deviation.
        Scale to [-1, 1] by (2*pixel value)/(max(channel)) - 1

        Input:
            image - (h, w, c) image array in R_+
        Output:
            image - (h, w, c) image array normalized within [-1, 1]
    """
    temp = np.reshape(image, (-1, image.shape[-1]))

    limits = np.mean(temp, 0) + 3.0 * np.std(temp, 0)
    for i, limit in enumerate(limits):
        channel = temp[:, i]
        channel = np.clip(channel, 0, limit)
        ma, mi = np.max(channel), np.min(channel)
        # channel = 2.0 * (channel - mi) / (ma - mi) - 1
        channel = 2.0 * (channel / ma) - 1
        temp[:, i] = channel

    return np.reshape(temp, image.shape)


def _normalize(image):
    """
    Normlize to [-1,1] in every channel
    """
    temp = np.reshape(image, (-1, image.shape[-1]))

    for i in range(image.shape[-1]):
        channel = temp[:, i]

        ma, mi = np.max(channel), np.min(channel)
        channel = 2 * (channel - mi) / (ma - mi) - 1
        temp[:, i] = channel

    return np.reshape(temp, image.shape)


if __name__ == "__main__":
    def save_bmp(img):
        img = (_normalize(img) + 1) / 2 * 255
        img = img.astype(np.uint8)
        return img


    cal_1, cal_2, cal_gt = \
        read_mat_california('../../../data/optsar/California', 0.25, ([1, 2, 3], [0, 1, 2]))
    cal_1 = save_bmp(cal_1)
    cal_2 = save_bmp(cal_2)
    cal_gt = cal_gt.astype(np.uint8) * 255

    # cv2.imwrite('../../../data/optsar/California/California_1.bmp', cal_1)
    # cv2.imwrite('../../../data/optsar/California/California_2.bmp', cal_2)
    # cv2.imwrite('../../../data/optsar/California/California_gt.bmp', cal_gt)

    cv2.imwrite('../../../data/optsar/California_4/California_1.bmp', cal_1)
    cv2.imwrite('../../../data/optsar/California_4/California_2.bmp', cal_2)
    cv2.imwrite('../../../data/optsar/California_4/California_gt.bmp', cal_gt)

    # tex_1, tex_2, tex_gt = \
    #     read_mat_texas('../../../data/optsar/Texas', ([1, 2, 3], [3, 4, 7]))
    # tex_1 = save_bmp(tex_1)
    # tex_2 = save_bmp(tex_2)
    # tex_gt = tex_gt.astype(np.uint8) * 255
    # cv2.imwrite('../../../data/optsar/Texas/Texas_1.bmp', tex_1)
    # cv2.imwrite('../../../data/optsar/Texas/Texas_2.bmp', tex_2)
    # cv2.imwrite('../../../data/optsar/Texas/Texas_gt.bmp', tex_gt)
