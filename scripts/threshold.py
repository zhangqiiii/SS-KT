import os.path
from numbers import Number

import cv2
import numpy as np
import skimage.filters

from scripts.cluster_util import otsu
from scripts.util import _norm_one
from scripts.util import _normalize, save_image


def thres(img):
    """
    Args:
        img: (h, w)
    """
    bcm = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    thre = otsu(img.reshape(1, -1))
    bcm[img > thre] = 255.0
    return bcm


def otsu_thes(img_path, file1, file2, file_diff):

    # x = cv2.imread(os.path.join(img_path, file1)).astype(np.float)
    # y = cv2.imread(os.path.join(img_path, file2)).astype(np.float)
    # diff = cv2.imread(os.path.join(img_path, file_diff), cv2.IMREAD_GRAYSCALE)[..., None].astype(np.float)
    # x = _norm_one(x)
    # y = _norm_one(y)
    # diff = _norm_one(diff)
    # img = filtering(x, y, diff)

    img = cv2.imread(os.path.join(img_path, file_diff), cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    bcm = thres(img)
    save_image("change.jpg", bcm)


if __name__ == "__main__":
    # otsu_thes("Texas_0708161745/model_d(optsar)_m(cyclegan)_d(basic)_g(vit-unet)_cyclegan_vit-unet-6-none-lsgan-paper-cycle_high-256/gen_image/epoch_20_img/diff_b.jpg")
    otsu_thes("/home/omnisky/changeDetection/uvcgan-main/tmp_img_3/epoch_30_img",
              "ffeat_a.jpg", "ffeat_b.jpg", "diff_ffeat.jpg")


    # img1 = cv2.imread("/home/omnisky/changeDetection/uvcgan-main/tmp_img_2/epoch_30_img/w.jpg",
    #                   cv2.IMREAD_GRAYSCALE).astype(np.float)
    # img2 = cv2.imread("/home/omnisky/changeDetection/uvcgan-main/tmp_img_2/epoch_30_img/diff_spec.jpg",
    #                   cv2.IMREAD_GRAYSCALE).astype(np.float)
    # # img2 = skimage.filters.gaussian(img2)
    # img = thres(img2)
    # save_image("img.jpg", img)
