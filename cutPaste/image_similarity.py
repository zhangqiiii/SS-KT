"""
Several method to measure similarity of two images

Note: img1 and img1 are all range in [-1, 1]
"""

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity
from skimage import exposure


# 计算图片的余弦距离
def get_cosine_similarity(img1, img2):
    img1 = img1.reshape((1, -1))
    img2 = img2.reshape((1, -1))
    res = cosine_similarity(img1, img2)
    return res


def get_hist_similarity(img1, img2):
    def calculate(img1, img2):
        # 计算单通道的直方图的相似值
        hist1 = exposure.histogram(img1, 100)[0]
        hist2 = exposure.histogram(img2, 100)[0]
        # 计算直方图的重合度
        degree = 0
        for i in range(len(hist1)):
            if hist1[i] != hist2[i]:
                degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
            else:
                degree = degree + 1
        degree = degree / len(hist1)
        return degree
    # 分离通道，再计算每个通道的相似值
    if img1.ndim > 2:
        ndim = img1.shape[0]
        sub_image1 = np.split(img1, img1.shape[0], 0)
        sub_image2 = np.split(img2, img2.shape[0], 0)
    else:
        ndim = 1
        sub_image1 = img1
        sub_image2 = img2
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / ndim
    return sub_data


def get_SSIM(img1, img2):
    if img1.ndim > 2:
        img1 = img1.transpose((1, 2, 0))
        img2 = img2.transpose((1, 2, 0))
        mch = True
    else:
        mch = False
    ssim_score = structural_similarity(img1, img2, win_size=15, data_range=2, multichannel=mch)
    return ssim_score


if __name__ == "__main__":
    img1_ = np.empty((12, 12), dtype=np.uint8)
    img2_ = np.empty((12, 12), dtype=np.uint8)
    get_cosine_similarity(img1_, img2_)
    print(get_hist_similarity(img1_, img2_))
    print(get_SSIM(img1_, img2_))
