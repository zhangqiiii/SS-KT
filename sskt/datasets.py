import random
import torch
import numpy as np

import copy
import itertools
import os

from skimage.transform import resize
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from prototype import ProtoType
from cutPaste.curve_form import irregular_circle
from scripts.util import save_image
from cutPaste.image_classify import *
from cutPaste.image_similarity import *
from scripts.read_data import *


def cal_mean_std(img: np.ndarray):
    tmp = img.reshape((img.shape[0], -1))
    mean = np.mean(tmp, axis=-1)
    std = np.std(tmp, axis=-1)
    return mean, std


def cal_patch_info(length):
    """
        计算数据集的patch_size和patch_stride,patch_size为离(短边/5)最近的2^n,
        patch_stride=patch_size/4
    Returns:
        patch_size, patch_stride
    """
    anchor = length // 5
    old_v = 1
    for i in range(7):
        new_v = 2 ** i
        diff1 = anchor - old_v
        diff2 = new_v - anchor
        if diff1 >= 0 and diff2 >= 0:
            p_size = old_v if diff2 > diff1 else new_v
            return p_size, p_size // 4
        old_v = new_v
    return new_v, new_v // 4


def cutmix(cut_img, mix_img):
    """
    Args:
        cut_img: tensor[c, h, w]
        mix_img: tensor[c, h, w]

    Returns:
        tensor[c, h, w]
    """
    c, h, w = cut_img.size()
    half_size = w // 2

    begin_x = random.randint(0, half_size)
    begin_y = random.randint(0, half_size)
    width = half_size
    # width = random.randint(half_size // 2, half_size)
    cut_patch = cut_img[:, begin_y: begin_y + width, begin_x: begin_x + width]

    begin_x = random.randint(0, half_size)
    begin_y = random.randint(0, half_size)
    mix_img[:, begin_y: begin_y + width, begin_x: begin_x + width] = cut_patch
    change_map_tmp = mix_img[-1, begin_y: begin_y + width, begin_x: begin_x + width] + \
                     mix_img[-2, begin_y: begin_y + width, begin_x: begin_x + width]
    mix_img[-2, begin_y: begin_y + width, begin_x: begin_x + width] = torch.where(change_map_tmp >= 1, 1., 0.)
    return mix_img


class CDDatasets(Dataset):
    def __init__(self, patch_size, patch_stride, border_discard, dataset_name, path,
                 is_train=True, median_filter=7, transform=None, cluster_img_index=1,
                 pretrain=False, class_n=None):
        super(Dataset, self).__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        # 边界不完整的patch如何处理？True=丢弃，False=Pad
        self.border_discard = border_discard
        self.dataset_name = dataset_name
        # 原始尺寸（h,w）
        self.original_shape = None
        self.padded_shape = None
        self.median_filter = median_filter
        self.path = os.path.join(path, dataset_name)
        self.transform_option = None
        # 是否允许数据增强的开关
        self.allow_transform = False
        self.pretrain = pretrain
        self.prototype = ProtoType(1, 1)
        # paste区域选择的阈值
        self.paste_thr = 0.8
        # paste区域寻找的次数
        self.find_num = 50
        self.class_n = 3 if class_n is None else class_n

        """-------------------------------------read dataset-----------------------------------"""
        if not os.path.exists(self.path):
            print("Dataset path is not exist!")
        else:
            handle = read_handle[dataset_name]
            img_1, img_2, img_gt = handle["func"](self.path, **handle["args"])

        # 计算适合的 patch_size 和 patch_stride
        self.original_shape = img_1.shape[:2]

        self.channels = (img_1.shape[-1], img_2.shape[-1], img_gt.shape[-1])
        self.img_slice = (slice(0, self.channels[0]), slice(self.channels[0], sum(self.channels[:2])))

        self.img_1 = img_1.transpose((2, 0, 1))
        self.img_2 = img_2.transpose((2, 0, 1))
        self.img_gt = img_gt.transpose((2, 0, 1))

        """-------------------------------------prepare data-----------------------------------"""
        # self.prior_map = 1 - self.prior(self.img_1, self.img_2)
        self.pseudo_gt = np.zeros_like(self.img_gt)
        # 转置为通道分离表示 （h, w, c）=> (c, h, w)
        self.img = np.concatenate((self.img_1, self.img_2), axis=0)
        mean, std = cal_mean_std(self.img)
        self.norm = transforms.Normalize(mean, std)

        """-------------------------------------generate patches-----------------------------------"""
        # 切割为patch
        # img = np.concatenate((self.img, self.img_gt, self.pseudo_gt), axis=0)
        # self.__extract_patches(img)
        self.un_patches = []
        self.un_coord = []
        self.patches = []
        self.coord = []
        self.patch_thr = 1
        self.prior = None
        self.prior_3 = None
        self.class_sample_prob = None

        self.update_prior(np.zeros_like(self.img_gt), init=True)

        self.tensor_img = torch.tensor(self.img)
        self.tensor_img = self.norm(self.tensor_img)
        self.tensor_img_1 = self.tensor_img[:self.channels[0]].float()
        self.tensor_img_2 = self.tensor_img[self.channels[0]:sum(self.channels[:2])].float()

        if transform is not None:
            self.transform_option = transform

        """-------------------------------------generate bank-----------------------------------"""
        self.bank = None
        self.prior_bank = None
        self.cls_img = None
        self.multi_scale = None
        self.cluster_centers = None
        self.cluster_img_index = cluster_img_index
        # 主图
        self.clustered_img = self.img_1 if self.cluster_img_index == 0 else self.img_2
        # 副图
        self.clustered_img_other = self.img_2 if self.cluster_img_index == 0 else self.img_1
        self.gen_bank(self.clustered_img)
        print("bank building finished!")

    def __len__(self):
        return len(self.un_patches)

    def __getitem__(self, item):
        return self.__get_train_item(item)

    def __get_train_item(self, item):
        # 随机拼接
        tmp = self.un_patches[item].copy()
        tmp = torch.tensor(tmp)
        coord = self.un_coord[item]
        patch_cls_img = self.cls_img[coord[0]:coord[1], coord[2]:coord[3]]
        """------------------------------------采样paste patch---------------------------------------"""
        pseudo_gt = tmp[-1]

        i = 0
        # 初始化paste_op和paste_mask
        paste_op = -1
        paste_mask = None
        begin_x_max = 0
        begin_y_max = 0
        width_max = 0
        paste_mask_max = None
        select_ratio = 0
        while i < self.find_num:
            # 随机生成一块区域(起始点x,起始点y, 高, 宽)
            half_size = self.patch_size // 2
            begin_x = random.randint(0, half_size - 1)
            begin_y = random.randint(0, half_size - 1)
            width = random.randint(half_size // 2, half_size)
            # width = half_size

            gen_coord = (begin_y, begin_y + width, begin_x, begin_x + width)
            gen_cls = patch_cls_img[gen_coord[0]:gen_coord[1], gen_coord[2]:gen_coord[3]]

            # 生成粘贴的mask,并判断是不是满足类型阈值
            # 粘贴哪种形状的patch (方形, 圆形, 不规则形状)
            # paste_op = 1
            paste_op = random.randint(0, 1)
            # 方形
            if paste_op == 0:
                paste_mask = None
                paste_mask_ = paste_mask
            # 圆形
            elif paste_op == 1:
                mesh_a = torch.arange(0, width)
                mesh = torch.meshgrid(mesh_a, mesh_a)
                mesh = torch.stack(mesh, dim=0)
                mesh = mesh.float()
                circle_x = (width - 1) // 2
                center_mesh = torch.tensor(np.array([[[circle_x, ]], [[circle_x, ]]]))
                dist = torch.norm(mesh - center_mesh, dim=0)
                paste_mask = np.zeros_like(dist)
                paste_mask[dist <= circle_x] = 1.0
                paste_mask_ = paste_mask
            else:
                paste_mask = irregular_circle()
                paste_mask = resize(paste_mask, (width, width), preserve_range=True)
                paste_mask = np.where(paste_mask < 0.5, 0, 1)

                paste_mask_ = paste_mask if np.sum(paste_mask) != 0 else None

            # paste_mask_ = None
            # 判断所要被粘贴的位置是不是属于同一类
            select_patch = self.patch_class(gen_cls, self.paste_thr, mask=paste_mask_)

            # 记录最大的比例值
            if select_patch[1] > select_ratio:
                begin_x_max = begin_x
                begin_y_max = begin_y
                width_max = width
                paste_mask_max = paste_mask_

            # 如果选中了,覆盖掉保存最大值的几个变量
            if select_patch[0] > -1:
                begin_x_max = begin_x
                begin_y_max = begin_y
                width_max = width
                paste_mask_max = paste_mask_
                break
            # 继续循环
            i += 1

        if i == self.find_num:
            width_max = 0

        # 还原变量名
        begin_x = begin_x_max
        begin_y = begin_y_max
        width = width_max
        paste_mask_ = paste_mask_max

        if width == 0:
            paste_coord = None
        else:
            paste_coord = self.sample_patch(patch_cls_img, gen_coord, self.class_n, mask=paste_mask_)

        # 没有找到满足的paste patch则不粘贴
        if paste_coord is None:
            pass
        else:
            paste_patch = self.clustered_img[:, paste_coord[0]: paste_coord[1],
                          paste_coord[2]: paste_coord[3]]
            paste_patch = resize(paste_patch, (paste_patch.shape[0], width, width))
            paste_patch = torch.tensor(paste_patch)
            slice_tmp = self.img_slice[0] if self.cluster_img_index == 0 else self.img_slice[1]
            # 粘贴矩形patch
            # paste_op = random.randint(0, 1)
            if paste_op == 0:
                tmp[slice_tmp, gen_coord[0]:gen_coord[1], gen_coord[2]: gen_coord[3]] = paste_patch
                pseudo_gt[gen_coord[0]:gen_coord[1], gen_coord[2]: gen_coord[3]] = 1.0
            # 粘贴圆形patch
            elif paste_op == 1:
                paste_mask = torch.tensor(paste_mask)

                tmp_rect = tmp[slice_tmp, gen_coord[0]:gen_coord[1], gen_coord[2]: gen_coord[3]]
                paste_circle = torch.where(paste_mask == 1.0, paste_patch, tmp_rect)
                tmp[slice_tmp, gen_coord[0]:gen_coord[1], gen_coord[2]: gen_coord[3]] = paste_circle

                pseudo_gt[gen_coord[0]:gen_coord[1], gen_coord[2]: gen_coord[3]] = paste_mask
            else:
                # 生成不规则mask
                paste_mask = torch.tensor(paste_mask)
                # 待粘贴区域原来的值
                tmp_ir_circle = tmp[slice_tmp, gen_coord[0]:gen_coord[1], gen_coord[2]: gen_coord[3]]
                # 对矩形内重新赋值
                paste_ir_circle = torch.where(paste_mask == 1.0, paste_patch, tmp_ir_circle)
                tmp[slice_tmp, gen_coord[0]:gen_coord[1], gen_coord[2]: gen_coord[3]] = paste_ir_circle
                pseudo_gt[gen_coord[0]:gen_coord[1], gen_coord[2]: gen_coord[3]] = paste_mask

        """-----------------------------------数据增强(只允许空间变换的增强)------------------------------------"""
        if self.transform_option is not None and self.allow_transform:
            # 随机选择一种策略
            if random.random() > 0.5:
                # 随机选择一种策略
                for i_ in range(2):
                    itran = random.randrange(0, len(self.transform_option))
                    transform_ = self.transform_option[itran]
                    if transform_[1] == 1:
                    # 1 表示对图像进行逐像素变换，且不改变gt
                    #   tmp[:-3] = transform_[0](tmp[:-3])
                        tmp[self.img_slice[0]] = transform_[0](tmp[self.img_slice[0]])
                        tmp[self.img_slice[1]] = transform_[0](tmp[self.img_slice[1]])
                    else:
                        tmp = transform_[0](tmp)

        """------------------------------------------------------------------------------------------------"""
        # 先进行数据增强, 再normalize
        source_tmp = self.norm(tmp[:-3])
        gt = tmp[-3].float()
        patch_prior = tmp[-2].float()
        pseudo_gt = tmp[-1].float()
        t1 = source_tmp[:self.channels[0]].float()
        t2 = source_tmp[self.channels[0]:sum(self.channels[:2])].float()

        """-----------------------------------------cutmix-img--------------------------------------------"""
        mix_img = self.patches[random.randint(0, len(self.patches) - 1)].copy()
        mix_img = torch.tensor(mix_img)
        # mix_img = cutmix(tmp, mix_img)

        if self.transform_option is not None and self.allow_transform:
            # 随机选择一种策略
            if random.random() > 0.5:
                # 随机选择一种策略
                for i_ in range(2):
                    itran = random.randrange(0, len(self.transform_option))
                    transform_ = self.transform_option[itran]
                    if transform_[1] == 1:
                        # 1 表示对图像进行逐像素变换，且不改变gt
                        # tmp[:-3] = transform_[0](tmp[:-3])
                        mix_img[self.img_slice[0]] = transform_[0](mix_img[self.img_slice[0]])
                        mix_img[self.img_slice[1]] = transform_[0](mix_img[self.img_slice[1]])
                    else:
                        mix_img = transform_[0](mix_img)

        source_mix = self.norm(mix_img[:-3])
        gt_mix = mix_img[-3].float()
        patch_prior_mix = mix_img[-2].float()
        pseudo_gt_mix = mix_img[-1].float()
        t1_mix = source_mix[:self.channels[0]].float()
        t2_mix = source_mix[self.channels[0]:sum(self.channels[:2])].float()
        """---------------------------------------------------------------------------------------------------"""


        """---------------------------------------collect---------------------------------------------------"""
        sample = [(t1, t1_mix), (t2, t2_mix), (gt, gt_mix),
                  (patch_prior, patch_prior_mix),
                  (pseudo_gt, pseudo_gt_mix)]

        return sample

    def update_prior(self, prior, prior_3=None, init=False):
        """
        更新prior图，同时需要更新训练集patches和补充prior_bank
        """
        self.prior = prior[None, ...] if prior.ndim == 2 else prior
        img = np.concatenate((self.img, self.img_gt, self.prior, self.pseudo_gt), axis=0)
        self.__extract_patches(img, self.un_patches, self.un_coord, self.patch_thr)
        self.__extract_patches(img, self.patches, self.coord, 1)
        print("patches extraction finish!")
        if not init:
            class_bin = self.cls_img[self.prior[0,...] == 1]
            class_bin = np.bincount(class_bin, minlength=len(self.cluster_centers))
            sum_bin = np.sum(class_bin)
            self.class_sample_prob = class_bin / sum_bin
            # print(self.class_sample_prob)

        if not init:
            # self.prior_bank = copy.deepcopy(self.bank)
            # self.prior_bank = []
            # for i in range(5):
            #     self.prior_bank.append([])
            # self.add_bank()
            pass
        else:
            self.patches = self.un_patches.copy()
            self.coord = self.un_coord.copy()

    def __extract_patches(self, img, patches, coord, patch_thr):
        patches.clear()
        coord.clear()
        """
        提取patch
        :param img:
        :return:
        """
        # offset_x = random.randint(0, self.patch_stride)
        # offset_y = random.randint(0, self.patch_stride)
        offset_x = 0
        offset_y = 0
        c, h, w = img.shape
        # 对图像镜像填充
        if not self.border_discard:
            _, h_pad_num = cal_border_with_pad(h, self.patch_size, self.patch_stride)
            _, w_pad_num = cal_border_with_pad(w, self.patch_size, self.patch_stride)
            img = np.pad(img, ((0, 0), (0, h_pad_num), (0, w_pad_num)), "reflect")
            self.padded_shape = img.shape[1:]
            h, w = self.padded_shape

        for i in range(offset_y, h - self.patch_size + 1, self.patch_stride):
            for j in range(offset_x, w - self.patch_size + 1, self.patch_stride):
                patch = img[:, i:i + self.patch_size, j:j + self.patch_size]
                # patch_prior 参考 update_prior() 中图像拼接时各个通道的意义
                patch_prior = patch[-2]
                area_ratio = np.sum(patch_prior) / (self.patch_size * self.patch_size)
                if area_ratio <= patch_thr:
                    patches.append(patch)
                    coord.append(np.array([i, i + self.patch_size, j, j + self.patch_size]))

    def patch_class_dist(self, patch1, patch2, mask=None):
        """
        计算两个patch之间的的相似度
        """
        equal = patch1 == patch2
        equal = equal.astype(np.int)
        sim = np.sum(equal) / (patch2.shape[0] * patch2.shape[1])
        return sim

    def select_class(self, patch, mask=None):
        """
        计算与给定patch类别中心距离最远的类别
        Args:
            :param patch: 分类图
        Returns:
            类别索引
        """

        def cal_center(bin):
            bin_sim = np.sum(bin)
            center = 0
            for i, cen in enumerate(self.cluster_centers):
                center += bin[i] / bin_sim * cen
            return center

        patch = patch.flatten()
        if mask is not None:
            mask = mask.flatten()
            patch = patch[mask == 1]
        patch_bin = np.bincount(patch, minlength=len(self.cluster_centers))
        cen = cal_center(patch_bin)
        dist = []
        for c_cen in self.cluster_centers:
            tmp_dist = np.linalg.norm(c_cen - cen)
            dist.append(tmp_dist)
        ind_sort = sorted(range(len(dist)), key=lambda k: dist[k], reverse=True)
        return ind_sort

    def patch_class(self, patch, ratio=0.95, mask=None):
        """
        Returns: integer in range(cluster) demonstrate class
                 -1 demonstrate no class
        """
        tmp = patch.reshape((-1,))
        if mask is not None:
            mask = mask.reshape((-1,))
            tmp = tmp[mask == 1]
        bin = np.bincount(tmp)
        loc = np.where(bin == np.max(bin))

        # if len(loc) == 1:
        #     index = loc[0][0]
        #     res = bin[index] / np.sum(bin)
        #     if res > ratio:
        #         return index, res

        index = loc[0][0]
        res = bin[index] / np.sum(bin)
        if res > ratio:
            return index, res
        return -1, res

    def gen_bank(self, which_img):
        """
        generate patch bank
        """
        self.cluster = 5
        # self.cluster = 6
        # self.cls_img, self.cluster_centers = cls_cluster(which_img, '', cluster)
        cluster_pkl = f'cutPaste/cluster_5_filter_{self.median_filter}/{self.dataset_name}_{self.cluster_img_index+1}.pkl'
        cluster_file = open(cluster_pkl, 'rb')
        self.cls_img, self.cluster_centers = pickle.load(cluster_file)
        cluster_file.close()

        self.min_size = self.patch_size // 8
        stride = 2 * self.min_size
        self.multi_scale = [self.min_size, 2 * self.min_size, 4 * self.min_size]
        # self.multi_scale = [2 * self.min_size, 4 * self.min_size]

        """--------------------------初始化嵌套的bank列表---------------------------------------------"""
        self.bank = []
        for i in range(self.cluster):
            self.bank.append([])
        """------------------------------------构建bank--------------------------------------"""

        for i in range((self.original_shape[0] - self.multi_scale[-1]) // stride - 1):
            for j in range((self.original_shape[1] - self.multi_scale[-1]) // stride - 1):
                for k_ind, k in enumerate(self.multi_scale):
                    tmp = self.cls_img[i * stride: i * stride + k, j * stride: j * stride + k]
                    # 判断是否属于某一类
                    index = self.patch_class(tmp, 0.95)[0]
                    if index != -1:
                        # 记录下坐标
                        self.bank[index].append((i * stride, i * stride + k, j * stride, j * stride + k))
        """---------------------------------------------------------------------------------------"""
        self.all_samples = list(itertools.chain(*[self.bank[i] for i in range(self.cluster)]))


    def add_bank(self):
        stride = 2 * self.min_size
        for i in range((self.original_shape[0] - self.multi_scale[-1]) // stride - 1):
            for j in range((self.original_shape[1] - self.multi_scale[-1]) // stride - 1):
                for k_ind, k in enumerate(self.multi_scale):
                    tmp_prior = self.prior[:, i * stride: i * stride + k, j * stride: j * stride + k]
                    tmp_cls = self.cls_img[i * stride: i * stride + k, j * stride: j * stride + k]
                    # 判断是否属于某一类
                    index = self.patch_class(tmp_prior)[0]
                    if index == 1:
                        index = self.select_class(tmp_cls)[-1]
                        # 记录下坐标
                        self.prior_bank[index].append((i * stride, i * stride + k, j * stride, j * stride + k))

    def sample_patch(self, patch_cls_img, coord, num=2, mask=None):
        """
        采样patch
        Args:
            patch_cls_img: 某个patch的分类图
            coord: 生成的被粘贴坐标
            num: 要选择类别的数量
            mask: 需要考虑的mask

        Returns: 返回采样的图像块在bank中的二级索引
        """

        if self.prior_bank is None:
            tmp_bank = self.bank
        else:
            tmp_bank = self.prior_bank

        tmp = patch_cls_img[coord[0]:coord[1], coord[2]:coord[3]]
        # 返回类别的编号
        cls_inds = self.select_class(tmp, mask=mask)
        # 防止没有符合的样本
        cls_num = [len(tmp_bank[i]) for i in cls_inds]
        while sum(cls_num[:num]) == 0:
            # return None
            num += 1

        samples = itertools.chain(*[tmp_bank[i] for i in cls_inds[:num]])
        # if self.class_sample_prob is None:
        #     sample_prob = None
        # else:
        #     sample_prob = itertools.chain(*[[self.class_sample_prob[i]] * len(tmp_bank[i]) for i in cls_inds[:num]])
        # coord = random.choices(list(samples), weights=sample_prob, k=1)[0]
        coord = random.choice(list(samples))
        return coord


def cal_border_with_pad(num, patch_size, patch_stride):
    """
    Calculate how many pactches a side can split.
    :param num:
    :param patch_size:
    :param patch_stride:
    :return: patch_num, need_pad_num
    """
    residual = (num - patch_size) % patch_stride
    # 正好被完全切分
    if residual == 0:
        return (num - patch_size) / patch_stride + 1, 0
    # 需要填充
    else:
        return (num - patch_size) // patch_stride + 1, patch_size - residual


read_handle = {
    "California": {
        "func": read_mat_california,
        "args": {
            "reduce": 0.5,
            # "select_channel": ([1, 2, 3], [0, 1, 2])
        }
    },
    "Texas": {
        "func": read_mat_texas,
        "args": {
            # "select_channel": ([1, 2, 3], [3, 4, 7])
        }
    },
    "city": {
        "func": read_tiff,
        "args": {
            # "select_channel": ([1, 2, 3], [0, 1, 1])
        }
    },
    "Italy": {"func": read_RGB, "args": {}},
    "shuguang": {"func": read_RGB, "args": {"reduce": 1.0}},
    # "shuguang_cut": {"func": read_RGB, "args": {"reduce": 1.0}},
    "wuhan": {"func": read_RGB, "args": {}},
    "airport": {"func": read_RGB, "args": {}},
    "yellow": {"func": read_RGB, "args": {}},
    # "yellow_cut": {"func": read_RGB, "args": {}},
    "farmland": {"func": read_RGB, "args": {}},
    "Gloucester1": {"func": read_RGB, "args": {
        "reduce": 1
    }},
    "Gloucester2": {"func": read_RGB, "args": {
        "reduce": 0.3
    }},
}

if __name__ == "__main__":
    # a = CDDatasets(64, 32, True, 'wuhan', '../data/optsar', is_train=False, transform=None)
    for dataset_ in ['Italy', 'yellow', 'shuguang', 'Texas', 'California']:
        handle = read_handle[dataset_]
        img_1, img_2, img_gt = handle["func"](os.path.join('../data/optsar', dataset_), **handle["args"])
        ratio = np.sum(img_gt)/(img_gt.shape[0]*img_gt.shape[1])
        print(dataset_, ratio)
