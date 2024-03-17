import os
import random
import time
from functools import partial

import torch
import numpy as np

import cv2
import skimage.filters
import torch.nn.functional as F
from skimage.transform import rescale
from torch import nn

from clip import clip
from scripts.util import _norm_one
from scripts.metric import confusion_map
from scripts.threshold import thres
from scripts.util import save_image as save_img


def validation(save_path, model, dataset, epoch, device, save_image=partial(save_img, pkl=True)):
    """
    Validation process used for inference and obtain detection result.
    Args:
        save_path (str): path to save validation results
        model: trained model
        dataset: CCDataset instance used for inference
        epoch: not used
        device: gpu or cpu to inference
        save_image: function to save image, if you don't want to save image, assign to a void function
    Returns:
        res: change map as prior for next iteration or as final change map
        res_3: not used
    """
    a = dataset.tensor_img_1.to(device)
    b = dataset.tensor_img_2.to(device)
    a = torch.unsqueeze(a, 0)
    b = torch.unsqueeze(b, 0)

    # scale为2^n,根据网络下采样的次数决定
    scale = 4
    t_h = (scale - a.size()[-1] % scale)
    t_w = (scale - b.size()[-2] % scale)
    a = F.pad(a, (t_h, 0, t_w, 0), 'reflect')
    b = F.pad(b, (t_h, 0, t_w, 0), 'reflect')

    pre1, enc = model(a, b)
    pre1 = F.softmax(pre1, dim=1)
    pre = pre1

    pre = pre[:, :,  t_w:, t_h:]
    enc = enc[:, :,  t_w:, t_h:]

    cosine_loss = nn.CosineSimilarity(dim=0)

    pre = pre.cpu().numpy().squeeze()
    enc = enc.squeeze()

    path = os.path.join(save_path, f"epoch_{epoch}_img")
    os.makedirs(path, exist_ok=True)

    """--------------------------------------------------------------------------------"""

    # save_image(os.path.join(path, "pre_1.png"), pre1[1])
    save_image(os.path.join(path, "pre.png"), pre[1])

    res_tmp = pre[1]
    thr1 = 0.5
    thr2 = 0.5
    # 具有三个类别的变化信息图
    res_3 = np.where(res_tmp > thr1, 1, 0)
    res_3 = np.where(res_tmp > thr2, 2, res_3)
    save_image(os.path.join(path, "res_3.png"), res_3)

    res = pre[1] >= pre[0]
    # 返回的prior
    res = res.astype(np.int)

    save_image(os.path.join(path, "res.png"), res)
    con_map, metrics0 = confusion_map(res[None, ...], dataset.img_gt)
    save_image(os.path.join(path, "res_con.png"), con_map)

    thre = thres(pre[1])
    save_image(os.path.join(path, "thre.png"), thre)
    con_map, metrics1 = confusion_map(thre[None, ...] / 255, dataset.img_gt)
    save_image(os.path.join(path, "thre_con.png"), con_map)
    # fix_thre = np.where(pre[1] > 0.8, 1, 0)
    # save_image(os.path.join(path, "fix_thre.png"), fix_thre)
    """-------------------------------------------------------------------------------"""

    proto_changed, proto_unchanged = dataset.prototype.inference(enc.detach())
    proto_changed = proto_changed.cpu().numpy()
    proto_unchanged = proto_unchanged.cpu().numpy()

    proto_changed = _norm_one(proto_changed)
    proto_unchanged = _norm_one(proto_unchanged)

    save_image(os.path.join(path, "proto_changed.png"), proto_changed)
    save_image(os.path.join(path, "proto_unchanged.png"), proto_unchanged)

    """-------------------------------------------------------------------------------------------"""
    masked_changed = np.where(proto_changed > 0.5, 0, 1)
    weighted_unchanged = _norm_one(np.where(masked_changed == 1,
                                            proto_unchanged + (1 - proto_changed) * 0.99, proto_unchanged))
    save_image(os.path.join(path, "weighted_unchanged.png"), weighted_unchanged)
    proto_unchanged = weighted_unchanged
    """-------------------------------------------------------------------------------------------"""

    """------------------------------prototype vector-------------------------------"""
    proto_changed = clip(proto_changed).squeeze()
    proto_unchanged = clip(1 - proto_unchanged).squeeze()

    save_image(os.path.join(path, "proto_changed_clip.png"), proto_changed)
    save_image(os.path.join(path, "proto_unchanged_clip.png"), proto_unchanged)

    proto_changed = cv2.resize(proto_changed, dsize=(res.shape[1], res.shape[0]))
    proto_unchanged = cv2.resize(proto_unchanged, dsize=(res.shape[1], res.shape[0]))

    fusion = 0.7 * pre[1] + 0.2 * proto_changed + 0.1 * proto_unchanged
    # fusion = pre[1] * proto_changed * proto_unchanged
    # fusion = 0.7 * pre[1] + 0.3 * proto_changed
    # fusion = DS_evidence(res, proto_changed, proto_unchanged)
    # fusion = clip(fusion[None, ...]).squeeze()
    save_image(os.path.join(path, "fusion.png"), fusion)

    proto_changed = (proto_changed > 0.5).astype(np.int)
    proto_unchanged = (proto_unchanged > 0.5).astype(np.int)

    fusion_vote = proto_changed + proto_unchanged + res
    fusion_vote = (fusion_vote == 3).astype(np.int)
    save_image(os.path.join(path, "fusion_vote.png"), fusion_vote)
    con_map, metrics4 = confusion_map(fusion_vote[None, ...], dataset.img_gt)
    save_image(os.path.join(path, "fusion_vote_con.png"), con_map)

    fusion_res = (fusion > 0.5).astype(np.int)
    save_image(os.path.join(path, "fusion_bin.png"), fusion_res)
    con_map, metrics2 = confusion_map(fusion_res[None, ...], dataset.img_gt)
    save_image(os.path.join(path, "fusion_bin_con.png"), con_map)

    thre = thres(fusion)
    save_image(os.path.join(path, "fusion_thr.png"), thre)
    con_map, metrics3 = confusion_map(thre[None, ...] / 255, dataset.img_gt)
    save_image(os.path.join(path, "fusion_thr_con.png"), con_map)

    """-----------------------------------------------------------------------------"""

    with open(os.path.join(path, "metrics.txt"), 'w') as f:
        f.writelines(f"res:\n"
                     f"  kappa: {metrics0[0]}, recall: {metrics0[1]}, "
                     f"precision: {metrics0[2]}, overall_accuracy: {metrics0[3]}\n")
        # f.writelines(f"res_thr:\n"
        #              f"  kappa: {metrics1[0]}, recall: {metrics1[1]}, "
        #              f"precision: {metrics1[2]}, overall_accuracy: {metrics1[3]}\n")
        f.writelines(f"fusion:\n"
                     f"  kappa: {metrics2[0]}, recall: {metrics2[1]}, "
                     f"precision: {metrics2[2]}, overall_accuracy: {metrics2[3]}\n")
        # f.writelines(f"fusion_thr:\n"
        #              f"  kappa: {metrics3[0]}, recall: {metrics3[1]}, "
        #              f"precision: {metrics3[2]}, overall_accuracy: {metrics3[3]}\n")
        f.writelines(f"fusion_vote:\n"
                     f"  kappa: {metrics4[0]}, recall: {metrics4[1]}, "
                     f"precision: {metrics4[2]}, overall_accuracy: {metrics4[3]}\n")

    return res, res_3





