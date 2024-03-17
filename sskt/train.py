import random
import time

from scripts.funcs import *

import math
import os.path
from datetime import datetime

from torch import nn
from torch.optim import SGD, Adam
from tqdm import tqdm

from sskt.datasets import CDDatasets
from sskt.loss import plot_loss
from cutPaste.loss import ContrastLoss
from sskt.network import TUNet_ld, MaskPooling, TUNet_wo_skip
from sskt.validate import validation
from scripts.transforms import select_transform
from scripts.analyze_metric import collect_metric
from scripts.plot_curve import plot_pull_loss
from scripts.weight_init import init_weights
from scripts.util import save_image
from scripts.funcs import get_torch_device_smart, seed_everything
import warnings

warnings.filterwarnings("ignore")


patch_setting = {
    'Italy': [(64, 16), 0],
    'yellow': [(48, 48), 1],
    'yellow_cut': [(48, 48), 1],
    'California': [(128, 128), 1],
    'Texas': [(128, 64), 1],
    'shuguang': [(64, 32), 1],
    'shuguang_cut': [(64, 32), 1],
}

median_setting = {
    'Italy': 7,
    'yellow': 5,
    'yellow_cut': 5,
    'California': 1,
    'Texas': 13,
    'shuguang': 13,
    'shuguang_cut': 13,
}


def null_save_img(path, img):
    pass


def train(dataset_name='California', exp_path=''):
    seed_everything(8888)
    print(f"{dataset_name} begin training!")
    device = get_torch_device_smart()
    time_str = datetime.now().strftime("%m%d%H%M%S")

    # 多个数据增强策略,仅允许空间变换,不允许改变像素值
    # 元组第二个元素表示是否仅变换原图像(而不变换标签)
    transform = [
        (select_transform([{'name': 'random-flip-vertical', 'p': 0.5}]), 0),
        (select_transform([{'name': 'random-flip-horizontal', 'p': 0.5}]), 0),
        (select_transform([{'name': 'center-crop', 'size': patch_setting[dataset_name][0][0] / 2},
                           {'name': 'resize', 'size': patch_setting[dataset_name][0][0]}]), 0),
        (select_transform([{'name': 'gaussian-blur', 'kernel_size': 5}]), 1),
    ]

    dataset = CDDatasets(*patch_setting[dataset_name][0], True, dataset_name,
                         'data', is_train=False,
                         median_filter=median_setting[dataset_name],
                         transform=transform,
                         cluster_img_index=patch_setting[dataset_name][1])
                         # class_n=class_n)
    channel = dataset.channels
    dataloader = torch.utils.data.DataLoader(dataset, 16, shuffle=True, num_workers=4)

    # 保存文件的路径
    save_dir = f"sskt_result/{exp_path}train_{dataset_name}_{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    f_path = os.path.join(save_dir, 'train.log')
    f_log = open(f_path, 'w')

    network_layer = [32, 64]
    # epoch 参数
    max_epoch = 100
    max_epoch_ = max_epoch
    val_interval = 5
    lr = 2e-3
    lr_decay = [30, 50, 80]
    lr_decay_w = 0.1
    dataset.find_num = find_num


    # model = UNet(channel[0] + channel[1], 2, [16, 32, 64, 128, 256], 'linear')
    # model = TUNet(channel[0], channel[1], 2, [16, 32, 64, 128, 256])
    model = TUNet_ld(channel[0], channel[1], 2, network_layer)
    model.to(device)

    init_weights(model, {'name': 'normal', 'init_gain': 0.02})

    optim = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, lr_decay, lr_decay_w)
    CE_loss = nn.CrossEntropyLoss(reduction='none')
    cosine_loss = nn.CosineSimilarity(dim=0)

    """----------------------------------------------Train------------------------------------------------"""
    dataset.pretrain = False
    print("Training")
    mask_pooling = MaskPooling()
    dataset.changed_vector = None
    dataset.unchanged_vector = None
    for epoch in range(1, max_epoch_ + 1):
        alpha = epoch / max_epoch
        if epoch > 10:
            dataset.allow_transform = True
        dataset.paste_thr = thr_min + epoch/max_epoch * (thr_max - thr_min)
        for i, batch in enumerate(dataloader):
            a = batch[0][0].to(device)
            b = batch[1][0].to(device)
            pseudo_gt = batch[-1][0].to(device).long()
            batch_prior = batch[-2][0].to(device)
            # mistake has been fixed
            prior = batch[4][0].to(device).unsqueeze(1)
            pre1_ori, enc_res = model(a, b)

            """------------------------------加入已经预测的信息-------------------------------"""
            unchanged_mask = torch.logical_not(batch_prior.int() | pseudo_gt.int())
            unchanged_mask = unchanged_mask.float()
            changed_mask = pseudo_gt.int() & (torch.logical_not(batch_prior.int()))
            changed_mask = changed_mask.float()
            prior_gt = torch.logical_not(unchanged_mask).long()

            # 原型向量
            paste_unchanged_vector, paste_changed_vector = mask_pooling(enc_res, prior_gt.float())
            dataset.prototype.update(enc_res.detach(), prior_gt.detach())

            prior_weight = 0.01
            if epoch // val_interval == 0:
                prior_weight += 0.2 * epoch / max_epoch
            if epoch <= 0:
                loss = CE_loss(pre1_ori, batch_prior.long())
            else:
                loss = CE_loss(pre1_ori, prior_gt)
            loss = (1 - (1 - prior_weight) * prior) * loss
            loss = loss.mean()

            """----------------------------------cut-mix数据-----------------------------------"""

            a = batch[0][1].to(device)
            b = batch[1][1].to(device)
            prior_gt = batch[-2][1].to(device).long()
            pre1_mix, enc_res = model(a, b)

            mix_unchanged_vector, mix_changed_vector = mask_pooling(enc_res, prior_gt.float())

            loss_mix = CE_loss(pre1_mix, prior_gt).mean()

            # Visual intermediate result.
            # if epoch == 41:
            #     os.makedirs(os.path.join(save_dir, "vis_mix"), exist_ok=True)
            #     for j in range(a.size()[0]):
            #         save_image(os.path.join(save_dir, "vis_mix", f"a_ori_{j}.png"), a[j].cpu().numpy())
            #         save_image(os.path.join(save_dir, "vis_mix", f"b_ori_{j}.png"), b[j].cpu().numpy())
            #         save_image(os.path.join(save_dir, "vis_mix", f"gt_ori_{j}.png"), prior_gt[j].cpu().numpy())

            """------------------------------------维护原型向量--------------------------------------"""
            dataset.prototype.update(enc_res.detach(), prior_gt.detach())

            """------------------------------------------------------------------------------------------"""
            unch_sim = -torch.abs(cosine_loss(paste_unchanged_vector, mix_unchanged_vector))
            paste_dissim = torch.abs(cosine_loss(paste_changed_vector, paste_unchanged_vector))
            ch_sim = -torch.abs(cosine_loss(paste_changed_vector, mix_changed_vector))
            mix_dissim = torch.abs(cosine_loss(mix_changed_vector, mix_unchanged_vector))

            unch_sim = 0 if torch.isnan(unch_sim) else unch_sim
            paste_dissim = 0 if torch.isnan(paste_dissim) else paste_dissim
            ch_sim = 0 if torch.isnan(ch_sim) else ch_sim
            mix_dissim = 0 if torch.isnan(mix_dissim) else mix_dissim

            loss_mix_integ = 2 * loss_mix + unch_sim + ch_sim + mix_dissim
            loss = (1 - alpha) * loss + 0.5 * alpha * loss_mix_integ + 1 * paste_dissim

            """---------------------------------------------------------------------------------------"""
            log_line = f"Epoch:{epoch} ({i}/{len(dataloader)})  loss: {loss}  loss: {loss}"
            f_log.write(log_line + '\n')
            print(log_line)

            loss.backward()
            optim.step()
            optim.zero_grad()

        scheduler.step()

        if epoch % val_interval == 0 or epoch == max_epoch:
            with torch.no_grad():
                model.eval()
                # save image or not
                prior, prior_3 = validation(save_dir, model, dataset, epoch, device)
                # prior, prior_3 = validation(save_dir, model, dataset, epoch, device, save_image=null_save_img)

                model.train()
            # 调节patch_thr的策略
            patch_thr = 0.01
            dataset.patch_thr = patch_thr
            print(f'patch_thr: {patch_thr}')
            dataset.update_prior(prior, prior_3, init=False)
            save_image(os.path.join(save_dir, f'epoch_{epoch}_img', 'prior.png'), prior)

    f_log.close()
    collect_metric(save_dir)
    plot_loss(os.path.join(save_dir, 'train.log'), ('loss', 'mix_loss'))


if __name__ == "__main__":
    find_num = 150
    thr_min = 0.80
    thr_max = 0.85
    # exp_dataset = ['Italy', 'yellow', 'shuguang', 'Texas', 'California']
    exp_dataset = ['yellow_cut', 'shuguang_cut']
    # exp_dataset *= 10

    for d in exp_dataset:
        seed_everything(8888)
        t_start = time.time()
        train(d, f'result/')
        print(f"dataset {d} compute time: {time.time() - t_start}")



