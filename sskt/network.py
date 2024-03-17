import torch
from torch import nn
import torch.nn.functional as F

from cutPaste.triple_unet import DownSampling, UpSampling, Conv, UpSampling_wo_skip


class TUNet_wo_skip(nn.Module):
    def __init__(self, ch_in_1, ch_in_2, ch_out=2, ch_list=None):
        super(TUNet_wo_skip, self).__init__()
        if ch_list is None:
            ch_list = [16, 32]

        self.down_1 = nn.ModuleList([Conv(ch_in_1, ch_list[0])])
        self.down_2 = nn.ModuleList([Conv(ch_in_2, ch_list[0])])
        self.up = nn.ModuleList([])

        for i in range(1, len(ch_list)):
            self.down_1.append(DownSampling(ch_list[i - 1]))
            # 多加了一块Conv
            self.down_1.append(nn.Sequential(Conv(ch_list[i - 1], ch_list[i]),
                                             Conv(ch_list[i], ch_list[i])))
            self.down_2.append(DownSampling(ch_list[i - 1]))
            self.down_2.append(nn.Sequential(Conv(ch_list[i - 1], ch_list[i]),
                                             Conv(ch_list[i], ch_list[i])))

        for i in range(len(ch_list) - 1, 0, -1):
            self.up.append(UpSampling_wo_skip(ch_list[i]))
            self.up.append(Conv(ch_list[i - 1], ch_list[i - 1]))

        self.pred1 = nn.Sequential(
            nn.Conv2d(ch_list[0], ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU()
        )

    def forward(self, x1, x2):
        for m in self.down_1:
            x1 = m(x1)

        for m in self.down_2:
            x2 = m(x2)

        # 编码后的结果
        encoder_res = torch.abs(x1 - x2)
        out = encoder_res

        for m in self.up:
            out = m(out)

        return self.pred1(out), encoder_res


class TUNet_ld(nn.Module):
    """
    ld表示less-down, 下采样次数更少的TUNet
    """

    def __init__(self, ch_in_1, ch_in_2, ch_out=2, ch_list=None):
        super(TUNet_ld, self).__init__()
        if ch_list is None:
            ch_list = [16, 32]

        # 4次下采样
        self.down_1 = nn.ModuleList([Conv(ch_in_1, ch_list[0])])
        self.down_2 = nn.ModuleList([Conv(ch_in_2, ch_list[0])])
        self.up = nn.ModuleList([])

        for i in range(1, len(ch_list)):
            self.down_1.append(DownSampling(ch_list[i - 1]))
            # 多加了一块Conv
            self.down_1.append(nn.Sequential(Conv(ch_list[i - 1], ch_list[i]),
                                             Conv(ch_list[i], ch_list[i])))
            self.down_2.append(DownSampling(ch_list[i - 1]))
            self.down_2.append(nn.Sequential(Conv(ch_list[i - 1], ch_list[i]),
                                             Conv(ch_list[i], ch_list[i])))

        for i in range(len(ch_list) - 1, 0, -1):
            # self.up.append(UpSampling(ch_list[i]))
            # self.up.append(Conv(ch_list[i], ch_list[i - 1]))
            self.up.append(UpSampling(ch_list[i], ch_list[i-1]))
            self.up.append(Conv(ch_list[i-1] * 2, ch_list[i - 1]))

        self.pred1 = nn.Sequential(
            nn.Conv2d(ch_list[0], ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU()
        )
        # self.pred2 = nn.Sequential(
        #     nn.Conv2d(ch_list[1], ch_out, 3, 1, 1),
        #     nn.BatchNorm2d(ch_out),
        #     # nn.ReLU()
        # )

    def forward(self, x1, x2):
        self.down_res_1 = [self.down_1[0](x1), ]
        self.down_res_2 = [self.down_2[0](x2), ]

        out = []

        for i in range(1, len(self.down_1), 2):
            self.down_res_1.append(
                self.down_1[i + 1](
                    self.down_1[i](self.down_res_1[-1])
                )
            )
            self.down_res_2.append(
                self.down_2[i + 1](
                    self.down_2[i](self.down_res_2[-1])
                )
            )

        # 编码后的结果
        encoder_res = torch.abs(self.down_res_1[-1] - self.down_res_2[-1])

        out.append(
            self.up[1](
                self.up[0](encoder_res,
                           torch.abs(self.down_res_1[-2] - self.down_res_2[-2]))
            )
        )

        for i in range(2, len(self.up), 2):
            ind = i // 2 + 2
            out.append(
                self.up[i + 1](
                    self.up[i](out[-1], torch.abs(self.down_res_1[-ind] - self.down_res_2[-ind]))
                )
            )

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.pred1(out[-1]), encoder_res


class UNet_ld(nn.Module):
    """
    ld表示less-down, 下采样次数更少的TUNet
    """

    def __init__(self, ch_in_1, ch_in_2, ch_out=2, ch_list=None):
        super(UNet_ld, self).__init__()
        if ch_list is None:
            ch_list = [16, 32]

        # 4次下采样
        self.down_1 = nn.ModuleList([Conv(ch_in_1 + ch_in_2, ch_list[0])])
        self.up = nn.ModuleList([])

        for i in range(1, len(ch_list)):
            self.down_1.append(DownSampling(ch_list[i - 1]))
            # 多加了一块Conv
            self.down_1.append(nn.Sequential(Conv(ch_list[i - 1], ch_list[i]),
                                             Conv(ch_list[i], ch_list[i])))

        for i in range(len(ch_list) - 1, 0, -1):
            self.up.append(UpSampling(ch_list[i], ch_list[i-1]))
            self.up.append(Conv(ch_list[i-1] * 2, ch_list[i - 1]))

        self.pred1 = nn.Sequential(
            nn.Conv2d(ch_list[0], ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU()
        )

    def forward(self, x1, x2):
        self.down_res_1 = [self.down_1[0](torch.cat([x1, x2], dim=1))]

        out = []

        for i in range(1, len(self.down_1), 2):
            self.down_res_1.append(
                self.down_1[i + 1](
                    self.down_1[i](self.down_res_1[-1])
                )
            )

        # 编码后的结果
        encoder_res = self.down_res_1[-1]

        out.append(
            self.up[1](
                self.up[0](encoder_res, self.down_res_1[-2])
            )
        )

        for i in range(2, len(self.up), 2):
            ind = i // 2 + 2
            out.append(
                self.up[i + 1](
                    self.up[i](out[-1], torch.abs(self.down_res_1[-ind] - self.down_res_2[-ind]))
                )
            )

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.pred1(out[-1]), encoder_res


class MaskPooling(nn.Module):
    def forward(self, x, mask):
        """
        Args:
            x: tensor[B, C, H, W]
            mask: tensor[B, H, W]

        Returns:
            Two tensors stand for the unchanged and changed.
        """
        B, C, H, W = x.size()
        mask = mask.unsqueeze(1)
        if x.size()[2:] != mask.size()[1:]:
            # mask = F.interpolate(mask, size=x.size()[2:], mode="nearest")
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear")
            mask = torch.where(mask > 0.5, 1., 0.)
        x = x.permute(0, 2, 3, 1)
        mask = mask.permute(0, 2, 3, 1)
        ch = torch.masked_select(x, mask.bool()).reshape((-1, C)).mean(dim=0)
        unch = torch.masked_select(x, (1 - mask).bool()).reshape((-1, C)).mean(dim=0)
        return unch, ch


if __name__ == "__main__":
    mask_pool = MaskPooling()
    mask_test = torch.randint(0, 2, (3, 8, 8)).float()
    x_test = torch.randn((3, 4, 4, 4)).float()
    zero_test, one_test = mask_pool(x_test, mask_test)
    print(zero_test, one_test)

