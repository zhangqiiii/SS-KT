"""
参考代码见 https://blog.csdn.net/weixin_45074568/article/details/114901600
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


from scripts.select import get_activ_layer


# 基本卷积块
class Conv(nn.Module):
    def __init__(self, ch_in, ch_out, layer_act='relu'):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            # 防止过拟合
            # nn.Dropout(0.1),
            get_activ_layer(layer_act),

            nn.Dropout(0.5),

            nn.Conv2d(ch_out, ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            # 防止过拟合
            get_activ_layer(layer_act),
            # nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, ch, layer_act='relu'):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(ch, ch, 3, 2, 1),
            nn.BatchNorm2d(ch),
            get_activ_layer(layer_act)
        )

    def forward(self, x):
        return self.Down(x)


class UpSampling(nn.Module):

    def __init__(self, ch, ch_out=None, layer_act='relu'):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        if ch_out is None:
            ch_out = ch // 2
        self.Up = nn.Sequential(
            nn.Conv2d(ch, ch_out, 1, 1),
            nn.BatchNorm2d(ch_out),
            get_activ_layer(layer_act)
        )

    def forward(self, x, r):
        # 使用邻近插值进行上采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 上采样模块,没有skip连接
class UpSampling_wo_skip(nn.Module):

    def __init__(self, ch, layer_act='relu'):
        super(UpSampling_wo_skip, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 1, 1),
            nn.BatchNorm2d(ch // 2),
            get_activ_layer(layer_act)
        )

    def forward(self, x):
        # 使用邻近插值进行上采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return x

# 主干网络
class TUNet(nn.Module):

    def __init__(self, ch_in_1, ch_in_2, ch_out=2, ch_list=None):
        super(TUNet, self).__init__()
        if ch_list is None:
            ch_list = [16, 32, 64, 128, 256]

        # 4次下采样
        self.down_1 = nn.ModuleList([Conv(ch_in_1, ch_list[0])])
        self.down_2 = nn.ModuleList([Conv(ch_in_2, ch_list[0])])
        self.up = nn.ModuleList([])

        for i in range(1, len(ch_list)):
            self.down_1.append(DownSampling(ch_list[i-1]))
            self.down_1.append(Conv(ch_list[i-1], ch_list[i]))
            self.down_2.append(DownSampling(ch_list[i-1]))
            self.down_2.append(Conv(ch_list[i-1], ch_list[i]))

        for i in range(len(ch_list)-1, 0, -1):
            self.up.append(UpSampling(ch_list[i]))
            self.up.append(Conv(ch_list[i], ch_list[i-1]))

        self.pred1 = nn.Sequential(
            torch.nn.Conv2d(ch_list[0], ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU()
        )
        self.pred2 = nn.Sequential(
            torch.nn.Conv2d(ch_list[1], ch_out, 3, 1, 1),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU()
        )

    def forward(self, x1, x2):
        self.down_res_1 = [self.down_1[0](x1),]
        self.down_res_2 = [self.down_2[0](x2),]

        out = []

        for i in range(1, len(self.down_1), 2):
            self.down_res_1.append(
                self.down_1[i+1](
                    self.down_1[i](self.down_res_1[-1])
                )
            )
            self.down_res_2.append(
                self.down_2[i+1](
                    self.down_2[i](self.down_res_2[-1])
                )
            )

        out.append(
            self.up[1](
                self.up[0](torch.abs(self.down_res_1[-1]-self.down_res_2[-1]),
                           torch.abs(self.down_res_1[-2]-self.down_res_2[-2]))
            )
        )

        for i in range(2, len(self.up), 2):
            ind = i//2 + 2
            out.append(
                self.up[i + 1](
                    self.up[i](out[-1], torch.abs(self.down_res_1[-ind]-self.down_res_2[-ind]))
                )
            )

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.pred1(out[-1]), self.pred2(out[-2])


if __name__ == '__main__':
    a = torch.randn(2, 3, 256, 256)
    b = torch.randn(2, 3, 256, 256)
    net = TUNet(3, 3, 2)
    print(net(a, b).shape)

