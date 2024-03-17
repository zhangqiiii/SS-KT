import random
import torch
import torch.nn.functional as F


class ProtoType:
    """
    维护一组原型向量,全部都是tensor操作
    """
    def __init__(self, changed_num=3, unchanged_num=3):
        self.device = 'cuda:0'
        self.changed_num = changed_num
        self.unchanged_num = unchanged_num
        self.changed_vector = []
        self.unchanged_vector = []
        self.distance = torch.nn.CosineSimilarity(dim=2)
        self.distance_inf = torch.nn.CosineSimilarity(dim=1)

    def update(self, feature_map, mask_map):
        """
        更新原型向量,和聚类过程相似,第一次更新随即选择向量初始化原型向量,之后每次以现有的向量为锚点进行更新
        Args:
            feature_map: tensor (B, C, H, W)
            mask_map: tensor (B, H, W)

        Returns:
            None
        """
        B, C, H, W = feature_map.size()
        mask_map = mask_map.unsqueeze(1)

        if feature_map.size()[2:] != mask_map.size()[2:]:
            mask_map = F.interpolate(mask_map.float(), size=(H, W), mode="bilinear")
            mask_map = torch.where(mask_map > 0.5, 1., 0.)

        # (B, H, W, C)
        feature_map = feature_map.permute((0, 2, 3, 1))
        # (B, H, W, 1)
        mask_map = mask_map.permute((0, 2, 3, 1))
        # feature_map (B, H, W, C) => feature_vector (B, 2, C), 2表示变化和不变
        ch_feature_vector = None
        unch_feature_vector = None
        for batch in range(B):
            changed_mean = torch.masked_select(feature_map[batch, ...], mask_map[batch, ...].bool())
            changed_mean = changed_mean.reshape((-1, C)).mean(dim=0, keepdim=True)
            unchanged_mean = torch.masked_select(feature_map[batch, ...], (1 - mask_map[batch, ...]).bool())
            unchanged_mean = unchanged_mean.reshape((-1, C)).mean(dim=0, keepdim=True)

            # changed_mean 和 unchanged_mean 都有可能为空
            if ch_feature_vector is None:
                ch_feature_vector = changed_mean
            else:
                ch_feature_vector = torch.cat((ch_feature_vector, changed_mean), dim=0)

            if unch_feature_vector is None:
                unch_feature_vector = unchanged_mean
            else:
                unch_feature_vector = torch.cat((unch_feature_vector, unchanged_mean), dim=0)

        torch.cuda.empty_cache()

        # 初始化
        if type(self.changed_vector) == list:
            tmp_init = self._init_vector(self.changed_num, ch_feature_vector)
            if isinstance(tmp_init, torch.Tensor) and not torch.isnan(tmp_init[0, 0]):
                self.changed_vector = tmp_init
        else:
            self._update_vector(self.changed_vector, ch_feature_vector)

        if type(self.unchanged_vector) == list:
            # 索引出表示变化或者不变的通道
            tmp_init = self._init_vector(self.unchanged_num, unch_feature_vector)
            if isinstance(tmp_init, torch.Tensor) and not torch.isnan(tmp_init[0, 0]):
                self.unchanged_vector = tmp_init
        # 更新
        else:
            self._update_vector(self.unchanged_vector, unch_feature_vector)


    def _init_vector(self, num, feature_vector):
        """
        Args:
            num: 
            feature_vector: (B, C)

        Returns:
            tensor(num, C)
        """
        index = torch.tensor(random.sample(range(feature_vector.size()[0]), num), device=self.device)
        # tensor(num, C)
        vector = torch.index_select(feature_vector, 0, index)
        return vector

    def _update_vector(self, vector, feature_vector):
        """

        Args:
            vector: (num, C)
            feature_vector: (B, C)

        Returns:
            None
        """
        S, C = feature_vector.size()
        # 直接用广播机制来运算,如下:
        tmp_vector = vector.unsqueeze(1)
        tmp_feature_vector = feature_vector.unsqueeze(0)
        # ch_dis: (num, B)
        ch_dis = self.distance(tmp_feature_vector, tmp_vector)
        # max_index:(B,)
        max_index = torch.argmax(ch_dis, dim=0)
        torch.cuda.empty_cache()
        for i in range(vector.size()[0]):
            # 利用广播机制 feature_vector:(B, C), max_index:(B,)
            mean_vector = torch.masked_select(feature_vector, (max_index == i).unsqueeze(1))
            mean_vector = mean_vector.reshape((-1, C)).mean(dim=0)
            if torch.isnan(mean_vector[0]):
                break
            vector[i] += mean_vector
            vector[i] /= 2

    def inference(self, enc):
        """
        推理的时候用
        Args:
            enc: tensor[C, H, W] 编码的特征图

        Returns:
            tensor(H, W), tensor(H, W) 两个距离度量图
        """
        # changed_map = self._inference(enc, self.changed_vector, 'sum')
        changed_map = self._inference(enc, self.changed_vector)
        # unchanged_map = self._inference(enc, self.unchanged_vector, 'sum')
        unchanged_map = self._inference(enc, self.unchanged_vector)
        return changed_map, unchanged_map

    def _inference(self, enc, vector, mode='max'):
        """

        Args:
            enc: tensor (C, H, W)
            vector: tensor (num, C)
            num: int

        Returns:
            torch (H, W)
        """
        # 同样使用广播机制
        tmp_vector = vector.unsqueeze(-1).unsqueeze(-1)
        tmp_enc = enc.unsqueeze(0)
        distance_map = self.distance_inf(tmp_enc, tmp_vector)
        if mode == 'max':
            distance_map = torch.max(distance_map, dim=0).values
        elif mode == 'sum':
            distance_map = torch.sum(distance_map, dim=0)
        return distance_map


if __name__ == "__main__":
    a = torch.randn((2, 5, 7, 7))
    print(a[0, :, 0, 0])
    # 上下两行不相等
    print(a.reshape((2 * 7 * 7, 5))[0])
    # 正确做法是先转置再reshape
    b = a.permute(0, 2, 3, 1)
    print(b.size())
    print(b.reshape(-1, 5)[0])
    """
    a = torch.randn((5, 7))
    b = torch.zeros((5,))
    c = torch.masked_select(a, (b == 1).unsqueeze(1))
    c = c.reshape((-1, 7))
    print(a, c.size())
    """
