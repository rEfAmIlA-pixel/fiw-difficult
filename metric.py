"""
metric.py：多粒度模糊度量模块
功能：封装三类互补特征度量（绝对差、平方偏差、逐通道点积相似度）
对应论文多粒度联合模糊度量体系，可独立替换、消融单一度量分支
"""
import torch
import torch.nn as nn

class MultiGranularFuzzyMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, feat_p, feat_c):
        """
        Input:
            feat_p: [B, C, 1, 1] 父母提纯血缘特征
            feat_c: [B, C, 1, 1] 子女提纯血缘特征
        Output:
            multi_feat: [B, 3C, 1, 1] 三路度量拼接融合特征
        """
        # 1. 绝对差值度量：捕捉全局整体特征偏移
        abs_diff = torch.abs(feat_p - feat_c)
        # 2. 平方差值度量：放大细微五官纹理差异（难样本增益）
        sq_diff = torch.square(feat_p - feat_c)
        # 3. 逐通道点积度量：建模遗传特征通道协同相关性
        dot_sim = feat_p * feat_c
        # 多粒度特征通道拼接
        multi_feat = torch.cat([abs_diff, sq_diff, dot_sim], dim=1)
        return multi_feat

# 基线单一余弦度量（消融对比专用）
class SingleCosMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
    def forward(self, feat_p, feat_c):
        cos_sim = torch.sum(feat_p * feat_c, dim=1, keepdim=True) / (
            torch.norm(feat_p, dim=1, keepdim=True) * torch.norm(feat_c, dim=1, keepdim=True) + self.eps
        )
        return cos_sim