"""
mt_fmf.py 四元多粒度自适应模糊单元 MT-FMF
独立模块：高斯/Sigmoid/梯形/三角四类可学习隶属 + 1×1自适应卷积融合
输入特征维度：[B, C, 1, 1] 4D特征图，适配Swin主干输出
可开关三角分支用于消融实验
"""
import torch
import torch.nn as nn
import config

class GaussianMF(nn.Module):
    """高斯隶属：拟合平稳五官静态纹理"""
    def __init__(self, dim):
        super().__init__()
        self.c = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.sigma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return torch.exp(-((x - self.c) ** 2) / (2 * self.sigma ** 2 + 1e-8))


class SigmoidMF(nn.Module):
    """Sigmoid隶属：区分男女二元性别边界"""
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.c = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        return 1.0 / (1.0 + torch.exp(-self.a * (x - self.c)))


class TrapezoidalMF(nn.Module):
    """梯形隶属：规整区间人脸稳态特征"""
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Parameter(torch.full((1, dim, 1, 1), -1.0))
        self.b = nn.Parameter(torch.full((1, dim, 1, 1), -0.3))
        self.c = nn.Parameter(torch.full((1, dim, 1, 1), 0.3))
        self.d = nn.Parameter(torch.full((1, dim, 1, 1), 1.0))

    def forward(self, x):
        eps = 1e-8
        m1 = (x > self.a) & (x <= self.b)
        m2 = (x > self.b) & (x <= self.c)
        m3 = (x > self.c) & (x <= self.d)
        left = (x - self.a) / (self.b - self.a + eps)
        right = (self.d - x) / (self.d - self.c + eps)
        out = torch.zeros_like(x)
        out = torch.where(m1, left, out)
        out = torch.where(m2, torch.ones_like(x), out)
        out = torch.where(m3, right, out)
        return out


class TriangularMF(nn.Module):
    """三角隶属：拟合幼年至老年连续面部老化纹理（本文新增）"""
    def __init__(self, dim):
        super().__init__()
        self.k = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        return torch.maximum(torch.zeros_like(x), 1 - torch.abs(self.k * x + self.b))


class MT_FMF(nn.Module):
    """四元多粒度模糊单元，支持USE_MTFMF全局开关消融三角分支"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.gauss = GaussianMF(hidden_dim)
        self.sigmoid = SigmoidMF(hidden_dim)
        self.trap = TrapezoidalMF(hidden_dim)
        self.tri = TriangularMF(hidden_dim)
        # 四路隶属特征自适应1×1卷积融合
        self.fuse_conv = nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        fg = self.gauss(x)
        fs = self.sigmoid(x)
        ftrap = self.trap(x)
        # 消融：关闭MT-FMF舍弃三角支路，退化为GiF三元FMF
        if config.USE_MTFMF:
            ftri = self.tri(x)
            cat_all = torch.cat([fg, fs, ftrap, ftri], dim=1)
        else:
            ftri = torch.zeros_like(fg).detach()
            cat_all = torch.cat([fg, fs, ftrap, ftri], dim=1)
        out = self.fuse_conv(cat_all)
        return out