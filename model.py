import os
import torch
import torch.nn as nn
import config
import adaface_net
from config import IMG_SIZE, ADAFACE_ARCH, ADAFACE_CKPT
from metric import MultiGranularFuzzyMetric
from mt_fmf import MT_FMF


class DualFDM(nn.Module):
    def __init__(self, feat_dim, mid_dim):
        super().__init__()
        self.com_mf = MT_FMF(feat_dim, mid_dim)
        self.gen_mf = MT_FMF(feat_dim, mid_dim)
        self.age_mf = MT_FMF(feat_dim, mid_dim)
        self.conv_com = nn.Conv2d(mid_dim, feat_dim, 1)
        self.conv_gen_mask = nn.Conv2d(mid_dim, feat_dim, 1)
        self.conv_age_mask = nn.Conv2d(mid_dim, feat_dim, 1)
        if not config.USE_AGE_MASK:
            for module in (self.age_mf, self.conv_age_mask):
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        feat_com = self.conv_com(self.com_mf(x))
        feat_gen_raw = self.gen_mf(x)
        mask_g = torch.sigmoid(self.conv_gen_mask(feat_gen_raw))
        F_g = x * mask_g
        if config.USE_AGE_MASK:
            feat_age_raw = self.age_mf(x)
            mask_a = torch.sigmoid(self.conv_age_mask(feat_age_raw))
            F_a = x * mask_a
            F_kin = x * (1 - mask_g.detach()) * (1 - mask_a.detach()) + feat_com
        else:
            mask_a = torch.zeros_like(x)
            F_a = torch.zeros_like(x)
            F_kin = x * (1 - mask_g.detach()) + feat_com
        return F_kin, F_g, F_a, mask_g, mask_a


class FIM(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.metric = MultiGranularFuzzyMetric()
        self.fmf = MT_FMF(feat_dim * 3, hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, feat_dim, 1),
            nn.LayerNorm([feat_dim, 1, 1]),
            nn.ReLU(),
        )

    def forward(self, fp, fc):
        fuse = self.metric(fp, fc)
        fuse = self.fmf(fuse)
        return self.head(fuse)


def _resolve_ckpt(path):
    if os.path.isfile(path):
        full = os.path.abspath(path)
    else:
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.isfile(alt):
            full = os.path.abspath(alt)
        else:
            raise FileNotFoundError(f"找不到 AdaFace 权重: {path}")
    if os.path.getsize(full) == 0:
        raise RuntimeError(f"这是 0KB 占位文件: {full}，请换成真实 AdaFace ckpt。")
    return full


def _load_adaface_state(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"AdaFace ckpt 应为含 state_dict 的字典: {path}")
    state = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    if not state:
        raise RuntimeError(f"AdaFace ckpt 没有 model.* 键: {path}")
    return state


class AdaFaceBackbone(nn.Module):
    """IR-101：L2 归一化 512 维 embedding。"""

    def __init__(self, ckpt_path=None, arch=None):
        super().__init__()
        if IMG_SIZE != 112:
            raise RuntimeError(f"AdaFace IR-101 按 112×112 训练，当前 IMG_SIZE={IMG_SIZE}。")
        arch = arch or ADAFACE_ARCH
        ckpt_path = _resolve_ckpt(ckpt_path or ADAFACE_CKPT)
        self.net = adaface_net.build_model(arch)
        state = _load_adaface_state(ckpt_path)
        self.net.load_state_dict(state, strict=True)
        self._frozen = False
        print(f"[预训练] AdaFace {arch}: {ckpt_path} | 键 {len(state)}")

    def forward(self, x):
        emb, _ = self.net(x)
        return emb.unsqueeze(-1).unsqueeze(-1)

    def train(self, mode=True):
        if self._frozen:
            mode = False
        return super().train(mode)

    def freeze_backbone(self):
        for param in self.net.parameters():
            param.requires_grad = False
        self._frozen = True
        self.eval()


class SwinTiny112(nn.Module):
    """112-imagenet-mlp-head 的 Swin 主干：9 通道 conv_in，输出 [B, 768, 1, 1]。"""

    def __init__(self, pretrained=False):
        super().__init__()
        import timm
        self.conv_in = nn.Conv2d(9, 3, kernel_size=1, bias=False)
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            img_size=112, patch_size=4,
            in_chans=3, pretrained=pretrained, num_classes=0,
        )
        self.proj_out = nn.Linear(768, 768)
        self._frozen = False
        if pretrained:
            print("[预训练] Swin-Tiny ImageNet")

    def forward(self, x):
        x = self.conv_in(x)
        feat_map = self.backbone.forward_features(x)
        feat_flat = feat_map.mean(dim=[1, 2])
        out = self.proj_out(feat_flat)
        return out.unsqueeze(-1).unsqueeze(-1)

    def train(self, mode=True):
        super().train(mode)
        if self._frozen:
            self.backbone.eval()
        return self

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._frozen = True
        self.backbone.eval()

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        self._frozen = False


class MultiGranFuzzyKinNetSwin(nn.Module):
    """112-MLP：训练得到 weights/112-imagenet-mlp-head/best.pth。"""

    def __init__(self, pretrained=False):
        super().__init__()
        feat_dim, mid_dim = 768, 256
        self.backbone = SwinTiny112(pretrained=pretrained)
        self.fdm = DualFDM(feat_dim, mid_dim)
        self.fim = FIM(feat_dim, mid_dim)
        self.head_kv = nn.Sequential(
            nn.Linear(feat_dim, mid_dim), nn.ReLU(inplace=True), nn.Linear(mid_dim, 1)
        )
        self.head_gr = nn.Sequential(
            nn.Linear(feat_dim, mid_dim), nn.ReLU(inplace=True), nn.Linear(mid_dim, 1)
        )
        self.head_age = nn.Linear(feat_dim, 1)

    def forward_pair(self, img_p, img_c):
        fp_raw = self.backbone(img_p)
        fc_raw = self.backbone(img_c)
        Fkin_p, Fg_p, Fa_p, Mg_p, Ma_p = self.fdm(fp_raw)
        Fkin_c, Fg_c, Fa_c, Mg_c, Ma_c = self.fdm(fc_raw)
        fuse_feat = self.fim(Fkin_p, Fkin_c).flatten(1)
        pred_kv = self.head_kv(fuse_feat)
        pred_gr_p = self.head_gr(Fg_p.flatten(1))
        pred_gr_c = self.head_gr(Fg_c.flatten(1))
        zeros = fuse_feat.new_zeros((fuse_feat.shape[0], 1))
        kin_p = Fkin_p.flatten(1)
        kin_c = Fkin_c.flatten(1)
        return kin_p, kin_c, pred_kv, pred_gr_p, pred_gr_c, zeros, zeros, Mg_p, Ma_p
