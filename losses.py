import torch
import torch.nn.functional as F
import config
from config import A_DFC, B_DFC, HARD_WEIGHT, HARD_THRESH, ALPHA, BETA, GAMMA

def mf_cosine(x, y):
    """余弦相似度矩阵 [B, B]"""
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    return x_norm @ y_norm.T

def mf_euclidean(x, y):
    """负欧氏距离相似度矩阵 [B, B]"""
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    dist = torch.norm(diff, p=2, dim=-1)
    return -dist

def dfc_valid_neg_mask(fid_p, person_p, fid_c, person_c):
    """[B,B]：True 表示可作为 DFC 负样本（不同家且不同人，且非对角）。"""
    fid_p = fid_p.reshape(-1)
    person_p = person_p.reshape(-1)
    fid_c = fid_c.reshape(-1)
    person_c = person_c.reshape(-1)
    same_family = (
        (fid_p[:, None] == fid_p[None, :])
        | (fid_p[:, None] == fid_c[None, :])
        | (fid_c[:, None] == fid_p[None, :])
        | (fid_c[:, None] == fid_c[None, :])
    )
    same_person = (
        (person_p[:, None] == person_p[None, :])
        | (person_p[:, None] == person_c[None, :])
        | (person_c[:, None] == person_p[None, :])
        | (person_c[:, None] == person_c[None, :])
    )
    eye = torch.eye(fid_p.shape[0], dtype=torch.bool, device=fid_p.device)
    return ~(same_family | same_person | eye)


def info_nce(sim_matrix, tau, valid_neg=None, row_weight=None):
    """向量化InfoNCE损失，正样本为对角线；valid_neg 为 False 的位置不进分母。
    row_weight: [B] 逐行权重（B2 难样本加权），按 sum(w·L)/sum(w) 归一。"""
    sim = sim_matrix / tau
    # 数值稳定处理
    sim_max = torch.max(sim, dim=1, keepdim=True)[0].detach()
    sim = sim - sim_max
    exp_sim = torch.exp(sim)
    numerator = torch.diag(exp_sim)
    if valid_neg is None:
        denominator = torch.sum(exp_sim, dim=1)
        loss = -torch.log(numerator / (denominator + 1e-8))
        if row_weight is None:
            return loss.mean()
        w = row_weight.to(loss.dtype)
        return (loss * w).sum() / w.sum().clamp(min=1e-8)
    neg_sum = (exp_sim * valid_neg.to(exp_sim.dtype)).sum(dim=1)
    denominator = numerator + neg_sum
    loss = -torch.log(numerator / (denominator + 1e-8))
    has_neg = valid_neg.any(dim=1)
    if not torch.any(has_neg):
        return sim_matrix.new_zeros(())
    if row_weight is None:
        return loss[has_neg].mean()
    w = row_weight.to(loss.dtype)
    return (loss[has_neg] * w[has_neg]).sum() / w[has_neg].sum().clamp(min=1e-8)

def L_MF(x, y, mf_func, tau, valid_neg=None, row_weight=None):
    """双向对称对比损失"""
    sim_xy = mf_func(x, y)
    sim_yx = mf_func(y, x)
    l1 = info_nce(sim_xy, tau, valid_neg, row_weight)
    l2 = info_nce(sim_yx, tau, valid_neg, row_weight)
    return (l1 + l2) / 2.0

def loss_dfc(kin_p, kin_c, fid_p=None, person_p=None, fid_c=None, person_c=None, row_weight=None):
    """双度量对比损失。对比在 FP32 中计算。row_weight: [B] 逐行难样本权重。"""
    kin_p = kin_p.float()
    kin_c = kin_c.float()
    valid_neg = None
    if fid_p is not None and person_p is not None and fid_c is not None and person_c is not None:
        valid_neg = dfc_valid_neg_mask(fid_p, person_p, fid_c, person_c)
    l_cs = L_MF(kin_p, kin_c, mf_cosine, config.TAU_CS, valid_neg, row_weight)
    l_ed = L_MF(kin_p, kin_c, mf_euclidean, config.TAU_ED, valid_neg, row_weight)
    loss_base = A_DFC * l_cs + B_DFC * l_ed
    
    if config.USE_WEIGHT_DFC:
        # 难样本加权：正对余弦相似度低于阈值的样本加大损失权重
        sim_pos = torch.diag(mf_cosine(kin_p, kin_c))
        weight = torch.where(sim_pos < HARD_THRESH,
                            torch.full_like(sim_pos, HARD_WEIGHT),
                            torch.ones_like(sim_pos))
        loss_weighted = loss_base * weight.mean()
        return loss_weighted
    return loss_base

def loss_bce(logits, label):
    return F.binary_cross_entropy_with_logits(logits, label)

def hard_margin_weight(kin_p, kin_c, margin, gain, lo, hi, higher_is_harder):
    """按对余弦算难样本权重；权重已 detach，不参与梯度。返回 [B]。"""
    kin_p = kin_p.float()
    kin_c = kin_c.float()
    cos = torch.cosine_similarity(kin_p, kin_c, dim=-1).detach()
    gap = (cos - margin) if higher_is_harder else (margin - cos)
    w = 1.0 + gain * torch.clamp(gap, min=0.0)
    return torch.clamp(w, min=lo, max=hi)

def weighted_bce(logits, label, weight):
    """逐样本加权 BCE：sum(w·L)/sum(w)，保持与未加权同量级。"""
    bce = F.binary_cross_entropy_with_logits(logits, label, reduction="none")  # [B,1]
    w = weight.reshape(-1, 1)  # [B,1] 对齐
    return (bce * w).sum() / w.sum().clamp(min=1e-8)

def loss_l1(pred_age, real_age):
    return F.l1_loss(pred_age, real_age)

def _age_weight():
    """年龄关闭时 GAMMA 不得占用 DFC 权重。"""
    return GAMMA if config.USE_AGE_MASK else 0.0

# 完整总损失
def total_multi_loss(kin_p, kin_c, pred_kv, pred_gr_p, pred_gr_c, pred_ap, pred_ac, lab_kv, lab_gr_p, lab_gr_c, lab_ap, lab_ac, fid_p=None, person_p=None, fid_c=None, person_c=None):
    if config.USE_HW_BCE:
        w_pos = hard_margin_weight(kin_p, kin_c, config.HW_MARGIN_POS,
                                   config.HW_GAIN, config.HW_LO, config.HW_HI,
                                   higher_is_harder=False)
        l_kv = weighted_bce(pred_kv, lab_kv, w_pos)
    else:
        l_kv = loss_bce(pred_kv, lab_kv)
    l_gr_p = loss_bce(pred_gr_p, lab_gr_p)
    l_gr_c = loss_bce(pred_gr_c, lab_gr_c)
    l_gr = (l_gr_p + l_gr_c) / 2

    gamma = _age_weight()
    if config.USE_AGE_MASK:
        l_age_p = loss_l1(pred_ap, lab_ap)
        l_age_c = loss_l1(pred_ac, lab_ac)
        l_age = (l_age_p + l_age_c) / 2
    else:
        l_age = kin_p.new_zeros(())

    if config.USE_HW_DFC:
        # B2：同一套 margin 软加权放到 DFC InfoNCE 上，逐行加权（权重已 detach）
        w_dfc = hard_margin_weight(kin_p, kin_c, config.HW_MARGIN_POS,
                                   config.HW_GAIN, config.HW_LO, config.HW_HI,
                                   higher_is_harder=False)
        l_dfc = loss_dfc(kin_p, kin_c, fid_p=fid_p, person_p=person_p,
                         fid_c=fid_c, person_c=person_c, row_weight=w_dfc)
    else:
        l_dfc = loss_dfc(kin_p, kin_c, fid_p=fid_p, person_p=person_p, fid_c=fid_c, person_c=person_c)
    total = ALPHA * l_kv + BETA * l_gr + gamma * l_age + (1 - ALPHA - BETA - gamma) * l_dfc
    return total, l_dfc, l_kv, l_gr, l_age