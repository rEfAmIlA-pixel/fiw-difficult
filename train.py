import os
import random
import cv2
import numpy as np
from datetime import datetime
import torch
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from config import *
from dataset import KinDataset, load_fiw_pos_neg, parse_fiw_ids
from model import MultiGranFuzzyKinNetSwin
from losses import total_multi_loss, loss_bce, hard_margin_weight, weighted_bce

cv2.setNumThreads(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 固定输入尺寸 112×112，打开 benchmark 换速度；不再强求逐 bit 可复现
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _worker_init(_worker_id):
    cv2.setNumThreads(0)


def _use_cuda():
    return DEVICE.type == "cuda"


def _parent_fid(pair):
    return parse_fiw_ids(pair[0])[0]


def _family_weights(pairs, power=1.0):
    """每条样本权重 = 1 / n_fid^power。power=1 各家总次数相同；0.5 为平方根。"""
    fids = [_parent_fid(p) for p in pairs]
    counts = Counter(fids)
    pwr = float(power)
    weights = [1.0 / (max(counts[fid], 1) ** pwr) for fid in fids]
    return torch.tensor(weights, dtype=torch.double), counts


def _top_family_expected_share(counts, power):
    """加权后最大家占总抽样的期望比例。"""
    masses = [cnt ** (1.0 - float(power)) for cnt in counts.values()]
    total = sum(masses)
    top_n = counts.most_common(1)[0][1]
    return (top_n ** (1.0 - float(power))) / total if total > 0 else 0.0


def _make_loader(pairs, train, batch_size=None, family_bal=False, sampler_seed=None):
    ds = KinDataset(pairs, use_age_label=USE_AGE_MASK, train=train)
    nw = NUM_WORKERS if NUM_WORKERS > 0 else 0
    kwargs = {
        "batch_size": BATCH_SIZE if batch_size is None else batch_size,
        "drop_last": train,
        "num_workers": nw,
        "pin_memory": bool(PIN_MEMORY and _use_cuda()),
        "worker_init_fn": _worker_init if nw > 0 else None,
    }
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = PREFETCH_FACTOR

    fam_info = None
    if train and family_bal:
        weights, counts = _family_weights(pairs, power=FAM_BAL_POWER)
        gen = torch.Generator()
        gen.manual_seed(RANDOM_SEED if sampler_seed is None else sampler_seed)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(ds), replacement=True, generator=gen
        )
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
        n_fam = len(counts)
        top_fid, top_n = counts.most_common(1)[0]
        fam_info = (
            n_fam, top_fid, top_n / len(pairs),
            _top_family_expected_share(counts, FAM_BAL_POWER),
        )
    else:
        kwargs["shuffle"] = train
    return DataLoader(ds, **kwargs), fam_info

class Logger:
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._fh = open(log_path, "a", encoding="utf-8")
    
    def log(self, msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()
    
    def close(self):
        self._fh.close()

class ModelEMA:
    """对参数做指数滑动平均；验证和 best.pth 用这份权重。"""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def _snapshot_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _validate_for_ckpt(model, val_loader, ema):
    """解冻后用 EMA 验证并返回可保存的权重；冻结阶段用当前模型。"""
    if ema is None:
        return validate(model, val_loader), _snapshot_state(model)
    raw = _snapshot_state(model)
    ema.copy_to(model)
    auc = validate(model, val_loader)
    save_state = _snapshot_state(model)
    model.load_state_dict(raw, strict=True)
    return auc, save_state


def validate(model, val_loader):
    """验证：固定正负样本对，计算真实AUC，作为早停依据"""
    model.eval()
    all_scores = []
    all_labels = []
    amp_on = bool(USE_AMP and _use_cuda())

    with torch.no_grad():
        for batch in val_loader:
            img_p, img_c, lab_kv, *_ = batch
            img_p = img_p.to(DEVICE, non_blocking=True)
            img_c = img_c.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_on):
                kin_p, kin_c, _, _, _, _, _, _, _ = model.forward_pair(img_p, img_c)
            scores = torch.cosine_similarity(kin_p.float(), kin_c.float())
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)
            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(lab_kv.reshape(-1).cpu().numpy().tolist())

    labels = np.asarray(all_labels)
    scores_np = np.asarray(all_scores, dtype=np.float64)
    if np.unique(labels).size < 2:
        raise RuntimeError("验证集标签只有一类，无法计算 AUC（请检查负样本是否标为 0）")
    if not np.isfinite(scores_np).all():
        scores_np = np.nan_to_num(scores_np, nan=0.0, posinf=1.0, neginf=-1.0)
    auc = roc_auc_score(labels, scores_np)
    return auc

def _trainable(modules_or_params):
    params = []
    for item in modules_or_params:
        if isinstance(item, torch.nn.Module):
            params.extend(p for p in item.parameters() if p.requires_grad)
        else:
            if item.requires_grad:
                params.append(item)
    return params


def _build_optimizer(model, stage):
    """stage=1 冻主干；stage=2 分层学习率。年龄关闭时不把年龄参数送进优化器。"""
    head_modules = [model.fdm, model.fim, model.head_kv, model.head_gr]
    if USE_AGE_MASK:
        head_modules.append(model.head_age)
    if stage == 1:
        return torch.optim.AdamW(
            _trainable(head_modules + [model.backbone.conv_in]),
            lr=LR_HEAD, weight_decay=1e-4
        )
    groups = [
        {"params": _trainable([model.backbone]), "lr": LR_BACKBONE},
        {"params": _trainable(head_modules), "lr": LR_HEAD_UNFREEZE},
    ]
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def _cosine_t_max():
    return max(1, EPOCHS - FREEZE_BACKBONE_EPOCH)


def _build_cosine_scheduler(optimizer, last_epoch=-1):
    """解冻后对所有 param group 做 cosine；T_max = 解冻后剩余轮数。"""
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=_cosine_t_max(),
        eta_min=ETA_MIN,
        last_epoch=last_epoch,
    )


def _grads_finite(model):
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def _lr_str(optimizer):
    lrs = [g["lr"] for g in optimizer.param_groups]
    if len(lrs) == 1:
        return f"lr={lrs[0]:.2e}"
    return f"lr_bb={lrs[0]:.2e} lr_head={lrs[1]:.2e}"


def _unpack_batch(batch):
    img_p, img_c, lab_kv, lab_gr_p, lab_gr_c, lab_ap, lab_ac, *rest = batch
    nb = _use_cuda()
    img_p = img_p.to(DEVICE, non_blocking=nb)
    img_c = img_c.to(DEVICE, non_blocking=nb)
    lab_kv = lab_kv.to(DEVICE, non_blocking=nb)
    lab_gr_p = lab_gr_p.to(DEVICE, non_blocking=nb)
    lab_gr_c = lab_gr_c.to(DEVICE, non_blocking=nb)
    if USE_AGE_MASK:
        lab_ap = lab_ap.to(DEVICE, non_blocking=nb)
        lab_ac = lab_ac.to(DEVICE, non_blocking=nb)
    else:
        lab_ap = lab_ac = None
    pair_ids = None
    if len(rest) >= 4:
        pair_ids = tuple(t.to(DEVICE, non_blocking=nb).reshape(-1) for t in rest[:4])
    return img_p, img_c, lab_kv, lab_gr_p, lab_gr_c, lab_ap, lab_ac, pair_ids


def run_epoch(model, pos_loader, neg_loader, optimizer=None, scaler=None, ema=None):
    """正样本走完整多任务损失（含 DFC）；负样本只做亲缘 BCE，避免 DFC 把非亲人当正对。"""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0
    n_skip = 0
    feat_norm_sum = 0.0
    grad_norm_sum = 0.0
    amp_on = bool(USE_AMP and _use_cuda())

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for pos_batch, neg_batch in zip(pos_loader, neg_loader):
            img_p, img_c_pos, lab_kv_pos, lab_gr_p, lab_gr_c, lab_ap, lab_ac, pos_ids = _unpack_batch(pos_batch)
            img_p_n, img_c_neg, lab_kv_neg, *_ = _unpack_batch(neg_batch)

            id_kwargs = {}
            if pos_ids is not None:
                id_kwargs = {
                    "fid_p": pos_ids[0], "person_p": pos_ids[1],
                    "fid_c": pos_ids[2], "person_c": pos_ids[3],
                }

            with torch.cuda.amp.autocast(enabled=amp_on):
                kin_p, kin_c_pos, pred_kv_pos, pg_pos, pc_pos, pap_pos, pac_pos, _, _ = model.forward_pair(img_p, img_c_pos)
                loss_pos, _, _, _, _ = total_multi_loss(
                    kin_p, kin_c_pos, pred_kv_pos, pg_pos, pc_pos, pap_pos, pac_pos,
                    lab_kv_pos, lab_gr_p, lab_gr_c, lab_ap, lab_ac, **id_kwargs
                )
                kin_p_n, kin_c_neg, pred_kv_neg, _, _, _, _, _, _ = model.forward_pair(img_p_n, img_c_neg)
                if USE_HW_BCE:
                    w_neg = hard_margin_weight(kin_p_n, kin_c_neg, HW_MARGIN_NEG,
                                               HW_GAIN, HW_LO, HW_HI,
                                               higher_is_harder=True)
                    loss_neg = weighted_bce(pred_kv_neg.float(), lab_kv_neg.float(), w_neg)
                else:
                    loss_neg = loss_bce(pred_kv_neg.float(), lab_kv_neg.float())
                loss_all = loss_pos + loss_neg

            if is_train:
                if not torch.isfinite(loss_all):
                    n_skip += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_on:
                    scaler.scale(loss_all).backward()
                    scaler.unscale_(optimizer)
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if (not torch.isfinite(gnorm)) or (not _grads_finite(model)):
                        n_skip += 1
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_all.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if (not torch.isfinite(gnorm)) or (not _grads_finite(model)):
                        n_skip += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    optimizer.step()
                if ema is not None:
                    ema.update(model)
                grad_norm_sum += float(gnorm)

            if torch.isfinite(loss_all):
                total_loss += loss_all.item()
                n_batches += 1
                feat_norm_sum += 0.5 * (
                    kin_p.detach().float().norm(dim=-1).mean()
                    + kin_c_pos.detach().float().norm(dim=-1).mean()
                ).item()

    avg_loss = total_loss / max(n_batches, 1)
    avg_norm = feat_norm_sum / max(n_batches, 1)
    avg_gnorm = grad_norm_sum / max(n_batches, 1)
    return avg_loss, avg_norm, avg_gnorm, n_skip, n_batches + n_skip


def train_fiw(logger):
    train_pos, train_neg = load_fiw_pos_neg(DATA_DIR, split='train')
    val_pos, val_neg = load_fiw_pos_neg(DATA_DIR, split='val')

    pos_loader, pos_fam = _make_loader(
        train_pos, train=True, family_bal=USE_FAM_BAL, sampler_seed=RANDOM_SEED
    )
    neg_loader, neg_fam = _make_loader(
        train_neg, train=True, family_bal=USE_FAM_BAL, sampler_seed=RANDOM_SEED + 1
    )
    val_loader, _ = _make_loader(val_pos + val_neg, train=False, batch_size=VAL_BATCH_SIZE)

    model = MultiGranFuzzyKinNetSwin(pretrained=PRETRAIN_BACKBONE).to(DEVICE)

    model.backbone.freeze_backbone()
    optimizer = _build_optimizer(model, stage=1)
    scheduler = None
    scaler = torch.cuda.amp.GradScaler(enabled=bool(USE_AMP and _use_cuda()))
    ema = None

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_auc = 0.0
    bad_epochs = 0
    best_path = os.path.join(SAVE_DIR, CKPT_BEST)
    last_path = os.path.join(SAVE_DIR, CKPT_LAST)

    logger.log(f"实验: {EXP_NAME} | 权重目录: {SAVE_DIR}")
    logger.log(f"阶段1：冻结 Swin 主干（conv_in 除外），训练头部与解耦模块，lr={LR_HEAD}")
    logger.log(
        f"阶段2：cosine T_max={_cosine_t_max()} eta_min={ETA_MIN} | "
        f"早停 patience={EARLY_STOP_PATIENCE}"
    )
    logger.log(f"年龄分支: {'开启' if USE_AGE_MASK else '关闭（不进前向/损失/优化器）'}")
    if USE_EMA:
        logger.log(f"EMA: 解冻后开启 | decay={EMA_DECAY} | 验证与 best.pth 用滑动平均")
    else:
        logger.log("EMA: 关")
    logger.log(
        f"训练集: {len(train_pos)}正 / {len(train_neg)}负 | "
        f"验证集: {len(val_pos)}正 / {len(val_neg)}负"
    )
    logger.log(
        f"加速: workers={NUM_WORKERS} | AMP={bool(USE_AMP and _use_cuda())} | "
        f"train_bs={BATCH_SIZE} | val_bs={VAL_BATCH_SIZE} | pin_memory={PIN_MEMORY}"
    )
    if pos_fam is not None:
        n_fam, top_fid, top_share, exp_share = pos_fam
        logger.log(
            f"家庭限权: 开 | power={FAM_BAL_POWER} | 正样本家庭数={n_fam} | "
            f"最大家 F{top_fid:04d} 原占比 {top_share:.1%} → 期望 {exp_share:.1%}"
        )
        if neg_fam is not None:
            n_neg, top_neg, share_neg, exp_neg = neg_fam
            logger.log(
                f"家庭限权: 负样本家庭数={n_neg} | "
                f"最大家 F{top_neg:04d} 原占比 {share_neg:.1%} → 期望 {exp_neg:.1%}"
            )
    else:
        logger.log("家庭限权: 关（均匀打乱）")
    logger.log(
        f"难样本加权: {'BCE' if USE_HW_BCE else ('DFC' if USE_HW_DFC else '关')} | "
        f"margin_pos={HW_MARGIN_POS} margin_neg={HW_MARGIN_NEG} | "
        f"gain={HW_GAIN} | clip=[{HW_LO},{HW_HI}]"
    )

    start_epoch = 1
    if RESUME_START_EPOCH > 0 and os.path.exists(last_path):
        model.load_state_dict(torch.load(last_path, map_location=DEVICE))
        start_epoch = RESUME_START_EPOCH
        best_auc = float(RESUME_BEST_AUC)
        if start_epoch > FREEZE_BACKBONE_EPOCH:
            model.backbone.unfreeze_all()
            optimizer = _build_optimizer(model, stage=2)
            scheduler = _build_cosine_scheduler(
                optimizer,
                last_epoch=start_epoch - FREEZE_BACKBONE_EPOCH - 2,
            )
        logger.log(f"从 {last_path} 恢复，下一轮={start_epoch}，Best AUC={best_auc:.4f}")

    for epoch in range(start_epoch, EPOCHS + 1):
        if epoch == FREEZE_BACKBONE_EPOCH + 1 and start_epoch <= FREEZE_BACKBONE_EPOCH + 1:
            model.backbone.unfreeze_all()
            optimizer = _build_optimizer(model, stage=2)
            scheduler = _build_cosine_scheduler(optimizer)
            if USE_EMA:
                ema = ModelEMA(model, EMA_DECAY)
            logger.log(
                f"===== 阶段2：解冻主干 + cosine，主干lr={LR_BACKBONE} "
                f"头部lr={LR_HEAD_UNFREEZE} T_max={_cosine_t_max()} ====="
            )
            if ema is not None:
                logger.log(f"EMA 已从解冻起点重置 | decay={EMA_DECAY}")

        train_loss, feat_norm, grad_norm, n_skip, n_seen = run_epoch(
            model, pos_loader, neg_loader, optimizer, scaler=scaler, ema=ema
        )
        torch.save(model.state_dict(), last_path)
        val_auc, save_state = _validate_for_ckpt(model, val_loader, ema)

        improved = val_auc > best_auc
        if improved:
            best_auc = val_auc
            bad_epochs = 0
            torch.save(save_state, best_path)
        else:
            bad_epochs += 1

        if scheduler is not None:
            scheduler.step()

        flag = " *" if improved else ""
        logger.log(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | Best AUC: {best_auc:.4f} | "
            f"feat_n={feat_norm:.2f} g_n={grad_norm:.3f} skip={n_skip}/{n_seen} | "
            f"{_lr_str(optimizer)}{flag}"
        )

        if EARLY_STOP_PATIENCE > 0 and bad_epochs >= EARLY_STOP_PATIENCE:
            logger.log(
                f"早停：验证 AUC 连续 {EARLY_STOP_PATIENCE} 轮未提升，"
                f"最优 {best_auc:.4f} @ {best_path}"
            )
            break

    logger.log("=" * 60)
    logger.log(f"训练完成 | 最优验证 AUC = {best_auc:.4f} | 权重 {best_path}")
    logger.log("下一步: python eval_fusion.py  （需 AdaFace ckpt + 本轮 best.pth）")
    logger.log("=" * 60)

if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"train_fiw_{stamp}.log")
    logger = Logger(log_path)
    
    logger.log(
        f"配置: EXP={EXP_NAME} | IMG={IMG_SIZE} | Swin ImageNet={PRETRAIN_BACKBONE} | "
        f"BATCH={BATCH_SIZE} | VAL_BATCH={VAL_BATCH_SIZE} | EPOCHS={EPOCHS} | "
        f"USE_AGE_MASK={USE_AGE_MASK} | cosine+early_stop={EARLY_STOP_PATIENCE} | "
        f"FAM_BAL={USE_FAM_BAL} | EMA={USE_EMA}"
    )
    gamma_eff = GAMMA if USE_AGE_MASK else 0.0
    logger.log(f"损失权重: BCE={ALPHA} | Gender={BETA} | Age={gamma_eff} | DFC={1-ALPHA-BETA-gamma_eff} | TAU_CS={TAU_CS}")
    logger.log(f"两阶段冻结: 前{FREEZE_BACKBONE_EPOCH}轮冻结主干")
    
    train_fiw(logger)
    logger.close()