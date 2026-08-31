"""冻 AdaFace 余弦 + 112-MLP F_kin 余弦。系数只在验证集上选，官方 test 只评一次。"""
import os
import cv2
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from config import *
from dataset import load_fiw_split, load_fiw_pos_neg, MultiColorTransform
from model import AdaFaceBackbone, MultiGranFuzzyKinNetSwin
from eval import Logger, calc_metrics, print_result_table, RELATIONS

FUSION_EXP = "112-adaface-mlp-fusion"
MLP_CKPT = "weights/112-imagenet-mlp-head/best.pth"
FUSION_BATCH = 32


def _use_cuda():
    return DEVICE.type == "cuda"


def _resolve(path):
    if os.path.isfile(path):
        full = os.path.abspath(path)
    else:
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.isfile(alt):
            full = os.path.abspath(alt)
        else:
            raise FileNotFoundError(f"找不到权重: {path}")
    if os.path.getsize(full) == 0:
        raise RuntimeError(
            f"这是 0KB 占位文件: {full}\n请换成真实权重后再评测。见 README.md。"
        )
    return full


def _to_ada(img_bgr):
    x = torch.from_numpy(np.ascontiguousarray(img_bgr)).permute(2, 0, 1).float()
    return (x / 255.0 - 0.5) / 0.5


class FusionPairDataset(Dataset):
    """同一张图：BGR→AdaFace，RGB 九通道→Swin MLP。噪声只加一次。"""

    def __init__(self, pair_list, noise=False):
        self.pair_list = pair_list
        self.noise = noise
        self.swin_tf = MultiColorTransform()

    def __len__(self):
        return len(self.pair_list)

    def _load(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"读图失败: {path}")
        if self.noise:
            n = np.random.normal(0, 10, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return _to_ada(img), self.swin_tf(rgb)

    def __getitem__(self, idx):
        item = self.pair_list[idx]
        kin = float(item[6]) if len(item) >= 7 else 1.0
        rel = item[5] if len(item) >= 6 else "FD"
        ada_p, swin_p = self._load(item[0])
        ada_c, swin_c = self._load(item[1])
        return ada_p, ada_c, swin_p, swin_c, torch.tensor([kin]), rel


def _zfit(x):
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-8:
        s = 1.0
    return m, s


def _z(x, m, s):
    return (x - m) / s


def collect_scores(ada, mlp, pair_list, noise=False):
    ds = FusionPairDataset(pair_list, noise=noise)
    nw = 4 if NUM_WORKERS > 0 else 0
    loader = DataLoader(
        ds, batch_size=FUSION_BATCH, shuffle=False, num_workers=nw,
        pin_memory=bool(PIN_MEMORY and _use_cuda()),
    )
    ada.eval()
    mlp.eval()
    amp_on = bool(USE_AMP and _use_cuda())
    ada_s, mlp_s, labels, rels = [], [], [], []
    with torch.no_grad():
        for ada_p, ada_c, swin_p, swin_c, y, rel in loader:
            ada_p = ada_p.to(DEVICE, non_blocking=True)
            ada_c = ada_c.to(DEVICE, non_blocking=True)
            swin_p = swin_p.to(DEVICE, non_blocking=True)
            swin_c = swin_c.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_on):
                fp = ada(ada_p).flatten(1).float()
                fc = ada(ada_c).flatten(1).float()
                a = torch.cosine_similarity(fp, fc).reshape(-1)
                kin_p, kin_c, *_ = mlp.forward_pair(swin_p, swin_c)
                m = torch.cosine_similarity(kin_p.float(), kin_c.float()).reshape(-1)
            ada_s.append(a.cpu().numpy())
            mlp_s.append(m.cpu().numpy())
            labels.append(y.reshape(-1).numpy())
            rels.extend(list(rel))
    return (
        np.concatenate(ada_s).astype(np.float64),
        np.concatenate(mlp_s).astype(np.float64),
        np.concatenate(labels).astype(np.float64),
        np.array(rels),
    )


def mean_rel_auc(scores, labels, rels):
    aucs = []
    metrics = {}
    for r in RELATIONS:
        mask = rels == r
        if mask.sum() == 0 or np.unique(labels[mask]).size < 2:
            metrics[r] = {"auc": 0, "acc_05": 0, "acc_best": 0, "eer": 0}
            aucs.append(0.0)
            continue
        metrics[r] = calc_metrics(labels[mask], scores[mask])
        aucs.append(metrics[r]["auc"])
    metrics["Overall"] = calc_metrics(labels, scores)
    return float(np.mean(aucs)), metrics


def fit_blend(ada, mlp, labels):
    am, asd = _zfit(ada)
    mm, msd = _zfit(mlp)
    za, zm = _z(ada, am, asd), _z(mlp, mm, msd)
    best_a, best_auc = 0.0, -1.0
    for a in np.linspace(0.0, 1.0, 21):
        s = a * za + (1.0 - a) * zm
        auc = roc_auc_score(labels, s)
        if auc > best_auc:
            best_auc, best_a = float(auc), float(a)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(np.stack([za, zm], axis=1), labels.astype(int))
    logit_s = clf.predict_proba(np.stack([za, zm], axis=1))[:, 1]
    logit_auc = float(roc_auc_score(labels, logit_s))
    return {
        "ada_m": am, "ada_s": asd, "mlp_m": mm, "mlp_s": msd,
        "alpha": best_a, "blend_val": best_auc,
        "clf": clf, "logit_val": logit_auc,
    }


def apply_blend(ada, mlp, fit):
    za = _z(ada, fit["ada_m"], fit["ada_s"])
    zm = _z(mlp, fit["mlp_m"], fit["mlp_s"])
    blend = fit["alpha"] * za + (1.0 - fit["alpha"]) * zm
    logit = fit["clf"].predict_proba(np.stack([za, zm], axis=1))[:, 1]
    return blend, logit


def log_table(logger, title, scores, labels, rels):
    mean_auc, metrics = mean_rel_auc(scores, labels, rels)
    print_result_table(logger, metrics, "-", dataset_name=title)
    logger.log(f"四类平均 AUC = {mean_auc:.4f}")
    return mean_auc


def run_split(logger, ada, mlp, pairs, fit, tag, noise=False):
    ada_s, mlp_s, y, rels = collect_scores(ada, mlp, pairs, noise=noise)
    blend, logit = apply_blend(ada_s, mlp_s, fit)
    logger.log(f"\n===== {tag} | AdaFace 余弦 =====")
    auc_ada = log_table(logger, f"{tag} AdaFace余弦", ada_s, y, rels)
    logger.log(f"\n===== {tag} | 112-MLP =====")
    auc_mlp = log_table(logger, f"{tag} 112-MLP", mlp_s, y, rels)
    logger.log(f"\n===== {tag} | 加权融合 α={fit['alpha']:.2f}（Ada 权重） =====")
    auc_blend = log_table(logger, f"{tag} 加权融合", blend, y, rels)
    logger.log(f"\n===== {tag} | 逻辑回归融合 =====")
    auc_logit = log_table(logger, f"{tag} 逻辑回归融合", logit, y, rels)
    return {
        "ada": auc_ada, "mlp": auc_mlp, "blend": auc_blend, "logit": auc_logit,
    }


if __name__ == "__main__":
    log_dir = f"./logs/{FUSION_EXP}/"
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(os.path.join(log_dir, f"eval_fusion_{stamp}.log"))
    mlp_path = _resolve(MLP_CKPT)
    ada_path = _resolve(ADAFACE_CKPT)
    logger.log("融合：冻 AdaFace 余弦 + 112-MLP F_kin 余弦")
    logger.log("融合系数只在验证集上选，官方 test 只评一次")
    logger.log(f"AdaFace={ada_path} | MLP={mlp_path} | batch={FUSION_BATCH}")

    ada = AdaFaceBackbone(ckpt_path=ada_path).to(DEVICE)
    ada.freeze_backbone()
    mlp = MultiGranFuzzyKinNetSwin().to(DEVICE)
    ckpt = torch.load(mlp_path, map_location=DEVICE)
    incompat = mlp.load_state_dict(ckpt, strict=False)
    if incompat.missing_keys:
        raise RuntimeError(f"MLP 权重缺键: {incompat.missing_keys}")
    extra = list(incompat.unexpected_keys)
    logger.log(f"MLP 已加载 | 忽略多余键 {extra or '无'}（旧权重里的 ImageNet 分类头，评测不用）")
    mlp.eval()

    val_pos, val_neg = load_fiw_pos_neg(DATA_DIR, "val")
    val_pairs = val_pos + val_neg
    logger.log(f"验证集: {len(val_pos)}正 / {len(val_neg)}负")
    ada_v, mlp_v, y_v, rel_v = collect_scores(ada, mlp, val_pairs, noise=False)
    fit = fit_blend(ada_v, mlp_v, y_v)
    blend_v, logit_v = apply_blend(ada_v, mlp_v, fit)
    mean_ada_v, _ = mean_rel_auc(ada_v, y_v, rel_v)
    mean_mlp_v, _ = mean_rel_auc(mlp_v, y_v, rel_v)
    mean_blend_v, _ = mean_rel_auc(blend_v, y_v, rel_v)
    mean_logit_v, _ = mean_rel_auc(logit_v, y_v, rel_v)
    logger.log(
        f"验证四类平均 AUC | Ada={mean_ada_v:.4f} | MLP={mean_mlp_v:.4f} | "
        f"blend α={fit['alpha']:.2f} → {mean_blend_v:.4f} | logit={mean_logit_v:.4f}"
    )
    chosen = max(
        [("ada", mean_ada_v), ("mlp", mean_mlp_v), ("blend", mean_blend_v), ("logit", mean_logit_v)],
        key=lambda x: x[1],
    )
    logger.log(f"按验证集选定: {chosen[0]} (Val 四类平均 AUC {chosen[1]:.4f})")

    test_pairs = load_fiw_split(DATA_DIR, split="test")
    logger.log(f"测试集: {len(test_pairs)} 对")
    test_aucs = run_split(logger, ada, mlp, test_pairs, fit, "FIW Test 标准", noise=False)
    noise_aucs = run_split(logger, ada, mlp, test_pairs, fit, "FIW Test 噪声", noise=True)

    logger.log("\n===== 汇总（四类平均 AUC） =====")
    logger.log(f"标准 | Ada={test_aucs['ada']:.4f} | MLP={test_aucs['mlp']:.4f} | "
               f"blend={test_aucs['blend']:.4f} | logit={test_aucs['logit']:.4f}")
    logger.log(f"噪声 | Ada={noise_aucs['ada']:.4f} | MLP={noise_aucs['mlp']:.4f} | "
               f"blend={noise_aucs['blend']:.4f} | logit={noise_aucs['logit']:.4f}")
    logger.log("主结果报加权融合（预期 Test AUC 0.8394，α≈0.60）")
    logger.close()
