import os
import datetime
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

RELATIONS = ("FD", "FS", "MD", "MS")
REL_COLS = ("F-D", "F-S", "M-D", "M-S")


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.fh = open(log_path, "a", encoding="utf-8")

    def log(self, msg=""):
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{t}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def calc_metrics(labels, scores):
    probs = scores.copy()
    if probs.max() > 1.0 or probs.min() < 0.0:
        probs = 1.0 / (1.0 + np.exp(-probs))
    auc = roc_auc_score(labels, probs)
    pred_05 = (probs >= 0.5).astype(int)
    acc_05 = np.mean(pred_05 == labels)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thr = thresholds[best_idx]
    pred_best = (probs >= best_thr).astype(int)
    acc_best = np.mean(pred_best == labels)
    frr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - frr))
    eer = fpr[eer_idx]
    return {
        "auc": auc,
        "acc_05": acc_05,
        "acc_best": acc_best,
        "eer": eer,
        "best_thr": best_thr,
        "eer_thr": thresholds[eer_idx],
    }


def print_result_table(logger, metrics_dict, gender_acc, dataset_name="Test"):
    col_w = 12
    sep = "-" * (col_w * 6 + 2)
    logger.log(sep)
    logger.log(f"数据集：{dataset_name} (FD/FS/MD/MS)")
    logger.log(sep)
    header = "".join([c.ljust(col_w) for c in (*REL_COLS, "Mean", "Gender")])
    logger.log(header)
    logger.log(sep)

    def cells(*vals):
        out = []
        for v in vals:
            if isinstance(v, str):
                out.append(v.ljust(col_w))
            else:
                out.append(f"{float(v):.4f}".ljust(col_w))
        return "".join(out)

    aucs = [metrics_dict[r]["auc"] for r in RELATIONS]
    logger.log("AUC".ljust(8) + cells(*aucs, np.mean(aucs), gender_acc))
    acc05s = [metrics_dict[r]["acc_05"] for r in RELATIONS]
    logger.log("ACC@0.5".ljust(8) + cells(*acc05s, np.mean(acc05s), "-"))
    accbests = [metrics_dict[r]["acc_best"] for r in RELATIONS]
    logger.log("ACC@best".ljust(8) + cells(*accbests, np.mean(accbests), "-"))
    eers = [metrics_dict[r]["eer"] for r in RELATIONS]
    logger.log("EER".ljust(8) + cells(*eers, np.mean(eers), "-"))
    logger.log(sep)
    logger.log("说明：AUC/EER为阈值无关指标；ACC@0.5为固定阈值准确率；ACC@best为最优阈值准确率")
    logger.log(sep)
