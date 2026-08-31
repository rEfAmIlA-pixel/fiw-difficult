import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config import *

class AdaFaceTransform:
    """BGR uint8 → AdaFace：((x/255)-0.5)/0.5，通道保持 BGR。"""

    def __call__(self, img_bgr):
        if not isinstance(img_bgr, np.ndarray):
            img_bgr = np.asarray(img_bgr)
        img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(np.ascontiguousarray(img_bgr)).permute(2, 0, 1).float()
        return (x / 255.0 - 0.5) / 0.5


class MultiColorTransform:
    """RGB uint8 → Swin 九通道：RGB+HSV+Lab，除以 255。"""

    def __call__(self, img_rgb):
        if not isinstance(img_rgb, np.ndarray):
            img_rgb = np.asarray(img_rgb)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        cat = np.concatenate([img_rgb, hsv, lab], axis=-1)
        return torch.from_numpy(np.ascontiguousarray(cat)).permute(2, 0, 1).float() / 255.0


class SwinTrainTransform:
    """BGR 图 → 可选翻转 → RGB 九通道。112-MLP 训练用。"""

    def __init__(self, train=False):
        self.train = train
        self.color = MultiColorTransform()

    def __call__(self, img_bgr):
        if not isinstance(img_bgr, np.ndarray):
            img_bgr = np.asarray(img_bgr)
        img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        if self.train and np.random.rand() < 0.5:
            img_bgr = cv2.flip(img_bgr, 1)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.color(rgb)


def resolve_fiw_image(data_dir, rel_path):
    rel = rel_path.replace("\\", "/").lstrip("/")
    if rel.startswith("FIDs/") or rel.startswith(FID_DIR + "/"):
        return os.path.join(data_dir, rel)
    return os.path.join(data_dir, FID_DIR, rel)


def load_fiw_pair_file(data_dir, pair_file, keep_neg=True):
    parent_gender = {"fd": 0, "fs": 0, "md": 1, "ms": 1,
                     "FD": 0, "FS": 0, "MD": 1, "MS": 1}
    pairs = []
    total_lines = 0
    img_miss = 0
    n_pos = n_neg = 0

    if not os.path.exists(pair_file):
        print(f"[错误] 划分文件不存在: {pair_file}")
        return pairs
    print(f"[加载] FIW 配对: {pair_file}")
    with open(pair_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"列数应为5，实际{len(parts)}列")
            rel = parts[3]
            label = int(parts[4])
            if label == 0 and not keep_neg:
                continue
            p_path = resolve_fiw_image(data_dir, parts[1])
            c_path = resolve_fiw_image(data_dir, parts[2])
            if not os.path.isfile(p_path) or not os.path.isfile(c_path):
                img_miss += 1
                continue
            gender_label = parent_gender.get(rel, parent_gender.get(rel.lower(), 0))
            kin = 1.0 if label == 1 else 0.0
            pairs.append((p_path, c_path, gender_label, 30.0, 30.0, rel.upper(), kin))
            if kin == 1.0:
                n_pos += 1
            else:
                n_neg += 1
    print(f"  读取行数: {total_lines} | 有效正样本: {n_pos} | 有效负样本: {n_neg} | 图片缺失: {img_miss}")
    return pairs


def load_fiw_split(data_dir, split="train", keep_neg=None):
    if keep_neg is None:
        keep_neg = (split == "test")
    pair_file = os.path.join(data_dir, PAIR_DIR, f"{split}.txt")
    return load_fiw_pair_file(data_dir, pair_file, keep_neg=keep_neg)


def load_fiw_pos_neg(data_dir, split):
    pos_file = os.path.join(data_dir, PAIR_DIR, f"{split}.txt")
    neg_file = os.path.join(data_dir, PAIR_DIR, f"{split}-neg.txt")
    pos_pairs = load_fiw_pair_file(data_dir, pos_file, keep_neg=False)
    if not os.path.exists(neg_file):
        raise FileNotFoundError(f"缺少负样本文件: {neg_file}")
    neg_pairs = load_fiw_pair_file(data_dir, neg_file, keep_neg=True)
    if not neg_pairs:
        raise RuntimeError(f"负样本文件为空或未被加载: {neg_file}")
    return pos_pairs, neg_pairs


def align_face(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def genders_from_rel(rel):
    r = str(rel).upper()
    if r not in ("FD", "FS", "MD", "MS"):
        return None, None
    parent_g = 0.0 if r[0] == "F" else 1.0
    child_g = 1.0 if r[1] == "D" else 0.0
    return parent_g, child_g


def parse_fiw_ids(path):
    parts = str(path).replace("\\", "/").split("/")
    fid_num, mid_num = 0, 0
    for i, p in enumerate(parts):
        if len(p) >= 2 and p[0] in "Ff" and p[1:].isdigit():
            fid_num = int(p[1:])
            if i + 1 < len(parts) and parts[i + 1].upper().startswith("MID"):
                tail = parts[i + 1][3:]
                if tail.isdigit():
                    mid_num = int(tail)
            break
    return fid_num, fid_num * 100000 + mid_num


class KinDataset(Dataset):
    """112-MLP 训练/验证：九通道 Swin 输入。"""

    def __init__(self, pair_list, use_age_label=True, train=True):
        self.pair_list = pair_list
        self.use_age_label = use_age_label
        self.transform = SwinTrainTransform(train=train)

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        item = self.pair_list[idx]
        p_path, c_path, gender_label, age_p, age_c = item[:5]
        kin_label = float(item[6]) if len(item) >= 7 else 1.0
        img_p = cv2.imread(p_path)
        img_c = cv2.imread(c_path)
        if img_p is None or img_c is None:
            raise FileNotFoundError(f"读图失败: {p_path} 或 {c_path}")
        img_p = self.transform(img_p)
        img_c = self.transform(img_c)
        rel = item[5] if len(item) >= 6 else None
        parent_g, child_g = genders_from_rel(rel) if rel is not None else (None, None)
        if parent_g is None:
            parent_g = float(gender_label)
            child_g = float(gender_label)
        if self.use_age_label:
            label_age_p = torch.tensor([float(age_p)])
            label_age_c = torch.tensor([float(age_c)])
        else:
            label_age_p = torch.tensor([0.0])
            label_age_c = torch.tensor([0.0])
        fid_p, person_p = parse_fiw_ids(p_path)
        fid_c, person_c = parse_fiw_ids(c_path)
        return (
            img_p, img_c,
            torch.tensor([kin_label]),
            torch.tensor([parent_g]),
            torch.tensor([child_g]),
            label_age_p, label_age_c,
            torch.tensor(fid_p, dtype=torch.long),
            torch.tensor(person_p, dtype=torch.long),
            torch.tensor(fid_c, dtype=torch.long),
            torch.tensor(person_c, dtype=torch.long),
        )
