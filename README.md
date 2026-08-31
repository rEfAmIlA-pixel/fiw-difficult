# fiw-difficult：难样本加权实验工作目录（基线 0.8394）

本目录是 FIW Track I 亲子四类（FD / FS / MD / MS）的**难样本加权实验工作目录**。代码、日志、权重与文档完整继承自已验证的融合基线 `fiw-fusion/`（2026-08-20 状态，代码逐字节一致，基线权重 md5 相同）。当前目录**尚未包含任何难样本加权实现**（`config.py` 中 `USE_WEIGHT_DFC = False`）。

**当前基线最好结果**：冻 AdaFace 余弦 × 112-MLP（`F_kin` 余弦）加权融合，实验名 `112-adaface-mlp-fusion`，官方 Test 四类平均 AUC **0.8394**。

| 文件 | 内容 |
| --- | --- |
| [`说明文档.md`](说明文档.md) | 协议、模型、各轮实验、0.8394 构成 |
| [`评测结果.md`](评测结果.md) | 主结果与分表 |
| [`experiences.txt`](experiences.txt) / [`experiences.html`](experiences.html) | 逐轮实验谱 |

11 类（`fiw-11rel/`）、年龄（`fiw-age/`）、KinFaceW（`kinfacew/`）不写在本目录。人脸图像当前为占位（`data/FIW/FIDs/PUT_FIDs_HERE`），跑评测/训练前请放回真实图像；AdaFace 权重 `adaface_ir101_webface4m.ckpt` 与 `weights/112-imagenet-mlp-head/best.pth` 已是真实权重。

```bash
python eval_fusion.py   # 复现 0.8394（无再训练）
```

训练 112-MLP 前必须先改 `config.py` 的 `EXP_NAME`，否则会覆盖基线权重 `weights/112-imagenet-mlp-head/best.pth`。
