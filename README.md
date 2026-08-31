# 亲子四类融合（从 0.8394 基线）

本目录是 FIW Track I 亲子四类（FD / FS / MD / MS）的代码与文档。当前最好：冻 AdaFace 余弦 × 112-MLP 加权融合，官方 Test 四类平均 AUC **0.8394**。

| 文件 | 内容 |
| --- | --- |
| [`说明文档.md`](说明文档.md) | 协议、模型、各轮实验、0.8394 构成 |
| [`评测结果.md`](评测结果.md) | 主结果与分表 |
| [`experiences.html`](experiences.html) | 逐轮实验谱 |

11 类、年龄、KinFace 不写在本目录。人脸和权重若是占位，跑评测前请换成真实文件。

```bash
cd fiw-fusion
python eval_fusion.py
```
