import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 112
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
PIN_MEMORY = True
USE_AMP = True

# ----- 112-MLP 训练（Swin ImageNet → FIW）-----
BATCH_SIZE = 128
VAL_BATCH_SIZE = 256
EPOCHS = 50
LR_HEAD = 1e-4
LR_HEAD_UNFREEZE = 1e-4
LR_BACKBONE = 2e-5
ETA_MIN = 1e-6
EARLY_STOP_PATIENCE = 5
FREEZE_BACKBONE_EPOCH = 3
PRETRAIN_BACKBONE = True
USE_EMA = False
EMA_DECAY = 0.9995
RESUME_START_EPOCH = 0
RESUME_BEST_AUC = 0.0
EXP_NAME = "112-imagenet-mlp-head"
SAVE_DIR = f"./weights/{EXP_NAME}/"
LOG_DIR = f"./logs/{EXP_NAME}/"
CKPT_BEST = "best.pth"
CKPT_LAST = "last.pth"

# DFC / 多任务（与 112-imagenet-mlp-head 一致）
TAU_CS = 0.08
TAU_ED = 2.0
A_DFC = 1.0
B_DFC = 0.0
HARD_WEIGHT = 1.5
HARD_THRESH = 0.6
ALPHA = 0.25
BETA = 0.10
GAMMA = 0.0
USE_MTFMF = True
USE_AGE_MASK = False
USE_WEIGHT_DFC = False
USE_FAM_BAL = True
FAM_BAL_POWER = 0.5
RANDOM_SEED = 42

# ----- 融合评测：冻 AdaFace -----
ADAFACE_ARCH = "ir_101"
ADAFACE_CKPT = "adaface_ir101_webface4m.ckpt"
FEAT_DIM = 512
MID_DIM = 256

DATA_DIR = "data/FIW"
FID_DIR = "FIDs"
PAIR_DIR = "pairs"
