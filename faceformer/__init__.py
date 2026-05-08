import torch

NUM_POINTS = 478    # MediaPipe人脸固定关键点
DIM_MODEL = 256     # Transformer维度
NUM_HEADS = 4       # 注意力头数
HEAD_DIM = DIM_MODEL // NUM_HEADS
NUM_CLASSES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"