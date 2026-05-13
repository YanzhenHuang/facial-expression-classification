import os
import random
from pathlib import Path
from typing import DefaultDict

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

from proj import project_root
from utils import recursive_search

LABEL_NAMES = {word: i for i, word in
               enumerate([
                   "anger", "contempt", "disgust", "fear",
                   "happy", "neutral", "sad", "surprise"])}

NUM_POINTS = 478


class FileLoader:

    def __init__(self):
        self.labeled_datalist = DefaultDict(list)

    def _load(self, data_path: Path):
        with h5py.File(data_path, "r") as f:
            landmarks = np.array(f["landmarks"][:], dtype=np.float32)
            trans_matrix = np.array(f["transformation_matrix"][:], dtype=np.float32)

        ones = np.ones((NUM_POINTS, 1), dtype=np.float32)

        # [487, 3] -> [487, 4]
        landmarks_homo = np.concatenate([landmarks, ones], axis=1)  # type: ignore

        # 转换成正脸 [487, 3]
        aligned_landmarks = np.dot(landmarks_homo, trans_matrix.T)[:, :3]

        # 第五个landmark是鼻尖，遂转换成相对于第5个landmark的坐标，
        # 即训练的是从第5个landmark映射到其它landmark的向量束
        first_landmark = aligned_landmarks[4]
        aligned_landmarks -= first_landmark

        label = data_path.parent
        self.labeled_datalist[LABEL_NAMES[label.name]].append(aligned_landmarks)

    def load_split(self, split: str):
        """
        Load the entire split with all the labels.
        Generate two lists: Data list, and label list.
        """
        recursive_search(os.path.join(str(project_root), f"data/landmarks/{split}"), self._load)
        final_datalist = []
        final_labels = []

        for label, data_list in self.labeled_datalist.items():
            label_repeat = [label for _ in data_list]
            final_datalist.extend(data_list)
            final_labels.extend(label_repeat)

        combined = list(zip(final_datalist, final_labels))
        random.seed(42)
        random.shuffle(combined)
        final_datalist, final_labels = zip(*combined)

        return final_datalist, final_labels


class FaceDataset(Dataset):
    def __init__(self, data_list, labels=None):
        super().__init__()
        self.data = np.array(data_list)  # [N, 478, 3]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # 输出：对齐后的3D关键点 [478,3]，标签
        coords = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return coords, label
