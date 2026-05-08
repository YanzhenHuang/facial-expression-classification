import os
from typing import Callable
from pathlib import Path

from tqdm import tqdm

def recursive_search(_root_dir: str, f: Callable[[Path], None]):
    root_dir = Path(_root_dir)

    for _label_dir in os.listdir(root_dir):
        label_dir = Path.joinpath(root_dir, _label_dir)

        if not os.path.isdir(label_dir):
            continue

        # 收集该 label 下的所有图片文件
        image_files = []
        for _image_file in os.listdir(label_dir):
            image_file = Path.joinpath(label_dir, _image_file)
            image_files.append(image_file)

        # 使用 tqdm 显示进度
        for image_file in tqdm(image_files, desc=f"Processing {_label_dir}"):
            f(image_file)