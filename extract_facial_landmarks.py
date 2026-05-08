import os
import numpy as np
from typing import Callable
from pathlib import Path
import h5py
import cv2
import mediapipe as mp
from tqdm import tqdm

MODEL_PATH = "models/face_landmarker.task"
IMAGES_PATH = "./data/images"
LANDMARKS_PATH = "./data/landmarks"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    output_facial_transformation_matrixes=True  # 启用变换矩阵输出
)

detector = FaceLandmarker.create_from_options(options)

def recursive_search(_root_dir: str, f: Callable[[Path], None]):
    root_dir = Path(_root_dir)

    for _label_dir in os.listdir(root_dir):
        label_dir = Path.joinpath(root_dir, _label_dir)

        if not os.path.isdir(label_dir):
            continue

        # 收集该 label 下的所有图片文件
        image_files = []
        for _image_file in os.listdir(label_dir):
            if Path(_image_file).suffix in [".jpg", ".png"]:
                image_file = Path.joinpath(label_dir, _image_file)
                image_files.append(image_file)
        
        # 使用 tqdm 显示进度
        for image_file in tqdm(image_files, desc=f"Processing {_label_dir}"):
            f(image_file)


def extract(image_path_: Path):
    image_path = Path(image_path_)

    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        return
    
    h, w = img.shape[:2]
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建 MediaPipe Image
    mp_image = mp.Image(data=rgb_img, image_format=mp.ImageFormat.SRGB)

    result = detector.detect(mp_image)

    # 检查是否检测到人脸
    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return

    # 获取第一张人脸的关键点
    landmarks = result.face_landmarks[0]

    # 转换为 numpy 数组 (N, 3) 形状：x, y, z（归一化坐标）
    landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])

    # 获取变换矩阵
    transformation_matrix = np.array(result.facial_transformation_matrixes[0])

    # 构建输出路径（保持目录结构）
    label_name = image_path_.parent.stem
    split_name = image_path_.parent.parent.stem

    result_filename = f"{image_path.stem}_{h}x{w}.hdf5"

    destination = Path.joinpath(Path("data/landmarks"), split_name, label_name, result_filename)
    
    # 创建输出目录
    destination.parent.mkdir(parents=True, exist_ok=True)

    # 使用 HDF5 格式保存
    with h5py.File(destination, 'w') as f:
        f.create_dataset('landmarks', data=landmarks_array, compression='gzip')
        f.create_dataset('transformation_matrix', data=transformation_matrix, compression='gzip')
    

recursive_search(str(Path(IMAGES_PATH, "Train")), extract)
recursive_search(str(Path(IMAGES_PATH, "Test")), extract)
