# -- coding: utf-8 --
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import os
from itertools import islice
import torch
import csv
import chardet
from utils.config import config


class DatasetCFP(Dataset):
    """
    CFP 数据集加载器
    - 支持 train / val / test
    - 支持是否返回图像路径（用于 CAM / 可视化）
    """

    def __init__(self, root, mode='train', transform=None, return_path=False):
        """
        Args:
            root (str): fold_x 根目录
            mode (str): 'train' / 'val' / 'test'
            transform: 自定义 transform（可选）
            return_path (bool): 是否返回图像路径
        """
        self.root = root
        self.mode = mode
        self.return_path = return_path
        self.data_list = self._get_files()

        # ---------------------------
        # Transforms
        # ---------------------------
        if transform is not None:
            self.transforms = transform
        elif mode == 'train':
            self.transforms = T.Compose([
                T.Resize((config.img_height, config.img_weight)),
                T.RandomRotation(30),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:  # val / test
            self.transforms = T.Compose([
                T.Resize((config.img_height, config.img_weight)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def _get_files(self):
        """
        从 csv 文件中读取图像路径和标签
        """
        data_file = os.path.join(self.root, f'{self.mode}.csv')
        img_list = []

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"{data_file} not found!")

        raw_data = open(data_file, 'rb').read()
        encoding = chardet.detect(raw_data)['encoding']

        with open(data_file, 'r', encoding=encoding) as f:
            csv_reader = csv.reader(f)
            for line in islice(csv_reader, 1, None):  # 跳过表头
                try:
                    img_path = os.path.join(self.root, self.mode, line[0])
                    label = int(line[1])
                    if not os.path.exists(img_path):
                        print(f"[Warning] Image not found: {img_path}")
                        continue
                    img_list.append((img_path, label))
                except (IndexError, ValueError) as e:
                    print(f"[Error] Failed to parse line {line}: {e}")

        return img_list

    def __getitem__(self, index):
        img_path, label = self.data_list[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        if self.return_path:
            return img, label, img_path
        else:
            return img, label

    def __len__(self):
        return len(self.data_list)


def collate_fn(batch):
    """
    兼容是否返回 image path 的 collate_fn
    """
    if len(batch[0]) == 3:
        imgs, labels, paths = zip(*batch)
        return torch.stack(imgs, 0), list(labels), list(paths)
    else:
        imgs, labels = zip(*batch)
        return torch.stack(imgs, 0), list(labels)
