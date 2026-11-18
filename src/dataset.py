import json
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def load_object_map(objects_json: Union[str, Path]) -> dict[str, int]:
    """讀取 objects.json，回傳物件名稱到類別索引 (0~23) 的對照表。"""
    with open(objects_json, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_labels(
    label_names: list[str],
    object_map: dict[str, int],
) -> torch.Tensor:
    """將一組物件名稱列表轉成 (num_classes,) 的 multi-hot float32 向量。"""
    vec = torch.zeros(len(object_map), dtype=torch.float32)
    for name in label_names:
        if name not in object_map:
            raise KeyError(f"Unknown object label '{name}' not found in objects.json.")
        vec[object_map[name]] = 1.0
    return vec


def load_condition_file(
    json_path: Union[str, Path],
    objects_json: Union[str, Path],
) -> torch.Tensor:
    """讀取 test.json 或 new_test.json，依原順序回傳 (N, 24) multi-hot 條件張量。"""
    object_map = load_object_map(objects_json)
    with open(json_path, "r", encoding="utf-8") as f:
        label_lists = json.load(f)

    labels = [encode_labels(names, object_map) for names in label_lists]
    return torch.stack(labels, dim=0)


class ICLEVRDataset(Dataset):
    """i-CLEVR 訓練資料集：回傳正規化至 [-1, 1] 的 RGB 圖片與 (24,) multi-hot 條件向量。"""

    def __init__(
        self,
        data_dir: Union[str, Path],
        train_json: Union[str, Path],
        objects_json: Union[str, Path],
        image_size: int = 64,
        random_flip: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = Path(data_dir)
        self.object_map = load_object_map(objects_json)

        with open(train_json, "r", encoding="utf-8") as f:
            raw_annotations: dict[str, list[str]] = json.load(f)

        self.samples: list[tuple[Path, torch.Tensor]] = [
            (self.data_dir / filename, encode_labels(label_names, self.object_map))
            for filename, label_names in raw_annotations.items()
        ]

        if transform is not None:
            self.transform = transform
        else:
            tf_list = [transforms.Resize((image_size, image_size))]
            if random_flip:
                tf_list.append(transforms.RandomHorizontalFlip(p=0.5))
            tf_list.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            self.transform = transforms.Compose(tf_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            image = self.transform(img.convert("RGB"))
        return image, label


def build_dataloader(
    data_dir: Union[str, Path],
    train_json: Union[str, Path],
    objects_json: Union[str, Path],
    batch_size: int = 64,
    image_size: int = 64,
    num_workers: int = 4,
    random_flip: bool = True,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """建立 i-CLEVR 訓練用 DataLoader。"""
    dataset = ICLEVRDataset(
        data_dir=data_dir,
        train_json=train_json,
        objects_json=objects_json,
        image_size=image_size,
        random_flip=random_flip,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
