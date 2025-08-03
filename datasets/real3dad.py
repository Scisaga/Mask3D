# datasets/real3dad.py
import os
import os.path as osp
import json
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_category2id(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _masks_to_instance_ids(inst_masks: np.ndarray) -> np.ndarray:
    """(K,N) 实例掩码 -> (N,) 实例ID（无实例点=-1）。"""
    inst_masks = np.asarray(inst_masks)
    if inst_masks.size == 0:
        return np.full(0, -1, dtype=np.int64)
    any_pos = inst_masks.any(axis=0)
    inst_ids = np.full(inst_masks.shape[1], -1, dtype=np.int64)
    inst_ids[any_pos] = inst_masks[:, any_pos].argmax(axis=0).astype(np.int64)
    return inst_ids


class Real3DADDataset(Dataset):
    """
    读取 convert_real3dad_to_mask3d.py 产出的 .pt 样本，返回：
      (coordinates, features, labels, scene_name, raw_color, raw_normals, raw_coordinates, idx)

    - coordinates: (N,3) float32   原始坐标（不做体素量化）
    - features:    (N,C) float32   由 add_colors/add_normals/add_raw_coordinates 组合
    - labels:      (N,2 或 3) int32
        若 add_instance=True  -> [semantic, instance, segment]
        若 add_instance=False -> [semantic, segment]
      其中 segment 这里恒为 0（无 superpoint）
    - scene_name:  用文件名（去扩展名）
    """

    def __init__(
        self,
        dataset_name: str = "real3dad",
        data_dir: Union[str, Path] = "/data/processed/real3dad",
        mode: str = "train",                   # "train" | "val" | "test"
        voxel_size: float = 0.02,              # 仅占位，数据集内不使用
        ignore_label: int = 255,
        num_labels: int = -1,                  # 仅占位，使用 category2id.json 推断
        add_raw_coordinates: bool = True,
        add_colors: bool = False,
        add_normals: bool = False,
        add_instance: bool = True,
        filter_out_classes: List[int] = None,
        label_offset: int = 0,
        # 兼容上层可能存在的参数（不使用）
        **kwargs,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.voxel_size = float(voxel_size)
        self.ignore_label = int(ignore_label)
        self.add_raw_coordinates = bool(add_raw_coordinates)
        self.add_colors = bool(add_colors)
        self.add_normals = bool(add_normals)
        self.add_instance = bool(add_instance)
        self.filter_out_classes = set(int(x) for x in (filter_out_classes or []))
        self.label_offset = int(label_offset)

        split_dir = self.data_dir / self.mode
        assert split_dir.is_dir(), f"[Real3DADDataset] split dir not found: {split_dir}"

        # 优先用 splits.json；没有则直接枚举目录下的 .pt
        splits_json = self.data_dir / "meta" / "splits.json"
        if splits_json.exists():
            with open(splits_json, "r", encoding="utf-8") as f:
                splits = json.load(f)
            names = splits.get(self.mode, [])
            files = [str(split_dir / f"{n}.pt") for n in names if (split_dir / f"{n}.pt").exists()]
        else:
            files = sorted([str(split_dir / f) for f in os.listdir(split_dir) if f.endswith(".pt")])

        assert len(files) > 0, f"[Real3DADDataset] no .pt under {split_dir}"

        # 统一成 _data 列表（与参考实现类似）
        self._data = [
            {
                "filepath": fp,
                "raw_filepath": fp,
                "scene": Path(fp).stem,
            }
            for fp in files
        ]

        # 从 category2id.json 构建 label_info（键为语义 id，全部 validation=True）
        c2i_path = self.data_dir / "meta" / "category2id.json"
        assert c2i_path.exists(), f"[Real3DADDataset] missing {c2i_path}"
        cat2id = _load_category2id(c2i_path)
        all_ids = sorted(int(v) for v in cat2id.values())
        all_ids = [i for i in all_ids if i not in self.filter_out_classes]
        # 简单配色
        palette = [[180, 180, 180], [255, 0, 0], [0, 255, 0], [0, 128, 255], [255, 128, 0], [128, 0, 255]]
        self._labels = {
            cid: {"id": cid, "color": palette[k % len(palette)], "validation": True}
            for k, cid in enumerate(all_ids)
        }
        self.num_classes = len(self._labels)

    # ---- 兼容上层访问属性 ----
    @property
    def data(self):
        return self._data

    @property
    def label_info(self):
        return self._labels

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        rec = self._data[idx]
        obj = torch.load(rec["filepath"], map_location="cpu")

        # 基础字段
        xyz = np.asarray(obj["points"], dtype=np.float32, order="C")                # (N,3)
        sem = np.asarray(obj["semantic_labels_pp"], dtype=np.int64, order="C")      # (N,)

        # 过滤类别（若配置需要）
        if self.filter_out_classes:
            keep = ~np.isin(sem, list(self.filter_out_classes))
            xyz = xyz[keep]
            sem = sem[keep]

        # 实例标签
        if self.add_instance:
            if "instance_labels_pp" in obj and obj["instance_labels_pp"] is not None \
               and len(obj["instance_labels_pp"]) == len(sem):
                inst = np.asarray(obj["instance_labels_pp"], dtype=np.int64, order="C")
                if self.filter_out_classes:
                    inst = inst[keep]
            elif "instance_masks" in obj:
                inst = _masks_to_instance_ids(np.asarray(obj["instance_masks"], dtype=bool, order="C"))
                if len(inst) != len(sem):
                    inst = np.full(len(sem), -1, dtype=np.int64)
            else:
                inst = np.full(len(sem), -1, dtype=np.int64)
        else:
            inst = None

        # 处理 ignore / label_offset
        sem = sem.copy()
        sem[(sem == -1) | (sem == 255)] = self.ignore_label
        if self.label_offset != 0:
            valid = sem != self.ignore_label
            sem[valid] = sem[valid] + self.label_offset

        # segments（无 superpoint，统一置 0）
        segments = np.zeros(len(sem), dtype=np.int32)

        # 组装 labels（与参考语义/实例格式对齐）
        if self.add_instance:
            labels = np.stack([sem.astype(np.int32), inst.astype(np.int32)], axis=1)  # (N,2)
        else:
            labels = sem.astype(np.int32).reshape(-1, 1)                              # (N,1)
        labels = np.hstack([labels, segments[:, None]])                               # (N,2或3)

        # 特征：按需拼接；没有指定任何特征时，给出常数 1 通道
        feats_list = []
        if self.add_colors:
            # 无真实颜色时用 0 填充（N,3）
            feats_list.append(np.zeros((len(xyz), 3), dtype=np.float32))
            raw_color = feats_list[-1].copy()
        else:
            raw_color = np.zeros((len(xyz), 3), dtype=np.float32)

        if self.add_normals:
            feats_list.append(np.zeros((len(xyz), 3), dtype=np.float32))
            raw_normals = feats_list[-1].copy()
        else:
            raw_normals = np.zeros((len(xyz), 3), dtype=np.float32)

        if self.add_raw_coordinates:
            feats_list.append(xyz.astype(np.float32))

        if len(feats_list) == 0:
            features = np.ones((len(xyz), 1), dtype=np.float32)
        else:
            features = np.hstack(feats_list).astype(np.float32)

        raw_coordinates = xyz.copy().astype(np.float32)
        scene_name = Path(rec["raw_filepath"]).stem  # 用文件名作为场景名

        # print(f"[real3dad.py] __getitem__ idx={idx} feature shape: {features.shape}")

        return (
            xyz.astype(np.float32),     # coordinates
            features,                   # features
            labels.astype(np.int32),    # labels
            scene_name,                 # scene name
            raw_color,                  # raw_color
            raw_normals,                # raw_normals
            raw_coordinates,            # raw_coordinates
            idx,                        # index
        )
    
    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped
