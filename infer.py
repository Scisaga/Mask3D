#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
from typing import Optional
import hydra

import torch
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader

# 关键：复用你项目的参数装配 & 权重恢复逻辑
from main_instance_segmentation import get_parameters
from trainer.trainer import InstanceSegmentation, RegularCheckpointing

# 关键：按你仓库用法导入（若你仓库在 datasets.transforms，请自行改成 transforms）
from datasets.utils import VoxelizeCollate

# infer.py 顶部其它 import 下面补这行
from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)


# --------- PCD（ASCII）工具 ---------
def parse_pcd_ascii_xyz(pcd_text: str) -> np.ndarray:
    s = pcd_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]

    fields = points = None
    data_idx = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if low.startswith("fields"):
            fields = ln.split()[1:]
        elif low.startswith("points"):
            points = int(ln.split()[1])
        elif low.startswith("data"):
            if "ascii" not in low:
                raise ValueError("仅支持 ASCII PCD")
            data_idx = i + 1
            break
    if fields is None or points is None or data_idx is None:
        raise ValueError("PCD header 不完整")

    lf = [f.lower() for f in fields]
    ix, iy, iz = lf.index("x"), lf.index("y"), lf.index("z")

    data = lines[data_idx : data_idx + points]
    xyz = np.zeros((len(data), 3), dtype=np.float32)
    for i, ln in enumerate(data):
        cols = ln.split()
        xyz[i, 0] = float(cols[ix]); xyz[i, 1] = float(cols[iy]); xyz[i, 2] = float(cols[iz])
    return xyz


def make_pcd_ascii_xyz_sem_inst(xyz: np.ndarray, sem: np.ndarray, inst: np.ndarray) -> str:
    n = len(xyz)
    header = (
        "VERSION .7\nFIELDS x y z semantic instance\n"
        "SIZE 4 4 4 4 4\nTYPE F F F I I\nCOUNT 1 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA ascii\n"
    )
    buf = io.StringIO(); buf.write(header)
    for i in range(n):
        buf.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {int(sem[i])} {int(inst[i])}\n")
    return buf.getvalue()


# infer.py 中替换掉旧的 _SimplePCDDataset
class _InferPCDDataset(Dataset):
    def __init__(self, xyz: np.ndarray, cfg: DictConfig):
        self.xyz = xyz.astype(np.float32, copy=False)

        # 训练里 features 被当作 3 通道（utils.ensure_2d(..., last_dim=3)）
        # 所以这里强制只用 raw xyz 作为特征（3 通道），colors/normals 都关掉
        self.ignore_label = int(cfg.data.ignore_label)
        self.add_instance = bool(getattr(cfg.general, "add_instance", True))

    def __len__(self): return 1

    def __getitem__(self, idx):
        xyz = self.xyz  # (N,3) float32

        # labels: [sem,(inst)], 再拼 segment=0
        sem = np.full(len(xyz), self.ignore_label, dtype=np.int32)
        if self.add_instance:
            inst = np.full(len(xyz), -1, dtype=np.int32)
            labels = np.stack([sem, inst], axis=1)  # (N,2)
        else:
            labels = sem.reshape(-1, 1)            # (N,1)
        segments = np.zeros(len(xyz), dtype=np.int32)
        labels = np.hstack([labels, segments[:, None]])  # (N,2或3)

        # features：只放 raw xyz（3 通道），跟训练保持一致
        features = xyz.astype(np.float32)

        raw_coordinates = xyz.copy().astype(np.float32)
        scene_name = "inference"
        raw_color   = np.zeros((len(xyz), 3), dtype=np.float32)
        raw_normals = np.zeros((len(xyz), 3), dtype=np.float32)

        return (xyz.astype(np.float32),  # coordinates
                features,                # features (N,3)
                labels.astype(np.int32), # labels
                scene_name,              # file name
                raw_color,               # raw_color
                raw_normals,             # raw_normals
                raw_coordinates,         # raw_coordinates
                0)       

# --------- 可复用推理引擎：加载一次，多次字符串推理 ---------
class Mask3DInferencer:
    def __init__(self, cfg: DictConfig):
        # ✅ 只做必要的装配 + 权重恢复（与源码一致，但不引入训练期副作用）
        self.cfg = cfg
        model = InstanceSegmentation(cfg)

        if cfg.general.get("backbone_checkpoint"):
            cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
        if cfg.general.get("checkpoint"):
            cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

        self.model = model.eval()

        # 严格按你项目配置树读取体素化参数
        voxel_size   = OmegaConf.select(cfg, "data.voxel_size")
        ignore_label = OmegaConf.select(cfg, "data.ignore_label")
        if voxel_size is None or ignore_label is None:
            raise ValueError("缺少 data.voxel_size 或 data.ignore_label（请检查 conf/*）")

        # 用配置创建与训练一致的 Collate
        self.collate_fn = VoxelizeCollate(
            voxel_size=cfg.data.voxel_size,
            ignore_label=cfg.data.ignore_label,
            mode=getattr(cfg.data, "test_mode", "test"),
            task=cfg.general.task,  # "instance_segmentation"
        )

        # 需要 label 映射：实例化一次 validation_dataset（只读 label_info/offset）
        self.model.validation_dataset = hydra.utils.instantiate(cfg.data.validation_dataset)

        # 确保 forward 里 raw_coordinates 能用
        if not getattr(cfg.data, "add_raw_coordinates", False):
            cfg.data.add_raw_coordinates = True  # 强制打开（模型 forward 里需要）


        # 复用 Lightning Trainer（与你项目风格一致）
        self.trainer = Trainer(
            gpus=cfg.general.get("gpus", None),
            logger=False,
            enable_checkpointing=False,
            **cfg.trainer,
        )


    def predict_xyz(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        # 1) 构造与训练一致的样本 -> voxelize
        sample = _InferPCDDataset(xyz, self.cfg)[0]
        pack, target, filenames = self.collate_fn([sample])  # pack 是 NoGpu
        fname = filenames[0]

        # 2) 跑一次 eval_step（就是验证/测试同一条路径）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()
        with torch.no_grad():
            _ = self.model.eval_step((pack, target, filenames), batch_idx=0)

        # 3) 取出后处理好的预测（已 inverse_map 回到原分辨率）
        pred = self.model.preds.get(fname, None)
        if pred is None:
            # 模型可能在某些异常点云上返回空结果
            N = xyz.shape[0]
            return (np.full(N, self.cfg.data.ignore_label, dtype=np.int32),
                    np.full(N, -1, dtype=np.int32))

        # masks: 形状 (N, M)，N=点数，M=候选实例数；元素是 0/1（False/True），表示“点 i 是否属于实例 j”
        masks  = pred["pred_masks"]      

        # scores: 形状 (M,)，每个实例的置信度分数（通常来自分类得分与掩码质量的综合）
        scores = np.asarray(pred["pred_scores"])

        # classes: 形状 (M,)，每个实例的语义类别ID（已通过 _remap_model_output 做过ID映射到数据集原ID）
        classes= np.asarray(pred["pred_classes"])

        N = masks.shape[0]
        # 若没有任何候选实例（M=0），直接给所有点填“忽略语义 + 无实例”
        if masks.shape[1] == 0:
            return (np.full(N, self.cfg.data.ignore_label, dtype=np.int32),
                    np.full(N, -1, dtype=np.int32))

        # --- 将实例掩码 + 分数，折算成“每个点属于哪个实例” ---

        # 转成布尔，确保是 True/False
        masks_bool = masks.astype(bool)

        # 对每个点，把“不覆盖该点的实例”的分数屏蔽成 -inf；覆盖该点的实例保留其 score
        # 这样对每个点做 argmax，就能选到 “覆盖该点且分数最高” 的那个实例
        # 形状广播：masks_bool (N, M) 与 scores (M,) -> masked_scores (N, M)
        masked_scores = np.where(masks_bool, scores[None, :], -np.inf)

        # j_max[i] = 点 i 选择的实例索引（分数最高的那个 j）
        # 注意：如果某点完全不被任何实例覆盖，masked_scores[i,:] 全是 -inf，
        #       np.argmax 形式上会返回 0，但我们下面用 has_inst 屏蔽这种情况
        j_max = np.argmax(masked_scores, axis=1)

        # has_inst[i] = 该点是否被至少一个实例覆盖
        has_inst = masks_bool.any(axis=1)

        # --- 把选择到的实例索引 j_max（原本是0..M-1）压缩成“连续的实例ID 0..K-1” ---
        # 这样做的好处：只给实际被选中的实例分配ID，且ID从0开始、连续；没被任何点选中的实例不会出现
        uniq_js, remap = np.unique(j_max[has_inst], return_inverse=True)
        # uniq_js: 按升序排序后的“被用到的实例索引”列表（值域在 0..M-1）
        # remap  : 把 j_max[has_inst] 映射到 0..K-1 的新ID（K=len(uniq_js)）

        # 初始化 inst 为 -1（没实例）
        inst = np.full(N, -1, dtype=np.int32)
        # 只给 “被覆盖的点” 填新实例ID
        inst[has_inst] = remap.astype(np.int32)

        # 语义标签，默认填 ignore_label（比如 255）
        sem = np.full(N, self.cfg.data.ignore_label, dtype=np.int32)

        # 如果确实有被选中的实例
        if uniq_js.size > 0:
            # 对“被覆盖的点”，取其对应的实例 j_max[i] 的语义类别当作该点的语义
            # 注意：classes 是按原实例顺序（0..M-1），j_max 也是这个索引空间
            sem[has_inst] = classes[j_max[has_inst]].astype(np.int32)

        return sem, inst


    def predict_pcd_string(self, pcd_text: str) -> str:
        xyz = parse_pcd_ascii_xyz(pcd_text)
        sem, inst = self.predict_xyz(xyz)
        return make_pcd_ascii_xyz_sem_inst(xyz, sem, inst)