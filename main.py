import logging
import os
from hashlib import md5
from uuid import uuid4

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

# === 新增：引入我们的数据集 ===
from datasets.real3dad import Real3DADDataset


def get_parameters(cfg: DictConfig):
    """
    仍然保持你原来的参数解析/日志/ckpt 加载逻辑不变。
    """
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    # 如果保存目录已存在，则默认从 last-epoch 恢复（与你原逻辑一致）
    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    last_ckpt = os.path.join(cfg.general.save_dir, "last-epoch.ckpt")
    if os.path.exists(last_ckpt):
        print("EXPERIMENT ALREADY EXIST, FOUND CHECKPOINT")
        cfg["trainer"]["resume_from_checkpoint"] = last_ckpt

    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    model = InstanceSegmentation(cfg)

    # （可选）加载 backbone 或整模 ckpt（与你原逻辑一致）
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


# === 新增：仅当 dataset_name=real3dad 时，构建 dataloaders 与 collate_fn ===
def build_dataloaders_if_needed(cfg: DictConfig):
    """
    返回 (train_loader, val_loader, test_loader) 或 (None, None, None)
    - 当 cfg.data.dataset_name == "real3dad"（忽略大小写）时，构造并返回 dataloaders
    - 否则返回 (None, None, None)，表示沿用模型内部的 dataloaders
    """
    name = str(cfg.data.dataset_name).lower() if "dataset_name" in cfg.data else ""
    if name != "real3dad":
        return None, None, None

    # 读取必要配置（给默认值，避免配置里缺项报错）
    data_dir = cfg.data.get("data_dir", "data/processed/real3dad")
    voxel_size = float(cfg.data.get("voxel_size", 0.02))
    batch_size = int(cfg.training.get("batch_size", 2))
    eval_batch_size = int(cfg.training.get("eval_batch_size", 1))
    num_workers = int(cfg.training.get("num_workers", 4))
    pin_memory = bool(cfg.training.get("pin_memory", True))
    drop_last = bool(cfg.training.get("drop_last", False))

    # 构建 Dataset（仅返回单样本原子数据：量化坐标/特征/监督）
    train_ds = Real3DADDataset(root=data_dir, split="train", voxel_size=voxel_size)
    val_ds   = Real3DADDataset(root=data_dir, split="val",   voxel_size=voxel_size)
    test_ds  = Real3DADDataset(root=data_dir, split="test",  voxel_size=voxel_size)

    # 自定义 collate：对一个 batch 的样本做 sparse_collate，并把可变长监督保留为 list
    def collate_fn(batch):
        # 1) 收集当前 batch 内各样本的量化坐标与特征
        coords_list = [b["coords_q"] for b in batch]               # list of (Ni,3) int32
        feats_list  = [b["feats_f"]  for b in batch]               # list of (Ni,C) float32

        # 2) 用 MinkowskiEngine 把整批坐标/特征合并（自动在坐标前加 batch 维）
        coords, feats = ME.utils.sparse_collate(coords=coords_list, feats=feats_list)
        # coords: torch.IntTensor [sum(Ni), 1+3]，feats: torch.FloatTensor [sum(Ni), C]

        # 3) 其他监督（可变 K/N）保持为 list[Tensor]，避免默认 stack 报错
        batch_out = {
            "coords": coords,
            "feats": feats,
            "instance_masks": [torch.from_numpy(b["instance_masks"]) for b in batch],           # list of (Ki,Ni)
            "semantic_labels_inst": [torch.from_numpy(b["semantic_labels_inst"]) for b in batch], # list of (Ki,)
            "sem_labels_pp": [torch.from_numpy(b["sem_labels_pp"]) for b in batch],             # list of (Ni,)
            "inst_labels_pp": [torch.from_numpy(b["inst_labels_pp"]) for b in batch],           # list of (Ni,)
        }
        return batch_out

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, drop_last=drop_last
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, drop_last=False
    )

    return train_loader, val_loader, test_loader


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def train(cfg: DictConfig):
    # hydra 会切工作目录，这里切回工程根
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)

    # callbacks（你的原逻辑不变）
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks]
    callbacks.append(RegularCheckpointing())

    runner = Trainer(
        logger=loggers,
        gpus=cfg.general.gpus,
        callbacks=callbacks,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )

    # === 新增：当数据集是 real3dad 时，外部构建并传入 dataloaders ===
    train_loader, val_loader, _ = build_dataloaders_if_needed(cfg)
    if train_loader is None:
        # 回退：由模型内部的 dataloaders 决定（保持向后兼容）
        runner.fit(model)
    else:
        runner.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def test(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)

    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )

    # === 新增：当数据集是 real3dad 时，外部构建并传入 dataloaders ===
    _, val_loader, test_loader = build_dataloaders_if_needed(cfg)

    # 若提供了 checkpoint，就按 ckpt 测；否则按当前权重
    if cfg.general.get("checkpoint", None) is not None:
        # 这行保持与你的 get_parameters 一致（已在 get_parameters 中 load 过 ckpt）
        pass

    # 如果只想在 test split 上评估：
    if test_loader is not None:
        runner.test(model, dataloaders=test_loader)
    else:
        # 回退：由模型内部的 dataloaders 决定
        runner.test(model)


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def main(cfg: DictConfig):
    if cfg["general"]["train_mode"]:
        train(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()