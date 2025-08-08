# datasets/real3dad_convert.py
import os, json, glob
import numpy as np
import torch
from pathlib import Path
from pyntcloud import PyntCloud
from sklearn.cluster import DBSCAN


def load_pcd_xyzn(pcd_path: str):
    '''
    读取 PCD 文件，返回 (N,3) xyz float32 和 (N,) 语义 int32
    '''
    cloud = PyntCloud.from_file(pcd_path)
    df = cloud.points
    assert df.shape[1] == 4, f"{pcd_path} 需为 4 列 [x,y,z,semantic]"
    xyz = df.iloc[:, :3].to_numpy().astype(np.float32)
    sem = df.iloc[:, 3].to_numpy().astype(np.int32)
    if xyz.shape[0] == 0:
        raise ValueError(f"空点云: {pcd_path}")
    return xyz, sem


def pack_single_instance(
    xyz: np.ndarray,
    sem_pp: np.ndarray,
    bg_id: int = 0,
    keep_empty: bool = True,   # 是否为 0 实例场景生成“空监督”样本
    eps: float = 2.5,          # DBSCAN 聚类半径（可调）
    min_samples: int = 10,     # DBSCAN 最小聚类点数（可调）
):
    """
    输入:
      xyz: (N,3)
      sem_pp: (N,) 逐点语义, 背景=bg_id
    输出:
      dict 包含:
        - instance_labels_pp: (N,) 0/1/2/...（0为背景, 1~K为实例编号）
        - instance_masks: (K,N)  (K为实例数; 空场景时 K=0)
        - semantic_labels_inst: (K,)
      或者返回 None 表示丢弃该样本（用于 train 集跳过空场景）
    """
    if sem_pp is None or sem_pp.ndim != 1 or sem_pp.shape[0] != xyz.shape[0]:
        raise ValueError("缺少有效的逐点语义，无法监督训练")

    # 背景点
    bg_mask = (sem_pp == bg_id)
    instance_class_ids = [i for i in np.unique(sem_pp) if i != bg_id and i >= 0]

    # 只剩下背景
    if len(instance_class_ids) == 0:
        if keep_empty:
            return {
                "instance_labels_pp": np.zeros_like(sem_pp, dtype=np.int32),
                "instance_masks": np.zeros((0, xyz.shape[0]), dtype=bool),
                "semantic_labels_inst": np.zeros((0,), dtype=np.int32),
            }
        else:
            return None

    # ====== 基于空间聚类自动分割多个实例 ======
    instance_labels_pp = np.zeros_like(sem_pp, dtype=np.int32)
    instance_masks_list = []
    semantic_labels_inst = []
    cur_instance_id = 1

    for class_id in instance_class_ids:
        mask = (sem_pp == class_id)
        if not mask.any():
            continue
        xyz_def = xyz[mask]
        if xyz_def.shape[0] < min_samples:
            continue
        cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz_def)
        labels = cluster.labels_
        for inst_label in set(labels):
            if inst_label == -1:
                continue
            inst_mask_global = np.zeros_like(sem_pp, dtype=bool)
            inst_mask_local = (labels == inst_label)
            inst_mask_global[np.where(mask)[0][inst_mask_local]] = True
            instance_labels_pp[inst_mask_global] = cur_instance_id
            instance_masks_list.append(inst_mask_global)
            semantic_labels_inst.append(class_id)  # 直接用 class_id
            cur_instance_id += 1


    if len(instance_masks_list) == 0:
        # 全是噪声或极少点的情况
        if keep_empty:
            return {
                "instance_labels_pp": np.zeros_like(sem_pp, dtype=np.int32),
                "instance_masks": np.zeros((0, xyz.shape[0]), dtype=bool),
                "semantic_labels_inst": np.zeros((0,), dtype=np.int32),
            }
        else:
            return None

    instance_masks = np.stack(instance_masks_list, axis=0)

    return {
        "instance_labels_pp": instance_labels_pp,        # (N,) 0=背景, 1~K为实例编号
        "instance_masks": instance_masks,                # (K,N) bool
        "semantic_labels_inst": np.array(semantic_labels_inst, dtype=np.int32),       # (K,) 0/1/2
    }


def save_sample_pt(out_path: Path,
                   xyz, sem_pp, inst_pp, inst_masks, sem_labels_inst, meta: dict):
    '''
    保存单个样本到 .pt 文件
    '''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "points": xyz,                                # (N,3)
        "semantic_labels_pp": sem_pp,                 # (N,) 0/1/2
        "instance_labels_pp": inst_pp,                # (N,) 0/1/2/... (0=背景)
        "instance_masks": inst_masks,                 # (K,N) bool
        "semantic_labels_inst": sem_labels_inst,       # (K,) 0/1/2
        "meta": meta
    }
    torch.save(obj, str(out_path))


def generate_gt_file(out_root: Path, split: str, scene_id: str,
                     sem_pp: np.ndarray, inst_masks: np.ndarray):
    """
    写出 (N,) 的整数向量：
    gt_id = semantic_id * 1000 + instance_index(1-based; 无实例=0)
    - semantic_id: 0=good(作void), 1=bulge, 2=sink
    - inst_masks: (K,N) bool，K 可以为 0（空场景）
    """
    N = sem_pp.shape[0]
    inst_index_pp = np.zeros(N, dtype=np.int32)   # 每点实例号，默认 0
    for k in range(inst_masks.shape[0]):          # K==0 时不会进循环
        inst_index_pp[inst_masks[k]] = k + 1      # 1-based

    sem_for_gt = sem_pp.astype(np.int32)          # 直接写 0/1/2（0=void）
    gt_ids = sem_for_gt * 1000 + inst_index_pp    # (N,)

    gt_file_path = out_root / "instance_gt" / split / f"{scene_id}.txt"
    gt_file_path.parent.mkdir(parents=True, exist_ok=True)
    # 关键：保证一行一个整数
    np.savetxt(gt_file_path, gt_ids.reshape(-1, 1), fmt="%d")
    return gt_file_path


def convert_dataset(
    root="/data/Real3D-AD-PCD",
    out_root="/data/processed/real3dad",
    bg_id=0,
    split_rule=("train", "val", "test"),
    split_ratio=(0.88, 0.07, 0.05),
):
    root = Path(root)
    out_root = Path(out_root)
    assert abs(sum(split_ratio) - 1.0) < 1e-6

    # 构建类别映射
    categories = ['good', 'bulge', 'sink']
    categories = sorted(set(categories))
    category2id = {c: i for i, c in enumerate(categories)}
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    with open(out_root / "meta" / "category2id.json", "w", encoding="utf-8") as f:
        json.dump(category2id, f, ensure_ascii=False, indent=2)


    # 简化版本
    # 收集全部样本（含类型信息）
    all_samples = []
    for sub in sorted(os.listdir(root)):
        sub_dir = root / sub
        if not sub_dir.is_dir():
            continue
        pcds = sorted(glob.glob(str(sub_dir / "test_neo" / "*.pcd")))
        pcds = [p for p in pcds if "hybrid" not in os.path.basename(p)]
        for p in pcds:
            fname = os.path.basename(p).lower()
            cat = 'good'
            for d in categories[1:]:
                if d in fname:
                    cat = d
                    break
            all_samples.append((sub, cat, p))

    print(f"Total samples: {len(all_samples)}")

    # 打乱并分割
    rng = np.random.default_rng(2025)
    indices = np.arange(len(all_samples))
    rng.shuffle(indices)

    n_train = int(len(all_samples) * split_ratio[0])
    n_val = int(len(all_samples) * split_ratio[1])
    n_test = len(all_samples) - n_train - n_val

    split_indices = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:]
    }

    split_samples = {split: [all_samples[i] for i in idxs] for split, idxs in split_indices.items()}

    ###
    split_lists = {"train": [], "val": [], "test": []}
    skipped_train_empty = 0  # 统计：train 中因无实例而跳过的样本

    for split in ("train", "val", "test"):
        for sub, cat, pcd_path in split_samples[split]:

            scene_id = f"{sub}_{Path(pcd_path).stem}"

            # 读取点与逐点语义
            xyz, sem_pp = load_pcd_xyzn(pcd_path)

            # 1. 检查 shape 是否为 (N,3)，否则跳过
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                print(f"[SKIP] {scene_id} - xyz shape invalid: {xyz.shape}")
                continue

            # 2. 检查点数阈值，极少点或空点直接跳过
            if xyz.shape[0] < 10:
                print(f"[SKIP] {scene_id} - too few points: {xyz.shape[0]}")
                continue

            #
            sem_pp_train = sem_pp.copy()

            # 生成实例监督
            keep_empty = (split != "train")
            packed = pack_single_instance(xyz, sem_pp, bg_id=bg_id, keep_empty=keep_empty)

            if packed is None:
                if split == "train":
                    skipped_train_empty += 1
                    print(f"[WARN] {scene_id} ({split}) has no valid inst labels, skip.")
                    continue
                # else:
                #     print(f"[WARN] {scene_id} ({split}) has no valid labels, skip.")
                #     continue

            inst_pp   = packed["instance_labels_pp"]      # (N,)
            inst_masks= packed["instance_masks"]          # (K,N)
            sem_labels_inst = packed["semantic_labels_inst"]        # (K,)

            meta = {"file": pcd_path, "category": cat, "category_id": category2id[cat]}

            # 落盘
            out_file = out_root / split / f"{scene_id}.pt"
            save_sample_pt(out_file, xyz, sem_pp_train, inst_pp, inst_masks, sem_labels_inst, meta)

            # 生成并保存 GT 文件
            generate_gt_file(out_root, split, scene_id, sem_pp, inst_masks)

            # 只登记 scene_id
            split_lists[split].append(scene_id)

    # 保存划分清单
    with open(out_root / "meta" / "splits.json", "w", encoding="utf-8") as f:
        json.dump(split_lists, f, ensure_ascii=False, indent=2)

    print(
        f"Done. Saved to {out_root}. "
        f"train={len(split_lists['train'])}, val={len(split_lists['val'])}, test={len(split_lists['test'])} "
        f"(skipped_train_empty={skipped_train_empty})"
    )
    
convert_dataset()