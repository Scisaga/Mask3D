# datasets/real3dad_convert.py
import os, json, glob
import numpy as np
import torch
from pathlib import Path
from pyntcloud import PyntCloud
from sklearn.cluster import DBSCAN


def load_pcd_xyzn(pcd_path: str):
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

    # 允许把 -1/255 视作忽略标签（若存在）
    ignore_mask = (sem_pp == -1) | (sem_pp == 255)
    valid_mask = ~ignore_mask
    if not valid_mask.any():
        return None  # 全部不可用

    # 缺陷实例 = 非背景且有效
    defect_mask = (sem_pp != bg_id) & valid_mask

    if not defect_mask.any():
        # 0 实例场景
        if keep_empty:
            return {
                "instance_labels_pp": np.zeros_like(sem_pp, dtype=np.int32),
                "instance_masks": np.zeros((0, xyz.shape[0]), dtype=bool),
                "semantic_labels_inst_raw": np.zeros((0,), dtype=np.int32),
                "semantic_labels_inst": np.zeros((0,), dtype=np.int32),
            }
        else:
            return None  # 在 train 集跳过

    # ====== 改动：基于空间聚类自动分割多个实例 ======
    instance_labels_pp = np.zeros_like(sem_pp, dtype=np.int32)
    instance_masks_list = []
    semantic_labels_inst = []
    semantic_labels_inst_raw = []
    cur_instance_id = 1

    for defect_id, (cat_id, cat_model) in zip([1, 2], [(1, 0), (2, 1)]):  # (原始, 训练用)
        mask = (sem_pp == defect_id)
        if not mask.any():
            continue
        xyz_def = xyz[mask]
        if xyz_def.shape[0] < min_samples:
            continue
        # DBSCAN 聚类
        cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz_def)
        labels = cluster.labels_  # -1为噪声
        for inst_label in set(labels):
            if inst_label == -1:
                continue  # 忽略噪声
            # 全局点索引（mask对应原始点云中的下标）
            inst_mask_global = np.zeros_like(sem_pp, dtype=bool)
            inst_mask_local = (labels == inst_label)
            inst_mask_global[np.where(mask)[0][inst_mask_local]] = True
            instance_labels_pp[inst_mask_global] = cur_instance_id
            instance_masks_list.append(inst_mask_global)
            semantic_labels_inst.append(cat_model)
            semantic_labels_inst_raw.append(cat_id)
            cur_instance_id += 1

    if len(instance_masks_list) == 0:
        # 全是噪声或极少点的情况
        if keep_empty:
            return {
                "instance_labels_pp": np.zeros_like(sem_pp, dtype=np.int32),
                "instance_masks": np.zeros((0, xyz.shape[0]), dtype=bool),
                "semantic_labels_inst_raw": np.zeros((0,), dtype=np.int32),
                "semantic_labels_inst": np.zeros((0,), dtype=np.int32),
            }
        else:
            return None

    instance_masks = np.stack(instance_masks_list, axis=0)

    return {
        "instance_labels_pp": instance_labels_pp,        # (N,) 0=背景, 1~K为实例编号
        "instance_masks": instance_masks,                # (K,N) bool
        "semantic_labels_inst_raw": np.array(semantic_labels_inst_raw, dtype=np.int32),   # (K,) 1/2 -> 评估用
        "semantic_labels_inst":     np.array(semantic_labels_inst, dtype=np.int32),       # (K,) 0/1 -> 训练用
    }


def save_sample_pt(out_path: Path,
                   xyz, sem_pp_train, inst_pp, inst_masks, sem_inst_model, sem_inst_raw, meta: dict):
    '''
    保存单个样本到 .pt 文件
    '''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "points": xyz,                                # (N,3)
        "semantic_labels_pp": sem_pp_train,           # (N,) 255/0/1  (255忽略, 0=bulge, 1=sink)
        "instance_labels_pp": inst_pp,                # (N,) 0/1
        "instance_masks": inst_masks,                 # (K,N) bool
        "semantic_labels_inst": sem_inst_model,       # (K,) 0/1  <- 训练用
        "semantic_labels_inst_eval": sem_inst_raw,    # (K,) 1/2  <- 评估用（备份）
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

    #
    #
    # 采集文件（过滤 hybrid），按工件类型分层采样
    samples_by_type = dict()  # key: 工件类型（sub），value: [(cat, pcd_path)]
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
            samples_by_type.setdefault(sub, []).append((cat, p))

    # 分层采样
    split_samples = {"train": [], "val": [], "test": []}
    rng = np.random.default_rng(2025)
    for type_name, samples_one_type in samples_by_type.items():
        # 按缺陷类别分类
        by_defect = {"bulge": [], "sink": [], "good": []}
        for cat, pcd_path in samples_one_type:
            by_defect[cat].append((cat, pcd_path))

        assigned = {"train": [], "val": [], "test": []}

        # 1. 先确保 train 和 test 各有 bulge 和 sink（如果样本数允许）
        for defect in ["bulge", "sink"]:
            arr = by_defect[defect]
            rng.shuffle(arr)
            if len(arr) >= 2:
                assigned["train"].append(arr[0])
                assigned["test"].append(arr[1])
                remain = arr[2:]
            elif len(arr) == 1:
                assigned["train"].append(arr[0])
                remain = []
            else:
                remain = []
            by_defect[defect] = remain  # 剩余待后续分配

        # 2. 合并所有剩余样本（good+bulge+sink），按比例切分到三集
        remain_all = by_defect["bulge"] + by_defect["sink"] + by_defect["good"]
        n_remain = len(remain_all)
        n_train = int(n_remain * split_ratio[0])
        n_val = int(n_remain * split_ratio[1])
        indices = np.arange(n_remain)
        rng.shuffle(indices)
        for i in indices[:n_train]:
            assigned["train"].append(remain_all[i])
        for i in indices[n_train:n_train + n_val]:
            assigned["val"].append(remain_all[i])
        for i in indices[n_train + n_val:]:
            assigned["test"].append(remain_all[i])

        # 3. 汇总到全局 split_samples
        for split in ("train", "val", "test"):
            split_samples[split].extend(assigned[split])


    #
    # 简化版本
    # 收集全部样本（含类型信息）
    # all_samples = []
    # for sub in sorted(os.listdir(root)):
    #     sub_dir = root / sub
    #     if not sub_dir.is_dir():
    #         continue
    #     pcds = sorted(glob.glob(str(sub_dir / "test_neo" / "*.pcd")))
    #     pcds = [p for p in pcds if "hybrid" not in os.path.basename(p)]
    #     for p in pcds:
    #         fname = os.path.basename(p).lower()
    #         cat = 'good'
    #         for d in categories[1:]:
    #             if d in fname:
    #                 cat = d
    #                 break
    #         all_samples.append((sub, cat, p))

    # print(f"Total samples: {len(all_samples)}")

    # # 打乱并分割
    # rng = np.random.default_rng(2025)
    # indices = np.arange(len(all_samples))
    # rng.shuffle(indices)

    # n_train = int(len(all_samples) * split_ratio[0])
    # n_val = int(len(all_samples) * split_ratio[1])
    # n_test = len(all_samples) - n_train - n_val

    # split_indices = {
    #     "train": indices[:n_train],
    #     "val": indices[n_train:n_train + n_val],
    #     "test": indices[n_train + n_val:]
    # }

    # split_samples = {split: [all_samples[i] for i in idxs] for split, idxs in split_indices.items()}

    
    ###
    ###
    ###
    split_lists = {"train": [], "val": [], "test": []}
    skipped_train_empty = 0  # 统计：train 中因无实例而跳过的样本

    for split in ("train", "val", "test"):
        for cat, pcd_path in split_samples[split]:

            folder = os.path.basename(os.path.dirname(os.path.dirname(pcd_path)))
            scene_id = folder + "_" +Path(pcd_path).stem

            # 读取点与逐点语义
            xyz, sem_pp = load_pcd_xyzn(pcd_path)

            # >>>>> 加入以下验证逻辑 <<<<<
            # 1. 检查 shape 是否为 (N,3)，否则跳过
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                print(f"[SKIP] {scene_id} - xyz shape invalid: {xyz.shape}")
                continue

            # 2. 检查点数阈值，极少点或空点直接跳过
            if xyz.shape[0] < 10:
                print(f"[SKIP] {scene_id} - too few points: {xyz.shape[0]}")
                continue
            # <<<<< 结束验证逻辑 >>>>>


            # 训练用逐点语义：good(0)->255, bulge(1)->0, sink(2)->1
            sem_pp_train = sem_pp.copy()
            sem_pp_train[sem_pp_train == 0] = 255
            sem_pp_train[sem_pp_train == 1] = 0
            sem_pp_train[sem_pp_train == 2] = 1

            # 生成实例监督
            keep_empty = (split != "train")
            packed = pack_single_instance(xyz, sem_pp, bg_id=bg_id, keep_empty=keep_empty)

            if packed is None:
                if split == "train":
                    skipped_train_empty += 1
                    print(f"[WARN] {scene_id} ({split}) has no valid labels, skip.")
                    continue
                # else:
                #     print(f"[WARN] {scene_id} ({split}) has no valid labels, skip.")
                #     continue

            inst_pp   = packed["instance_labels_pp"]      # (N,)
            inst_masks= packed["instance_masks"]          # (K,N)
            sem_inst_raw   = packed["semantic_labels_inst_raw"]    # (K,)
            sem_inst_model = packed["semantic_labels_inst"]        # (K,)

            meta = {"file": pcd_path, "category": cat, "category_id": category2id[cat]}

            # 落盘
            out_file = out_root / split / f"{scene_id}.pt"
            save_sample_pt(out_file, xyz, sem_pp_train, inst_pp, inst_masks, sem_inst_model, sem_inst_raw, meta)

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