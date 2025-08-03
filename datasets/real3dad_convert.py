# convert_real3dad_to_mask3d.py
import os, json, glob
import numpy as np
import torch
from pathlib import Path
from pyntcloud import PyntCloud

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
):
    """
    输入:
      xyz: (N,3)
      sem_pp: (N,) 逐点语义, 背景=bg_id
    输出:
      dict 包含:
        - instance_labels_pp: (N,) 0/1
        - instance_masks: (K,N)  (默认 K=1; 空场景时 K=0)
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

    # 正常单实例场景
    inst_pp = np.zeros_like(sem_pp, dtype=np.int32)
    inst_pp[defect_mask] = 1
    inst_masks = np.zeros((1, xyz.shape[0]), dtype=bool)
    inst_masks[0, defect_mask] = True

    # 若缺陷语义不唯一，取多数票 -> 原始实例语义（1=bulge, 2=sink）
    vals = sem_pp[defect_mask]
    sem_inst_raw = int(np.bincount(vals).argmax())  # ∈ {1,2}

    # 训练用实例语义（0/1）
    if sem_inst_raw not in (1, 2):
        raise ValueError(f"未知实例语义 {sem_inst_raw}，应为 1(bulge)/2(sink)")
    
    sem_inst_model = 0 if sem_inst_raw == 1 else 1

    return {
        "instance_labels_pp": inst_pp,          # (N,) 0/1 (是否属于实例)
        "instance_masks": inst_masks,           # (K,N) bool
        "semantic_labels_inst_raw": np.array([sem_inst_raw], dtype=np.int32),   # (K,) 1/2 -> 评估用
        "semantic_labels_inst":     np.array([sem_inst_model], dtype=np.int32), # (K,) 0/1 -> 训练用
    }

def save_sample_pt(out_path: Path,
                   xyz, sem_pp_train, inst_pp, inst_masks, sem_inst_model, sem_inst_raw, meta: dict):
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
    split_ratio=(0.8, 0.1, 0.1),
):
    root = Path(root)
    out_root = Path(out_root)
    assert abs(sum(split_ratio) - 1.0) < 1e-6

    # 采集文件（过滤 hybrid）
    samples = []
    categories = ['good', 'bulge', 'sink']

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
            samples.append((cat, p))

    # 构建类别映射
    categories = sorted(set(categories))
    category2id = {c: i for i, c in enumerate(categories)}
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    with open(out_root / "meta" / "category2id.json", "w", encoding="utf-8") as f:
        json.dump(category2id, f, ensure_ascii=False, indent=2)

    # 划分（按样本索引打乱后切片）
    n = len(samples)
    n_train = int(n * split_ratio[0]); n_val = int(n * split_ratio[1])
    indices = np.arange(n)
    rng = np.random.default_rng(2025)
    rng.shuffle(indices)
    split_bins = {
        "train": set(indices[:n_train]),
        "val":   set(indices[n_train:n_train+n_val]),
        "test":  set(indices[n_train+n_val:])
    }

    split_lists = {"train": [], "val": [], "test": []}
    skipped_train_empty = 0  # 统计：train 中因无实例而跳过的样本

    for idx, (cat, pcd_path) in enumerate(samples):
        split = "train" if idx in split_bins["train"] else ("val" if idx in split_bins["val"] else "test")
        scene_id = Path(pcd_path).stem

        # 读取点与逐点语义
        xyz, sem_pp = load_pcd_xyzn(pcd_path)

        # 训练用逐点语义：good(0)->255, bulge(1)->0, sink(2)->1
        sem_pp_train = sem_pp.copy()
        sem_pp_train[sem_pp_train == 0] = 255
        sem_pp_train[sem_pp_train == 1] = 0
        sem_pp_train[sem_pp_train == 2] = 1

        # 生成实例监督：
        # - train：keep_empty=False（无缺陷的样本直接丢弃，不参与训练）
        # - val/test：keep_empty=True（保留为空监督样本，用于评估误检）
        keep_empty = (split != "train")
        packed = pack_single_instance(xyz, sem_pp, bg_id=bg_id, keep_empty=keep_empty)

        # pack_single_instance 可能返回 None（无有效标注或 train 空场景）
        if packed is None:
            if split == "train":
                skipped_train_empty += 1
                print(f"[WARN] {scene_id} ({split}) has no valid labels, skip.")
                # 训练集：空场景/无效标注 -> 跳过
                continue
            else:
                # 理论上 val/test 传 keep_empty=True 不会返回 None，这里做容错
                print(f"[WARN] {scene_id} ({split}) has no valid labels, skip.")
                continue

        # 从 dict 里取出三个字段（代替错误的“解包为三变量”）
        inst_pp   = packed["instance_labels_pp"]      # (N,)
        inst_masks= packed["instance_masks"]          # (K,N)，单实例时 K=1；空场景时 K=0
        sem_inst_raw   = packed["semantic_labels_inst_raw"]    # (K,) 1/2
        sem_inst_model = packed["semantic_labels_inst"]        # (K,) 0/1

        meta = {"file": pcd_path, "category": cat, "category_id": category2id[cat]}

        # 落盘
        out_file = out_root / split / f"{scene_id}.pt"
        save_sample_pt(out_file, xyz, sem_pp_train, inst_pp, inst_masks, sem_inst_model, sem_inst_raw, meta)

         # 生成并保存 GT 文件
        generate_gt_file(out_root, split, scene_id, sem_pp, inst_masks)

        # 记录 split 列表（无论空/非空样本，只要保存成功就登记）
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