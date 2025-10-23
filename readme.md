# 更新概览（自 da414a61c8b195d045e502148af240dae25916a6 起）

## 综述

自基线提交 `da414a61c8b195d045e502148af240dae25916a6` 以来，仓库经历了体系化的功能扩展与环境重构，重点包括：

- 新增并完整适配 Real3D‑AD 数据集，覆盖数据转换、数据集类、Hydra 配置与评估规程。
- 训练与评估链路的健壮性修复，处理空样本、单点跨注意力异常、Top‑K 边界与度量聚合容错。
- 容器与依赖体系重写，确保在镜像内部创建 Conda 环境并自动编译第三方库。
- 推理与可视化全新搭建，包含复用训练配置的推理引擎、命令行示例以及 Flask + Three.js 的 Web 服务。
- 默认配置调优，围绕“两类缺陷 + 小样本”场景重新设定超参、损失权重与日志策略。

共修改 31 个文件（含新增 Notebook 与静态资源），合并约 7.56M 新增行（大部分来自 Notebook）。

---

## 提交历史

按时间顺序列出的关键提交（省略重复“update”类提交）：

- `718d6d4` Refactor Dockerfile to streamline installation process and update environment setup  
- `94a08d8` Fix Dockerfile to remove NVIDIA developer source and streamline package installation  
- `3a7d286` update code  
- `2ef214e` Update Dockerfile to ensure pip installation is executed after conda environment creation  
- `bafe7e7` Update pip version in environment.yml to 24.0  
- `dec1a31` Update Dockerfile and environment.yml to include omegaconf dependency and ensure pip installation is performed correctly  
- `f3667fd` Update environment.yml to correct pyyaml version and add conda-forge::pycocotools dependency  
- `f650660` / `47dcf24` Update Dockerfile to use login shell for conda activation and fix activation command  
- `c522fab` Update Dockerfile and docker-compose.yml for improved environment setup and port configuration  
- `50174c1` 容器化 + 适配数据集 + 测试训练  
- `77cba65` 解决体素中只有一个点时的逻辑出错  
- `e7cb7bd` / `df4c37c` 增加推理服务与 Web 推理功能  
- `7bcd25a` / `0a77b1f` update code（整合前述改动）

---

## 环境与容器

- **Docker 重构**：镜像改用 `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04`，在容器内创建 `mask3d_cuda113` 环境，并通过 `conda run` 安装 PyTorch、torch-scatter、Detectron2、Lightning；补充 OpenBLAS、Cython 等构建依赖，同时自动编译 MinkowskiEngine、ScanNet Segmentator 与 pointnet2。（`Dockerfile`）
- **登录体验**：写入 `/etc/profile.d/conda_auto.sh` 以及用户 `.bashrc`，确保默认登录即激活环境。
- **docker-compose 调整**：端口改为 `52001:22` / `50001:5000`，数据卷挂载到 `/opt/sky-fabric/data`，使用 `runtime: nvidia` 并传递 GPU 相关环境变量。（`docker-compose.yml`）
- **Conda 依赖**：新增 `conda-forge` 渠道，将 `pyyaml`、`pycocotools` 等迁移到 Conda 依赖，加入 `pyntcloud` 支撑点云转换。（`environment.yml`）

---

## 数据集与预处理

- **Real3D‑AD 转换脚本**：`datasets/real3dad_convert.py` 读取 PCD（x y z + 语义），使用 DBSCAN 自动生成实例掩码，可配置是否保留空实例样本，并输出 `.pt` 样本、GT 文本、`splits.json` 与 `category2id.json`。
- **数据集类**：`datasets/real3dad.py` 实现 `Real3DADDataset`，构建 `label_info` 映射，按需拼接 raw xyz / 颜色 / 法向，默认输出 `(坐标, 特征, 语义+实例+segment, 场景名, raw 信息)`。
- **Hydra 配置**：`conf/data/datasets/real3dad.yaml` 为 train/val/test 配置统一的数据根、忽略标签、GT 路径与实例标记开关。
- **体素化工具增强**：`datasets/utils.py` 新增 `ensure_2d` 与 `debug_sparse_collate_inputs`，在稀疏量化前验证坐标/特征形状；保留原始坐标供后续还原使用。

---

## 配置与模型参数

- **实验基础配置**（`conf/config_base_instance_segmentation.yaml`）：
  - 实验名 `real3dad_exp1`，`num_targets: 3`
  - `scores_threshold: 0.4`，`iou_threshold: 0.5`
  - `topk_per_image: 6`，`dbscan_eps: 0.5`，`dbscan_min_points: 10`
  - 默认使用 minimal Logger，`gpus: [1]`
- **数据配置**（`conf/data/indoor.yaml`）：
  - 划分命名 `validation_mode: val`、`test_mode: test`
  - 关闭颜色特征，`num_labels: 3`，`batch_size: 10`
- **模型采样**（`conf/model/mask3d.yaml`）：
  - `num_queries: 6`，点采样阶梯减小至 `[32, 128, 512, 2048, 4096]`
- **损失与匹配**：
  - `conf/loss/set_criterion.yaml`：`eos_coef: 0.4`，`class_weights: [1.0, 1.6]`
  - `models/criterion.py`：`ignore_index` 改为 255
  - `models/matcher.py`：将 255 视为忽略标签
- **训练计划**：`conf/trainer/trainer600.yaml` 将验证频率从 50 epoch 缩短为 5 epoch。
- **日志**：`conf/logging/minimal.yaml` 使用实验名作为 CSVLogger 名称。

---

## 训练与评估流程

- **原始坐标重建**：`trainer/trainer.py` 在训练与评估中改为通过体素坐标 × `voxel_size` 还原 raw xyz，并断言 `features` 仍保留 >=1 个通道。
- **空样本处理**：遇到空坐标或单点导致的 cross-attention 异常时返回空字典，避免后续聚合报错。
- **Top-K 稳健性**：新的 `get_topk_proposals` 分支支持 `topk_per_image == -1`、无候选列等边界。
- **度量聚合容错**：`validation_epoch_end` / `test_epoch_end` 通过 `_safe_mean` 过滤非数值条目，防止空列表求均值抛异常。
- **GT 路径解析**：自动识别 `data_dir` 类型（字符串、Path、list），并对非 ScanNet/S3DIS 场景默认定位到 `instance_gt/<split>`。

---

## 推理与服务

- **推理引擎**：`infer.py` 引入 `Mask3DInferencer`，复用训练配置、collate 与权重加载，提供
  - `predict_xyz(xyz: np.ndarray)`：返回逐点语义/实例标签
  - `predict_pcd_string(text: str)`：ASCII PCD 字符串输入输出
- **命令行示例**：`infer_test.py` 读取 PCD，调用引擎并写出 `out/demo_pred.pcd`。
- **Flask 服务**：`server.py` 暴露 `/infer` JSON API 与 `/` 前端页面，内部使用全局锁串行化 GPU 推理。需注意异常处理处的 `e.print(...)` 应改为 `print(e)` 或 logging。
- **前端可视化**：`static/app.js` + `static/style.css` + `templates/index.html` 实现 Three.js 点云渲染、实例透明度控制、统计信息展示与结果下载；新增 LOGO 资源。

---

## 其他新增内容

- Notebook：`data_validation.ipynb`、`data_sampling.ipynb`、`show_metrics.ipynb` 用于数据检查与指标展示（文件体积较大）。
- 示例输出：`out/demo_pred.pcd` 保存 Web/CLI 推理示例结果。

---

## 潜在影响与建议

1. **特征切片逻辑已更改**：若外部脚本仍通过 `data.features[:, -3:]` 取得 raw xyz，需要同步修改，以免出现空通道断言失败。
2. **配置差异**：新默认针对 Real3D‑AD 小样本/少实例场景，与 ScanNet 等原始配置存在显著差异（`num_targets`、`num_queries` 等），混用时需检查一致性。
3. **Web 服务异常处理**：建议将 `server.py:60` 的 `e.print(...)` 修复为标准输出或日志调用。
4. **Notebook 体积大**：如需保持 Git 仓库精简，可考虑改用 Git LFS 或排除这些文件。

---

## 关键文件索引

- 环境：`Dockerfile`、`docker-compose.yml`、`environment.yml`
- 数据集：`datasets/real3dad_convert.py`、`datasets/real3dad.py`、`conf/data/datasets/real3dad.yaml`
- 配置：`conf/config_base_instance_segmentation.yaml`、`conf/data/indoor.yaml`、`conf/model/mask3d.yaml`、`conf/loss/set_criterion.yaml`、`conf/trainer/trainer600.yaml`
- 训练/评估：`trainer/trainer.py`、`benchmark/evaluate_semantic_instance.py`、`models/criterion.py`、`models/matcher.py`
- 推理/服务：`infer.py`、`infer_test.py`、`server.py`、`static/app.js`、`templates/index.html`
- 其他：`data_validation.ipynb`、`data_sampling.ipynb`、`show_metrics.ipynb`、`out/demo_pred.pcd`

## 主要命令

### 训练命令
```shell
python main_instance_segmentation.py

python main_instance_segmentation.py conf.data.datasets=real3dad general.experiment_name=test1
```
- 任何调用 python main_instance_segmentation.py 且未显式设置 general.train_mode=false 的变体（哪怕加了其它 Hydra override，比如 trainer.num_sanity_val_steps=0 等），都会在默认配置下进入训练模式。
- 命令中如果只有 general.checkpoint= 或 general.checkpoint_path= 但没有把 general.train_mode 设为 false，脚本仍会按训练逻辑启动，并尝试从指定 checkpoint 恢复训练。

### 推理命令

```shell
python main_instance_segmentation.py general.checkpoint=saved/real3dad_exp1/last-epoch.ckpt general.train_mode=false
python main_instance_segmentation.py general.checkpoint=saved/real3dad_exp1/last-epoch.ckpt general.train_mode=false data.test_mode=train data.validation_mode=train
python main_instance_segmentation.py general.checkpoint=saved/real3dad_exp1/last-epoch.ckpt general.train_mode=false data.test_mode=train data.validation_mode=train data.test_collation.mode=val
```
- general.checkpoint：指定要载入的 Lightning checkpoint。
- general.train_mode=false，控制主程序走训练还是评估分支。
   - 默认为 true，会调用 train(cfg)，即 Lightning 的 Trainer.fit。
   - 设置为 false 时执行 test(cfg)，调用 Trainer.test，从而以评估/推理模式运行，使用的是配置里的验证/测试数据加载器。
- data.test_mode= | data.validation_mode= 来源：conf/data/indoor.yaml:7-9
   - `data.test_mode=train / data.validation_mode=train` 在评估阶段强制把验证、测试数据集都指向训练集目录。这通常用于在训练集上跑推理、调试指标或导出结果。
   - `data.test_mode=val / data.validation_mode=val` 让验证和测试都基于 val/ 划分，常见于将整套评估集中到同一 split。
- data.test_collation.mode=val VoxelizeCollate 的实例配置，来源 conf/data/collation_functions/voxelize_collate.yaml:39
   - 如果设成 train，函数可能触发训练专用逻辑（如二次裁剪、小块拼接等），并只返回低分辨率标签；
   - 设成 val 或 test，则会保持评估路径，返回更多字段（例如 target_full、inverse_maps），以支持最终 AP 计算。

### 推理服务启动
```shell
python server.py
```

### 其他

#### 数据集格式转换
```shell
python datasets/real3dad_convert.py
```

#### ubuntu设定全局代理

在终端或 .bashrc / .zshrc 中设定
```shell
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export all_proxy="socks5://127.0.0.1:7890"

# 不经过代理的地址（注意用逗号分隔，不要空格）
export no_proxy="localhost,127.0.0.1,::1,*.local,192.168.0.0/16,10.0.0.0/8,172.16.0.0/12"
```
刷新
```shell
source ~/.bashrc
```

### Git命令

#### 设定仓库层级代理
```shell
git config --global http.https://github.com.proxy  http://127.0.0.1:7890
# 取消时：git config --global --unset http.https://github.com.proxy
```

#### 清理凭据 / 设定凭据

清理凭据

```shell
# 看看启用了哪些凭据助手（便于确认）
git config --show-origin --get-all credential.helper || true

# 全局/系统级取消 credential.helper（不存在会忽略）
git config --global --unset-all credential.helper || true
sudo git config --system --unset-all credential.helper 2>/dev/null || true

# 删除明文凭据文件（如果以前用过 store）
rm -f ~/.git-credentials

# 停掉 git-credential-cache（如果启用过）
git credential-cache exit 2>/dev/null || true

# 让 Git 立刻忘掉 github.com 的已缓存条目
printf "protocol=https\nhost=github.com\n" | git credential reject
```

设定凭据push

```shell
# 设定凭据助手为 store（读取 ~/.git-credentials）
git config --global credential.helper store

# 把你的用户名 & PAT 写入凭据文件（把占位替换为真实值）
git credential-store --file ~/.git-credentials store <<'EOF'
protocol=https
host=github.com
username=Scisaga
password=<YOUR_GITHUB_PAT>
EOF

# 确认远端用 HTTPS
git remote set-url origin https://github.com/Scisaga/Mask3D.git

# 推送
git push origin main
```

#### 回退特定版本 / 撤销
```shell
git reset --hard <commit>  # 本地回拨
git reset --hard ORIG_HEAD
```