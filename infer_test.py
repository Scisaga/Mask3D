#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from hydra.experimental import initialize, compose
from infer import Mask3DInferencer


def main():
    # 确保工作目录能找到 conf/
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    # hydra-core 1.0.x: 用 initialize(config_path="conf")，config_name 不带 .yaml
    with initialize(config_path="conf", job_name="infer"):
        cfg = compose(
            config_name="config_base_instance_segmentation",
            overrides=[
                # ❌ 删掉 +model / +trainer（默认已经在 defaults 里选了）
                # "model=mask3d",          # 只有当你想改成别的变体时才写覆盖
                # "trainer=trainer600",
                "general.train_mode=false",
                "general.checkpoint=saved/real3dad_exp1/last-epoch.ckpt",
                "+infer.input_pcd=/data/Real3D-AD-PCD/toffees/test_neo/606_bulge.pcd",
                "+infer.output_pcd=out/demo_pred.pcd",
            ],
        )

    engine = Mask3DInferencer(cfg)

    with open(cfg.infer.input_pcd, "r", encoding="utf-8") as f:
        in_pcd_str = f.read()

    out_pcd_str = engine.predict_pcd_string(in_pcd_str)

    os.makedirs(os.path.dirname(cfg.infer.output_pcd) or ".", exist_ok=True)
    with open(cfg.infer.output_pcd, "w", encoding="utf-8") as f:
        f.write(out_pcd_str)

    print(f"[OK] 已输出: {cfg.infer.output_pcd}")

if __name__ == "__main__":
    main()
