import os
import threading
from flask import Flask, request, jsonify, render_template
from hydra.experimental import initialize, compose
from infer import Mask3DInferencer  # 需保证同目录且类名一致

_ENGINE = None
_ENGINE_LOCK = threading.Lock()

def _load_cfg():
    """
    适配 hydra-core==1.0.5：使用 hydra.experimental.initialize/compose
    与原仓库一致：conf/config_base_instance_segmentation.yaml
    """
    ckpt = os.environ.get("MASK3D_CKPT", "saved/real3dad_exp1/last-epoch.ckpt")
    gpus = os.environ.get("MASK3D_GPUS", "1")
    with initialize(config_path="conf", job_name="infer"):
        cfg = compose(
            config_name="config_base_instance_segmentation.yaml",
            overrides=[
                "general.train_mode=false",
                f"general.checkpoint={ckpt}",
                f"general.gpus={gpus}",
                # 确保走测试逻辑
                "data.test_mode=test",
            ],
        )
    return cfg

def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        cfg = _load_cfg()
        _ENGINE = Mask3DInferencer(cfg)
    return _ENGINE


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/infer")
    def infer_endpoint():
        try:
            payload = request.get_json(force=True, silent=False)
            pcd_text = payload.get("pcd", "")
            if not isinstance(pcd_text, str) or len(pcd_text) < 10:
                return jsonify({"error": "pcd（ASCII PCD 字符串）是必填项"}), 400

            engine = _get_engine()
            with _ENGINE_LOCK:  # 简单串行，避免多请求争抢同一块 GPU
                out_pcd = engine.predict_pcd_string(pcd_text)

            return jsonify({"pcd": out_pcd})
        except Exception as e:
            e.print(f"推理失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.get("/healthz")
    def healthz():
        try:
            _ = _get_engine()
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    return app

app = create_app()

if __name__ == "__main__":
    # 建议：export CUDA_VISIBLE_DEVICES=0 选择显卡
    # 可选：export MASK3D_CKPT=/path/to/xxx.ckpt
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True, threaded=False)
