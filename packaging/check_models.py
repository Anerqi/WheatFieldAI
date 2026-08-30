# packaging/check_models.py
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "packaging" / "model_assets.yaml"

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from config import load_config  # noqa: E402
from pest_models import (  # noqa: E402
    load_pest_class_names,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-sha256",
        action="store_true",
    )
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(
            f"[FAIL] 资产清单不存在: "
            f"{MANIFEST}"
        )
        return 1

    manifest = yaml.safe_load(
        MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    cfg = load_config(
        ROOT / "config.yaml"
    )

    asset_paths = {
        "weed_yolo11s":
            cfg["weed"]["paths"]["yolo11_weights"],
        "weed_yolox_small":
            cfg["weed"]["paths"][
                "yolox_small_weights"
            ],
        "weed_yolox_base":
            cfg["weed"]["paths"][
                "yolox_base_weights"
            ],
        "pest_yolo11m":
            cfg["pest"]["paths"][
                "yolo11m_weights"
            ],
        "pest_yolo11l":
            cfg["pest"]["paths"][
                "yolo11l_weights"
            ],
        "pest_yolo11s":
            cfg["pest"]["paths"][
                "yolo11s_weights"
            ],
        "dataset_yaml":
            cfg["pest"]["dataset_yaml"],
        "classwise_wbf":
            cfg["pest"]["classwise_config"],
    }

    missing = []
    invalid = []
    hashed = []

    print("=" * 64)
    print("WheatFieldAI model/config asset check")
    print("=" * 64)

    for asset in manifest.get("assets", []):
        asset_id = asset["id"]

        if asset_id not in asset_paths:
            invalid.append(
                f"unknown asset id: {asset_id}"
            )
            print(
                "[FAIL] 未知资产 id："
                f"{asset_id}"
            )
            continue

        path = Path(
            asset_paths[asset_id]
        )

        if not path.exists():
            missing.append(str(path))
            print(
                f"[MISS] {asset_id}: {path}"
            )
            continue

        if path.stat().st_size == 0:
            invalid.append(str(path))
            print(
                f"[FAIL] {asset_id}: 空文件"
            )
            continue

        digest = sha256(path)
        prefix = str(
            asset.get(
                "sha256_prefix",
                "",
            )
        ).strip()

        if prefix and not digest.startswith(
            prefix
        ):
            invalid.append(str(path))
            print(
                f"[FAIL] {asset_id}: "
                "SHA-256 前缀不匹配，"
                f"实际 {digest[:12]}…"
            )
        else:
            print(
                f"[ OK ] {asset_id}: "
                f"{path}"
            )

        hashed.append(
            (asset_id, digest)
        )

    dataset_path = Path(
        asset_paths["dataset_yaml"]
    )

    if dataset_path.exists():
        try:
            names = load_pest_class_names(
                str(dataset_path)
            )

            if len(names) != 32:
                raise ValueError(
                    "names 数量不是 32"
                )

            print(
                "[ OK ] dataset.yaml: "
                f"32 类加载成功"
                f"（{names[0]} … {names[-1]}）"
            )

        except Exception as exc:
            invalid.append(
                str(dataset_path)
            )
            print(
                f"[FAIL] dataset.yaml: {exc}"
            )
    else:
        missing.append(
            str(dataset_path)
        )
        print(
            f"[MISS] dataset.yaml: "
            f"{dataset_path}"
        )

    classwise_path = Path(
        asset_paths["classwise_wbf"]
    )

    if classwise_path.exists():
        try:
            data = json.loads(
                classwise_path.read_text(
                    encoding="utf-8"
                )
            )

            configs = data.get(
                "class_configs"
            )

            if not isinstance(configs, dict):
                raise ValueError(
                    "缺少 class_configs"
                )

            ids = sorted(
                int(key)
                for key in configs
            )

            if ids != list(range(32)):
                raise ValueError(
                    "class_configs 必须完整覆盖 0..31"
                )

            required = {
                "model_weights",
                "wbf_iou",
                "skip",
                "score_mode",
            }

            for class_id in range(32):
                entry = configs[
                    str(class_id)
                ]

                if not required.issubset(
                    entry
                ):
                    raise ValueError(
                        f"class {class_id} "
                        "缺少融合字段"
                    )

                if len(
                    entry["model_weights"]
                ) != 3:
                    raise ValueError(
                        f"class {class_id} "
                        "model_weights 必须有 3 个值"
                    )

                if entry[
                    "score_mode"
                ] not in {
                    "agreement",
                    "sqrt_agreement",
                    "mean",
                    "max",
                }:
                    raise ValueError(
                        f"class {class_id} "
                        "score_mode 非法"
                    )

            print(
                "[ OK ] classwise JSON: "
                "32 类配置完整且字段合法"
            )

        except Exception as exc:
            invalid.append(
                str(classwise_path)
            )
            print(
                f"[FAIL] classwise JSON: {exc}"
            )
    else:
        missing.append(
            str(classwise_path)
        )
        print(
            f"[MISS] classwise JSON: "
            f"{classwise_path}"
        )

    if args.print_sha256:
        print("\nSHA-256:")
        for asset_id, digest in hashed:
            print(
                f"{asset_id}: {digest}"
            )

    print("-" * 64)
    print(
        f"missing={len(missing)}, "
        f"invalid={len(invalid)}"
    )

    if invalid:
        print(
            "[RESULT] 存在结构/校验错误，"
            "禁止启动。"
        )
        return 1

    if missing:
        print(
            "[RESULT] 资产不完整，但允许启动；"
            "Web 页面将明确显示「模型未配置」。"
        )
        return 2

    print(
        "[RESULT] 所有模型与配置资产检查通过。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
