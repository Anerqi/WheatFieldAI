# packaging/check_env.py
from __future__ import annotations

import importlib.metadata as md
import re
import sys


REQUIRED = {
    "streamlit": "1.59",
    "ultralytics": "8.4.81",
    "opencv-python": "4.10",
    "numpy": "1.26",
    "PyYAML": "6.0.3",
    "Pillow": "10.0",
    "pandas": "2.0",
}


def dist_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def version_tuple(version: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", version)
    return tuple(int(value) for value in nums[:4]) or (0,)


def check_minimum(
    actual: str,
    minimum: str,
) -> bool:
    return version_tuple(actual) >= version_tuple(minimum)


def main() -> int:
    errors = 0
    warnings = 0

    print("=" * 64)
    print("WheatFieldAI environment check")
    print("=" * 64)

    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info[:2] == (3, 13):
        print("[ OK ] Python 3.13.x")
    else:
        warnings += 1
        print("[WARN] 推荐 Python 3.13.x。")

    for package, minimum in REQUIRED.items():
        actual = dist_version(package)

        if actual is None:
            errors += 1
            print(f"[FAIL] {package}: 未安装")
            continue

        if check_minimum(actual, minimum):
            print(f"[ OK ] {package}: {actual}")
        else:
            errors += 1
            print(
                f"[FAIL] {package}: "
                f"{actual} < {minimum}"
            )

    torch_version = dist_version("torch")
    torchvision_version = dist_version("torchvision")

    if torch_version is None:
        errors += 1
        print("[FAIL] torch: 未安装")
    else:
        print(f"[ OK ] torch: {torch_version}")
        if torch_version != "2.11.0":
            warnings += 1
            print(
                "[WARN] 已验证基线为 "
                "torch 2.11.0+cu128。"
            )

    if torchvision_version is None:
        errors += 1
        print("[FAIL] torchvision: 未安装")
    else:
        print(
            f"[ OK ] torchvision: "
            f"{torchvision_version}"
        )
        if torchvision_version != "0.26.0":
            warnings += 1
            print(
                "[WARN] 已验证基线为 "
                "torchvision 0.26.0+cu128。"
            )

    try:
        import torch

        print(
            f"PyTorch CUDA build: "
            f"{torch.version.cuda}"
        )

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = (
                torch.cuda.get_device_properties(0)
                .total_memory
                / 1024**3
            )

            print("[ OK ] CUDA available: yes")
            print(f"[ OK ] GPU: {device_name}")
            print(
                f"[ OK ] VRAM: {vram:.2f} GB"
            )
        else:
            warnings += 1
            print(
                "[WARN] CUDA 不可用；"
                "应用可以启动，但建议使用快速单模型。"
            )

    except Exception as exc:
        errors += 1
        print(
            f"[FAIL] CUDA 检查异常: "
            f"{type(exc).__name__}: {exc}"
        )

    print("-" * 64)
    print(
        f"errors={errors}, "
        f"warnings={warnings}"
    )

    if errors:
        print(
            "[RESULT] 环境检查失败，"
            "启动器不会继续启动。"
        )
        return 1

    print("[RESULT] 环境检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
