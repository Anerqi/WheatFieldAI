# -*- coding: utf-8 -*-
"""公开仓库卫生检查（免权重、免网络）。

防止把本机绝对路径、真实用户名目录或凭据写回公开仓库。
测试夹具中的人名目录路径是故意保留的脱敏样例，不在此列。
"""

import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv", "node_modules"}
SKIP_EXT = {".otf", ".woff2", ".png", ".jpg", ".jpeg", ".onnx", ".pth", ".pt", ".zip", ".pdf", ".ico", ".gif"}

PATTERNS = [
    (re.compile(r"[A-Za-z]:[\\/]Users[\\/](?!(?:someone|example|user|username)\b)", re.IGNORECASE),
     "本机 Windows 用户目录绝对路径"),
    (re.compile(r"[A-Za-z]:[\\/](?:Desktop|Documents and Settings|AppData)[\\/]", re.IGNORECASE),
     "本机 Windows 桌面/配置目录绝对路径"),
    (re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"), "私钥"),
    (re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
     "疑似凭据赋值"),
]


def iter_text_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if os.path.splitext(name)[1].lower() in SKIP_EXT:
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    yield os.path.relpath(path, ROOT).replace("\\", "/"), fh.read()
            except OSError:
                continue


class PublicHygieneTest(unittest.TestCase):
    def test_no_machine_specific_paths_or_credentials(self):
        hits = []
        for rel, text in iter_text_files():
            for pattern, label in PATTERNS:
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    hits.append("%s:%d: %s: %r" % (rel, line, label, match.group(0)[:60]))
        self.assertEqual(hits, [], "公开仓库不得包含本机路径/凭据：\n" + "\n".join(hits[:20]))


if __name__ == "__main__":
    unittest.main()
