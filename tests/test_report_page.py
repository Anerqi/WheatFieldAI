# -*- coding: utf-8 -*-
"""汇报页移除后的导航回归测试。"""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / "pages"
REPORT_PAGE = PAGES_DIR / "2_汇报.py"


class ReportPageRemovalTests(unittest.TestCase):
    def test_report_page_entrypoint_is_absent(self):
        """Streamlit 不应再发现独立汇报页入口。"""
        self.assertFalse(REPORT_PAGE.exists())

    def test_no_python_page_entrypoints_remain(self):
        """pages/ 不应意外保留其它活动页面。"""
        self.assertEqual(list(PAGES_DIR.glob("*.py")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
