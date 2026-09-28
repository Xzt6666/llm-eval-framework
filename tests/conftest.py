"""pytest 全局配置与共享 fixture。

fixture 的作用：把「加载数据集」「构建评估器」这些重复的准备工作抽出来，
让每个测试函数都能直接复用，保持测试代码干净。

关于 import 路径：
    pyproject.toml 里已配置 pythonpath = ["src"]，pytest 运行时能直接
    `import llm_eval`。这里再显式插入一次 src 路径，是为了保证用
    `python tests/xxx.py` 或其它方式单独执行时也不会报 ImportError。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 项目根目录（tests/ 的上一级）
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_eval.dataset import load_dataset  # noqa: E402
from llm_eval.evaluator import Evaluator  # noqa: E402


@pytest.fixture(scope="session")
def samples():
    """加载样例评测集（整个测试会话只加载一次）。"""
    return load_dataset(ROOT / "data" / "samples.json")


@pytest.fixture(scope="session")
def evaluator():
    """提供一个默认阈值 0.5 的评估器实例。"""
    return Evaluator(threshold=0.5)
