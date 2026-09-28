"""llm_eval —— 面向大语言模型（LLM）输出的轻量评测框架。

本包提供从「评测数据集」到「指标计算」「评估」「LLM 裁判」「报告」的完整链路，
并且把评测用例组织成 pytest 测试，方便集成到 CI / 学习。

模块划分：
    - dataset   : 评测数据集的模型定义与加载
    - metrics   : 各种评测指标的纯 Python 实现
    - evaluator : 评估器，把模型输出与参考答案对比打分
    - judge     : LLM-as-judge（用大模型当裁判），含离线 Mock
    - report    : 生成 JSON / HTML 评测报告
"""

__version__ = "0.1.0"

# 对外暴露最常用的几个符号，方便 `from llm_eval import Evaluator` 这样导入
from .dataset import EvalSample, TaskType, load_dataset
from .evaluator import Evaluator, EvaluationResult
from .metrics import (
    exact_match,
    accuracy,
    rouge_l,
    bleu,
    token_f1,
    cosine_similarity,
    label_precision_recall_f1,
)

__all__ = [
    "EvalSample",
    "TaskType",
    "load_dataset",
    "Evaluator",
    "EvaluationResult",
    "exact_match",
    "accuracy",
    "rouge_l",
    "bleu",
    "token_f1",
    "cosine_similarity",
    "label_precision_recall_f1",
]
