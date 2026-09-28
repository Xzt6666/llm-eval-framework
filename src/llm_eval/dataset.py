"""评测数据集模块。

在 LLM 评测中，我们首先需要一个「评测集」：一组带参考答案的样本。
每条样本通常包含：
    - 输入提示词（prompt）：喂给大模型的问题 / 指令
    - 参考答案（reference）：期望的正确输出（黄金标准）
    - 模型实际输出（model_output）：待评测模型给出的回答
    - 任务类型（task_type）：决定用哪套指标来打分

本模块定义这些数据结构，并提供从 JSON 文件加载评测集的能力。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class TaskType(str, Enum):
    """任务类型枚举。

    不同任务关注的能力不同，因此评测指标也不同：
        - CLASSIFICATION : 分类（如情感分析、主题分类）→ 看准确率 / F1 / 精确匹配
        - GENERATION     : 生成（如摘要、翻译、问答）→ 看 ROUGE / BLEU / token F1
    """

    CLASSIFICATION = "classification"
    GENERATION = "generation"


@dataclass
class EvalSample:
    """单条评测样本。

    属性说明：
        id            : 样本唯一标识
        prompt        : 输入提示词（喂给模型的内容）
        reference     : 参考答案（黄金标准，用来对比）
        model_output  : 待评测模型的输出（可为空，评测前再填入）
        task_type     : 任务类型，决定用哪套指标
        metadata      : 可选扩展信息（如难度、标签、来源等）
    """

    id: str
    prompt: str
    reference: str
    model_output: str = ""
    task_type: TaskType = TaskType.GENERATION
    metadata: Dict[str, Any] = field(default_factory=dict)


def _coerce_task_type(raw: Any) -> TaskType:
    """把 JSON 里的字符串安全地转换成 TaskType 枚举。

    若传入了不认识的类型，默认按生成任务处理，避免整个加载过程报错。
    """
    if isinstance(raw, TaskType):
        return raw
    text = str(raw).lower().strip()
    if text in ("classification", "classify", "cls"):
        return TaskType.CLASSIFICATION
    return TaskType.GENERATION


def load_dataset(path: str | Path) -> List[EvalSample]:
    """从 JSON 文件加载评测集。

    期望的 JSON 结构（最外层是一个列表）：
        [
            {
                "id": "sample-001",
                "prompt": "今天天气真好，这句话的情感是？",
                "reference": "正面",
                "model_output": "正面",
                "task_type": "classification",
                "metadata": {"difficulty": "easy"}
            },
            ...
        ]

    参数：
        path : JSON 文件路径

    返回：
        解析出的 EvalSample 列表
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # 兼容两种结构：直接是列表，或者是 {"samples": [...]} 包裹的对象
    if isinstance(raw, dict):
        raw = raw.get("samples", [])

    samples: List[EvalSample] = []
    for item in raw:
        samples.append(
            EvalSample(
                id=str(item["id"]),
                prompt=item.get("prompt", ""),
                reference=item.get("reference", ""),
                model_output=item.get("model_output", ""),
                task_type=_coerce_task_type(item.get("task_type")),
                metadata=item.get("metadata", {}) or {},
            )
        )
    return samples


def save_dataset(samples: List[EvalSample], path: str | Path) -> None:
    """把评测集保存为 JSON 文件（方便把评测结果回写 / 分享）。

    参数：
        samples : 要保存的样本列表
        path    : 目标 JSON 路径
    """
    path = Path(path)
    data = [
        {
            "id": s.id,
            "prompt": s.prompt,
            "reference": s.reference,
            "model_output": s.model_output,
            "task_type": s.task_type.value,
            "metadata": s.metadata,
        }
        for s in samples
    ]
    with path.open("w", encoding="utf-8") as f:
        # ensure_ascii=False 让中文原样输出，indent=2 便于阅读
        json.dump(data, f, ensure_ascii=False, indent=2)
