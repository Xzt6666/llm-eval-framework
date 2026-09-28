"""评估器模块 —— 把「模型输出」与「参考答案」对比打分。

评估器（Evaluator）是整个评测框架的调度中心：
    1. 根据样本的 task_type 选择对应的指标集合
    2. 计算每个样本的各项分数
    3. 汇总所有样本，产出平均值、通过率等统计信息
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .dataset import EvalSample, TaskType
from . import metrics


@dataclass
class EvaluationResult:
    """单条样本的评测结果。

    属性说明：
        sample_id : 对应样本的 id
        task_type : 任务类型
        scores    : 各指标得分，例如 {"f1": 0.8, "rouge_l": 0.7, ...}
        passed    : 是否通过（主指标 >= 阈值）
        details   : 附加信息（如参考答案、模型输出原文，便于排查）
    """

    sample_id: str
    task_type: str
    scores: Dict[str, float] = field(default_factory=dict)
    passed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """评估器。

    使用方式：
        evaluator = Evaluator()
        results = evaluator.evaluate_batch(samples)

    阈值（threshold）与主指标（primary_metric）用于判定「是否通过」：
        - 分类任务主指标为 exact_match / accuracy
        - 生成任务主指标为 rouge_l 的 f1
    """

    # 不同任务类型对应的主指标，用于判定是否通过
    _PRIMARY_METRIC = {
        TaskType.CLASSIFICATION: "exact_match",
        TaskType.GENERATION: "rouge_l_f1",
    }

    def __init__(self, threshold: float = 0.5) -> None:
        """初始化评估器。

        参数：
            threshold : 通过阈值（0~1）。主指标 >= threshold 判定为通过。
        """
        self.threshold = threshold

    # ------------------------------------------------------------------
    # 核心：对单条样本打分
    # ------------------------------------------------------------------
    def evaluate(self, sample: EvalSample) -> EvaluationResult:
        """对单条样本打分，返回 EvaluationResult。

        内部根据 task_type 分发到对应指标函数。
        """
        pred = sample.model_output
        ref = sample.reference

        if sample.task_type == TaskType.CLASSIFICATION:
            scores = self._evaluate_classification(pred, ref)
        else:
            scores = self._evaluate_generation(pred, ref)

        # 主指标：用来判定这条样本是否「通过」
        primary_key = self._PRIMARY_METRIC[sample.task_type]
        primary_score = scores.get(primary_key, 0.0)

        return EvaluationResult(
            sample_id=sample.id,
            task_type=sample.task_type.value,
            scores=scores,
            passed=primary_score >= self.threshold,
            details={
                "prompt": sample.prompt,
                "reference": ref,
                "model_output": pred,
                "primary_metric": primary_key,
            },
        )

    def _evaluate_classification(self, pred: str, ref: str) -> Dict[str, float]:
        """分类任务：精确匹配 + 词级 F1（作为软指标）。"""
        em = 1.0 if metrics.exact_match(pred, ref) else 0.0
        f1 = metrics.token_f1(pred, ref)["f1"]
        return {"exact_match": em, "token_f1": f1}

    def _evaluate_generation(self, pred: str, ref: str) -> Dict[str, float]:
        """生成任务：ROUGE-L + BLEU + token F1 + 余弦相似度。"""
        rouge = metrics.rouge_l(pred, ref)
        bleu_score = metrics.bleu(pred, ref)
        f1 = metrics.token_f1(pred, ref)["f1"]
        cos = metrics.cosine_similarity(pred, ref)
        return {
            "rouge_l_f1": rouge["f1"],
            "rouge_l_recall": rouge["recall"],
            "bleu": bleu_score,
            "token_f1": f1,
            "cosine_similarity": cos,
        }

    # ------------------------------------------------------------------
    # 批量评测 与 汇总
    # ------------------------------------------------------------------
    def evaluate_batch(self, samples: List[EvalSample]) -> List[EvaluationResult]:
        """对一批样本打分，返回结果列表。"""
        return [self.evaluate(s) for s in samples]

    def aggregate(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """汇总评测结果，产出统计信息。

        返回：
            {
                "total": 总样本数,
                "passed": 通过数,
                "pass_rate": 通过率,
                "avg_scores": {指标名: 平均分},
            }
        """
        if not results:
            return {"total": 0, "passed": 0, "pass_rate": 0.0, "avg_scores": {}}

        passed = sum(1 for r in results if r.passed)
        pass_rate = passed / len(results)

        # 统计所有样本上出现过的指标，求平均值
        score_names: set[str] = set()
        for r in results:
            score_names.update(r.scores.keys())

        avg_scores: Dict[str, float] = {}
        for name in score_names:
            values = [r.scores[name] for r in results if name in r.scores]
            avg_scores[name] = sum(values) / len(values) if values else 0.0

        return {
            "total": len(results),
            "passed": passed,
            "pass_rate": pass_rate,
            "avg_scores": avg_scores,
        }
