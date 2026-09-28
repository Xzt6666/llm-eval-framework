"""端到端评测测试 —— 把「评测用例」组织成 pytest 参数化测试。

这是本项目的核心设计：每一条评测样本，都对应一条 pytest 测试用例。
这样做的价值：
    1. 评测结果直接以 pytest 的 pass / fail 呈现，直观且可接入 CI
    2. 借助 pytest 的参数化、fixture、报告插件，评测过程高度可复用
    3. 哪条样本没过、为什么没过，一眼就能定位

下面演示两种用法：
    - test_each_sample_score_in_range : 逐条校验分数合法（0~1）
    - test_pass_rate_meets_bar        : 整体通过率是否达标（类似 CI 门禁）
"""

from __future__ import annotations

import pytest

from llm_eval.evaluator import EvaluationResult


class TestEndToEndEvaluation:
    """端到端：加载真实评测集并逐条评估。"""

    def test_each_sample_score_in_range(self, samples, evaluator):
        """逐条评估：所有指标分数都应落在 0~1 之间。"""
        for sample in samples:
            result = evaluator.evaluate(sample)
            assert isinstance(result, EvaluationResult)
            for name, score in result.scores.items():
                assert 0.0 <= score <= 1.0, (
                    f"样本 {sample.id} 的指标 {name} 得分 {score} 超出 [0,1]"
                )

    def test_pass_rate_meets_bar(self, samples, evaluator):
        """整体通过率门禁：评测集整体通过率应不低于 0.25。

        （实际工程里，这就是「模型发版前必须过评测门禁」的雏形。）
        """
        results = evaluator.evaluate_batch(samples)
        agg = evaluator.aggregate(results)

        assert agg["total"] == len(samples)
        assert 0.0 <= agg["pass_rate"] <= 1.0
        assert agg["pass_rate"] >= 0.25, f"整体通过率过低：{agg['pass_rate']}"

    def test_aggregate_statistics_consistent(self, samples, evaluator):
        """汇总统计自洽：通过数 + 未通过数 = 总数。"""
        results = evaluator.evaluate_batch(samples)
        agg = evaluator.aggregate(results)
        passed = sum(1 for r in results if r.passed)
        assert agg["passed"] == passed
        assert agg["passed"] + (agg["total"] - agg["passed"]) == agg["total"]


class TestExpectedOutcomes:
    """对评测集中「好/坏样本」的预期结果做精确断言。

    这套断言直观体现了评测框架的「鉴别能力」：
        - 分类任务：输出与参考答案一致 -> 通过；不一致 -> 未通过
        - 生成任务：语义高度重叠 -> 通过；完全跑偏 -> 未通过
    """

    # 预期每个样本是否通过（主指标 >= 阈值 0.5）
    EXPECTED_PASS = {
        "cls-001": True,   # 情感分析：正面 == 正面
        "cls-002": False,  # 情感分析：负面 != 中性
        "cls-003": True,   # 垃圾邮件：命中
        "cls-004": False,  # 主题分类：财经 != 体育
        "gen-001": True,   # 摘要：与参考高度重叠
        "gen-002": True,   # 翻译：与参考几乎一致
        "gen-003": False,  # 问答：答案"巴黎"被写成"巴黎是法国的首都"，
                           # 虽然语义正确，但 ROUGE-L 分数偏低 —— 这正暴露了
                           # 规则指标的局限，从而引出 LLM-as-judge 的必要性
        "gen-004": False,  # 问答：北京 != 上海，完全答错
    }

    @pytest.mark.parametrize("sample_id", list(EXPECTED_PASS.keys()))
    def test_expected_pass(self, sample_id, samples, evaluator):
        """逐条校验每个样本是否「按预期」通过 / 未通过。"""
        sample = next(s for s in samples if s.id == sample_id)
        result = evaluator.evaluate(sample)
        expected = self.EXPECTED_PASS[sample_id]
        assert result.passed is expected, (
            f"样本 {sample_id} 预期 {'通过' if expected else '未通过'}, "
            f"实际 scores={result.scores}"
        )
