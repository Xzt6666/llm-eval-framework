"""LLM-as-judge（裁判）的测试。

这里只测试离线 MockJudge（不联网）。真正的 OpenAIJudge 需要 API Key，
不在此测试（避免测试依赖外部网络和密钥）。
"""

from __future__ import annotations

from llm_eval.judge import JudgeVerdict, MockJudge


class TestMockJudge:
    """离线模拟裁判。"""

    def test_perfect_answer(self):
        judge = MockJudge()
        verdict = judge.judge(
            prompt="法国的首都是哪里？",
            reference="巴黎",
            model_output="巴黎",
        )
        assert isinstance(verdict, JudgeVerdict)
        # 完全一致，分数应接近 1
        assert verdict.score == 1.0

    def test_wrong_answer(self):
        judge = MockJudge()
        verdict = judge.judge(
            prompt="中国的首都是哪里？",
            reference="北京",
            model_output="上海",
        )
        # 完全无关，分数应为 0
        assert verdict.score == 0.0

    def test_partial_answer(self):
        judge = MockJudge()
        verdict = judge.judge(
            prompt="总结这段文本",
            reference="公司营收增长百分之二十",
            model_output="公司营收增长",
        )
        # 部分重叠，分数应介于 0 和 1 之间
        assert 0.0 < verdict.score < 1.0

    def test_has_reason(self):
        judge = MockJudge()
        verdict = judge.judge("q", "a", "a")
        assert verdict.reason  # 理由非空
