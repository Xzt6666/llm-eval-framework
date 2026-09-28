"""评测指标的单元测试。

这里用一些「手算可知答案」的例子来验证每个指标的计算是否正确。
指标是评测框架的基石，只有指标可信，评测结论才可信。
"""

from __future__ import annotations

import pytest

from llm_eval import metrics


class TestExactMatch:
    """精确匹配指标。"""

    def test_identical(self):
        assert metrics.exact_match("正面", "正面") is True

    def test_case_insensitive(self):
        # 默认 normalize=True，忽略大小写
        assert metrics.exact_match("Positive", "positive") is True

    def test_whitespace(self):
        # 忽略首尾空白
        assert metrics.exact_match("  hello  ", "hello") is True

    def test_different(self):
        assert metrics.exact_match("正面", "负面") is False


class TestAccuracy:
    """准确率指标。"""

    def test_all_correct(self):
        assert metrics.accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_half_correct(self):
        assert metrics.accuracy(["a", "x", "c", "y"], ["a", "b", "c", "d"]) == 0.5

    def test_empty(self):
        assert metrics.accuracy([], []) == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            metrics.accuracy(["a"], ["a", "b"])


class TestLabelPRF1:
    """标签级精确率 / 召回率 / F1。"""

    def test_perfect(self):
        result = metrics.label_precision_recall_f1(["a", "b"], ["a", "b"])
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_partial(self):
        # 2 个正确、2 个错误 -> P=R=0.5 -> F1=0.5
        result = metrics.label_precision_recall_f1(["a", "b", "c", "d"], ["a", "b", "x", "y"])
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == pytest.approx(0.5)


class TestTokenF1:
    """词级 F1。"""

    def test_exact(self):
        result = metrics.token_f1("北京 上海", "北京 上海")
        assert result["f1"] == 1.0

    def test_no_overlap(self):
        result = metrics.token_f1("苹果", "香蕉")
        assert result["f1"] == 0.0

    def test_partial(self):
        # 输出 "北京 广州"，参考 "北京 上海"：命中 "北京"
        # P = 1/2 = 0.5, R = 1/2 = 0.5, F1 = 0.5
        result = metrics.token_f1("北京 广州", "北京 上海")
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(0.5)


class TestRougeL:
    """ROUGE-L（最长公共子序列）。"""

    def test_identical(self):
        result = metrics.rouge_l("今天天气很好", "今天天气很好")
        assert result["f1"] == 1.0
        assert result["recall"] == 1.0

    def test_subsequence_keeps_order(self):
        # "a b c d" 与 "b d" 的 LCS 是 "b d"（保持顺序）
        result = metrics.rouge_l("a b c d", "b d")
        # recall = 2/2 = 1.0
        assert result["recall"] == 1.0
        # precision = 2/4 = 0.5
        assert result["precision"] == 0.5


class TestBleu:
    """BLEU（n-gram 精确率）。"""

    def test_identical(self):
        # 完全相同，BLEU 应接近 1
        score = metrics.bleu("the cat is on the mat", "the cat is on the mat")
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_empty_prediction(self):
        assert metrics.bleu("", "the cat") == 0.0

    def test_partial(self):
        # 部分重叠时分数应介于 0 和 1 之间
        score = metrics.bleu("the cat is on", "the cat is on the mat")
        assert 0.0 < score < 1.0


class TestCosineSimilarity:
    """TF-IDF 余弦相似度。"""

    def test_identical(self):
        assert metrics.cosine_similarity("机器学习很有趣", "机器学习很有趣") == pytest.approx(1.0)

    def test_identical_result_never_exceeds_one(self):
        """回归测试：浮点误差不得让相似度越过上界。

        背景：CPython 3.12 起 sum() 对浮点启用了补偿求和，3.10 / 3.11 没有，
        完全相同的文本在旧版本上可能算出 1.0000000000000002。
        cosine_similarity 内部必须夹取到 [0, 1]，否则 CI 在 3.10 / 3.11 会红。
        """
        sim = metrics.cosine_similarity("the quick brown fox", "the quick brown fox")
        assert sim <= 1.0
        assert sim == 1.0

    def test_result_always_in_range(self):
        # 遍历若干组合，保证任何输入下相似度都落在 [0, 1]
        pairs = [
            ("机器学习很有趣", "机器学习很有趣"),
            ("a b c d e f", "f e d c b a"),
            ("今天下雨", "今天下雨吗"),
            ("", "空输入"),
        ]
        for text1, text2 in pairs:
            sim = metrics.cosine_similarity(text1, text2)
            assert 0.0 <= sim <= 1.0, (text1, text2, sim)

    def test_unrelated(self):
        assert metrics.cosine_similarity("今天下雨", "量子力学") == 0.0

    def test_partial(self):
        sim = metrics.cosine_similarity("我喜欢机器学习", "机器学习很有趣")
        assert 0.0 < sim < 1.0
