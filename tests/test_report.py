"""报告生成模块的测试。

验证 JSON / HTML 报告能正确生成，且内容包含关键统计信息。
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_eval.evaluator import Evaluator
from llm_eval.report import to_html, to_json


def _eval_and_aggregate(samples, evaluator):
    results = evaluator.evaluate_batch(samples)
    return results, evaluator.aggregate(results)


class TestReportGeneration:
    def test_json_report(self, samples, evaluator, tmp_path: Path):
        results, agg = _eval_and_aggregate(samples, evaluator)
        out = to_json(results, agg, tmp_path / "report.json")

        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["statistics"]["total"] == len(samples)
        assert "results" in data
        assert len(data["results"]) == len(samples)

    def test_html_report(self, samples, evaluator, tmp_path: Path):
        results, agg = _eval_and_aggregate(samples, evaluator)
        out = to_html(results, agg, tmp_path / "report.html")

        assert out.exists()
        content = out.read_text(encoding="utf-8")
        # HTML 报告应包含标题、通过率、明细表格
        assert "LLM 评测报告" in content
        assert "通过率" in content
        assert "<table>" in content
