"""报告生成模块 —— 把评测结果输出为 JSON / HTML 报告。

评测的价值在于「可读的结论」。本模块把 EvaluationResult 列表转成：
    - JSON 报告：机器可读，便于归档 / 二次分析 / 接入 CI
    - HTML 报告：人可读，带汇总卡片和逐条明细表格，适合分享
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .evaluator import EvaluationResult


def _fmt(score: float, digits: int = 4) -> str:
    """把分数格式化为固定小数位字符串。"""
    return f"{score:.{digits}f}"


def build_summary(results: List[EvaluationResult], aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """把评测结果整理成「汇总 + 明细」的结构，供两种报告共用。"""
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "statistics": aggregate,
        "results": [
            {
                "sample_id": r.sample_id,
                "task_type": r.task_type,
                "passed": r.passed,
                "scores": {k: round(v, 6) for k, v in r.scores.items()},
                "reference": r.details.get("reference", ""),
                "model_output": r.details.get("model_output", ""),
            }
            for r in results
        ],
    }
    return summary


def to_json(results: List[EvaluationResult], aggregate: Dict[str, Any], path: str | Path) -> Path:
    """生成 JSON 报告。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(build_summary(results, aggregate), f, ensure_ascii=False, indent=2)
    return path


def to_html(results: List[EvaluationResult], aggregate: Dict[str, Any], path: str | Path) -> Path:
    """生成 HTML 报告（内嵌 CSS，无需外部资源即可打开）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 汇总卡片数据 ----
    total = aggregate["total"]
    passed = aggregate["passed"]
    pass_rate = aggregate["pass_rate"]
    avg_scores = aggregate.get("avg_scores", {})

    # ---- 指标平均分卡片 ----
    avg_cards = ""
    for name, value in avg_scores.items():
        avg_cards += (
            f'<div class="card"><div class="card-value">{_fmt(value)}</div>'
            f'<div class="card-label">{name}</div></div>'
        )

    # ---- 逐条明细表格 ----
    rows = ""
    for r in results:
        score_text = ", ".join(f"{k}={_fmt(v)}" for k, v in r.scores.items())
        status = "通过" if r.passed else "未通过"
        status_cls = "pass" if r.passed else "fail"
        ref = _escape(r.details.get("reference", ""))
        out = _escape(r.details.get("model_output", ""))
        rows += (
            f"<tr>"
            f"<td>{_escape(r.sample_id)}</td>"
            f"<td>{_escape(r.task_type)}</td>"
            f"<td class='{status_cls}'>{status}</td>"
            f"<td class='mono'>{_escape(score_text)}</td>"
            f"<td>{ref}</td>"
            f"<td>{out}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>LLM 评测报告</title>
<style>
    body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; }}
    .sub {{ color: #6b7280; font-size: 13px; margin-bottom: 24px; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }}
    .card {{ background: #f3f4f6; border-radius: 10px; padding: 16px 20px; min-width: 120px; }}
    .card-value {{ font-size: 26px; font-weight: 700; color: #111827; }}
    .card-label {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f9fafb; font-weight: 600; }}
    .pass {{ color: #0f766e; font-weight: 700; }}
    .fail {{ color: #b91c1c; font-weight: 700; }}
    .mono {{ font-family: Consolas, monospace; font-size: 12px; }}
</style>
</head>
<body>
    <h1>LLM 评测报告</h1>
    <div class="sub">生成时间：{_escape(aggregate.get("generated_at", ""))}</div>

    <div class="cards">
        <div class="card"><div class="card-value">{total}</div><div class="card-label">样本总数</div></div>
        <div class="card"><div class="card-value">{passed}</div><div class="card-label">通过数</div></div>
        <div class="card"><div class="card-value">{_fmt(pass_rate)}</div><div class="card-label">通过率</div></div>
        {avg_cards}
    </div>

    <table>
        <thead>
            <tr><th>样本 ID</th><th>任务类型</th><th>结果</th><th>指标得分</th><th>参考答案</th><th>模型输出</th></tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def _escape(text: Any) -> str:
    """转义 HTML 特殊字符，防止内容破坏页面结构。"""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
