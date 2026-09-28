"""运行一次完整评测，并生成 JSON / HTML 报告。

使用方式（在项目根目录下执行）：
    python run_eval.py

它会：
    1. 从 data/samples.json 加载评测集
    2. 用 Evaluator 逐条打分
    3. 汇总统计
    4. 生成 reports/report.json 和 reports/report.html
    5. 可选：用 MockJudge 演示 LLM-as-judge 的打分结果

这是「评测框架」的入口示例，帮你理解评测流程的完整链路；
而 tests/test_evaluation.py 则演示了如何把评测做成 pytest 测试。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能 import 到 src 下的 llm_eval 包
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_eval.dataset import load_dataset  # noqa: E402
from llm_eval.evaluator import Evaluator  # noqa: E402
from llm_eval.judge import MockJudge  # noqa: E402
from llm_eval.report import to_html, to_json  # noqa: E402


def main() -> None:
    # 1. 加载评测集
    samples = load_dataset(ROOT / "data" / "samples.json")
    print(f"已加载评测集，共 {len(samples)} 条样本")

    # 2. 规则指标评估
    evaluator = Evaluator(threshold=0.5)
    results = evaluator.evaluate_batch(samples)
    agg = evaluator.aggregate(results)

    # 3. 打印汇总
    print("\n===== 规则指标评测汇总 =====")
    print(f"样本总数: {agg['total']}")
    print(f"通过数:   {agg['passed']}")
    print(f"通过率:   {agg['pass_rate']:.2%}")
    print("平均指标得分:")
    for name, value in agg["avg_scores"].items():
        print(f"    {name:<20} {value:.4f}")

    # 4. 生成报告
    json_path = to_json(results, agg, ROOT / "reports" / "report.json")
    html_path = to_html(results, agg, ROOT / "reports" / "report.html")
    print(f"\n已生成 JSON 报告: {json_path}")
    print(f"已生成 HTML 报告: {html_path}")

    # 5. 演示 LLM-as-judge（离线 Mock）
    print("\n===== LLM-as-judge 演示（离线 Mock）=====")
    judge = MockJudge()
    for s in samples[:4]:  # 只演示前 4 条，避免输出过长
        verdict = judge.judge(s.prompt, s.reference, s.model_output)
        print(f"[{s.id}] 分数={verdict.score:.3f}  {verdict.reason}")


if __name__ == "__main__":
    main()
