# LLM Eval —— 大语言模型（AI）评测框架

一个面向 **LLM 输出评测** 的轻量 Python 项目，用 **pytest 组织评测用例**，覆盖从
「评测数据集 → 指标计算 → 评估 → LLM 裁判 → 报告」的完整链路。

> 适合用于：补充「AI 评测」相关项目经验、学习 LLM 评测的核心概念、作为简历项目。

---

## 一、这个项目解决什么问题

大模型（LLM）上线前，需要回答一个关键问题：**它答得怎么样？** AI 评测就是
用一套「数据集 + 指标 + 流程」来量化回答这个问题。

本项目做了三件核心的事：

1. **定义评测数据集** —— 一组带参考答案的样本（分类 + 生成两类任务）
2. **实现评测指标** —— 精确匹配 / 准确率 / F1 / ROUGE-L / BLEU / 余弦相似度（纯 Python 手写）
3. **用 pytest 跑评测** —— 每一条评测样本对应一条测试用例，结果即通过/失败，可接入 CI

此外还实现了 **LLM-as-judge（用大模型当裁判）**，解决「规则指标无法判断语义」的局限。

---

## 二、核心概念速览（面试常考）

| 概念 | 说明 | 本项目对应 |
|------|------|-----------|
| 评测集（Eval Set） | 带黄金标准答案的样本集合 | `data/samples.json` |
| 指标（Metric） | 量化模型输出与参考答案的接近程度 | `metrics.py` |
| 精确匹配 | 输出 == 参考答案 | `exact_match` |
| 准确率 | 预测正确比例 | `accuracy` |
| ROUGE-L | 基于最长公共子序列，用于摘要 | `rouge_l` |
| BLEU | 基于 n-gram 精确率，用于翻译 | `bleu` |
| LLM-as-judge | 让大模型当裁判打分 | `judge.py` |

---

## 三、项目结构

```
llm_eval/
├── pyproject.toml          # pytest 配置 + 包信息
├── requirements.txt        # 依赖（仅 pytest、requests）
├── run_eval.py             # 入口脚本：跑一次完整评测并生成报告
├── README.md
├── data/
│   └── samples.json        # 评测集（8 条样例，分类 + 生成）
├── src/llm_eval/           # 核心包
│   ├── __init__.py
│   ├── dataset.py          # 评测数据模型与加载
│   ├── metrics.py          # 各评测指标（纯 Python 实现）
│   ├── evaluator.py        # 评估器：按任务类型打分、汇总
│   ├── judge.py            # LLM-as-judge（含离线 Mock 与 OpenAI 接口）
│   └── report.py           # 生成 JSON / HTML 报告
├── tests/                  # pytest 测试（评测用例）
│   ├── conftest.py         # 共享 fixture（加载数据、构建评估器）
│   ├── test_metrics.py     # 指标单元测试
│   ├── test_dataset.py     # 数据集加载测试
│   ├── test_evaluation.py  # 端到端评测（参数化）
│   ├── test_judge.py       # LLM 裁判测试
│   └── test_report.py      # 报告生成测试
└── reports/                # 生成的评测报告（运行后产生）
```

---

## 四、快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 跑全部评测用例（43 条测试，含指标单测 + 端到端评测）
pytest

# 3. 跑一次完整评测并生成报告
python run_eval.py
# 会生成 reports/report.json 和 reports/report.html
```

运行 `pytest` 后，你会看到每一条评测样本对应一条测试用例，例如：

```
tests/test_evaluation.py::TestExpectedOutcomes::test_expected_pass[cls-001] PASSED
tests/test_evaluation.py::TestExpectedOutcomes::test_expected_pass[gen-003] PASSED
```

---

## 五、一个值得思考的细节：规则指标的局限

看这条样本（`gen-003`）：

- 参考答案：`巴黎`
- 模型输出：`巴黎是法国的首都`

语义完全正确，但因为模型「啰嗦」了，ROUGE-L 分数只有约 0.4，被判为「未通过」。
这正说明**规则指标只看字符串重叠、无法判断语义**，从而引出 **LLM-as-judge**
的必要性 —— 面试时讲出这个点会非常加分。

---

## 六、LLM-as-judge 怎么用

项目里提供了两层实现（见 `judge.py`）：

1. **`MockJudge`**：离线模拟，不联网即可跑通流程（用于测试）
2. **`OpenAIJudge`**：用 `requests` 直接调用 OpenAI 兼容接口，把「评判」交给真实大模型

使用真实裁判：

```python
from llm_eval.judge import OpenAIJudge

judge = OpenAIJudge(base_url="https://api.openai.com/v1", model="gpt-4o-mini")
verdict = judge.judge(
    prompt="法国的首都是哪里？",
    reference="巴黎",
    model_output="巴黎是法国的首都",
)
print(verdict.score, verdict.reason)  # 语义正确，分数会比规则指标更高
```

需要先设置环境变量 `OPENAI_API_KEY`。

---

## 七、学习路线建议

1. 先读 `metrics.py`，逐个指标对照注释理解数学原理（建议手推一遍 ROUGE-L 的 DP）
2. 再读 `evaluator.py`，理解「按任务类型选指标」的调度逻辑
3. 读 `tests/test_evaluation.py`，理解「评测用例 = pytest 测试」的设计
4. 跑 `python run_eval.py`，打开生成的 HTML 报告看效果
5. 进阶：给 `data/samples.json` 加新样本，或给 `metrics.py` 加新指标，再补一条测试
