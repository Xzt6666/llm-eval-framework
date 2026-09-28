"""LLM-as-judge 模块 —— 用大语言模型当「裁判」来给答案打分。

传统的规则指标（ROUGE / BLEU / F1）只做字符串级别的匹配，无法判断
「语义是否真的正确」。例如：
    参考："巴黎是法国的首都"
    模型："法国首都是巴黎"        -> 语义正确，但 ROUGE 分数可能不高

LLM-as-judge 的思路：把「评判」本身也交给一个大模型，让它阅读参考答案和
模型输出，输出一个分数（如 1~5 分）和理由。这在真实工业评测（如 MT-Bench、
Chatbot Arena 等）里非常常用。

本模块提供两层实现：
    - BaseJudge  : 抽象基类，定义 judge() 接口
    - MockJudge  : 离线模拟裁判（用规则打分），不依赖网络，便于本地跑通
    - OpenAIJudge: 真正调用 OpenAI 兼容接口（用 requests 直接发 HTTP 请求）
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import requests

from . import metrics


@dataclass
class JudgeVerdict:
    """裁判给出的评判结果。

    属性说明：
        score  : 分数（0~1 之间的连续值，或 0~100）
        reason : 裁判给出的理由（自然语言）
        raw    : 原始返回内容（便于排查 / 调试）
    """

    score: float
    reason: str = ""
    raw: Any = None


class BaseJudge(ABC):
    """裁判抽象基类。

    所有裁判都需要实现 judge() 方法：
        输入：prompt（题目）、reference（参考答案）、model_output（待评输出）
        输出：JudgeVerdict（分数 + 理由）
    """

    @abstractmethod
    def judge(self, prompt: str, reference: str, model_output: str) -> JudgeVerdict:
        """对待评输出给出分数与理由。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 离线模拟裁判：不联网也能跑通整个评测流程
# ---------------------------------------------------------------------------
class MockJudge(BaseJudge):
    """离线模拟裁判。

    真实场景这里应该调用 LLM，但为了离线可运行 / 单元测试，我们用
    「规则打分」模拟：以 token F1 和余弦相似度的加权平均作为分数，
    并给出一个模板化的理由。
    """

    def judge(self, prompt: str, reference: str, model_output: str) -> JudgeVerdict:
        # 用规则指标近似"语义正确程度"
        f1 = metrics.token_f1(model_output, reference)["f1"]
        cos = metrics.cosine_similarity(model_output, reference)
        score = round(0.5 * f1 + 0.5 * cos, 4)

        reason = (
            f"离线规则裁判：token F1={f1:.3f}，余弦相似度={cos:.3f}，"
            f"加权得分={score:.3f}"
        )
        return JudgeVerdict(score=score, reason=reason, raw={"f1": f1, "cosine": cos})


# ---------------------------------------------------------------------------
# 真正的 LLM 裁判：调用 OpenAI 兼容接口
# ---------------------------------------------------------------------------
class OpenAIJudge(BaseJudge):
    """调用 OpenAI 兼容接口的 LLM 裁判。

    说明：
        - 直接使用 requests.post 发 HTTP 请求（而非封装的 fetch 之类），
          便于看清每一个请求 / 响应的细节。
        - 通过 system prompt 引导模型「只输出 JSON」，方便程序解析。
        - base_url 可替换为任何 OpenAI 兼容端点（如本地 vLLM、Ollama、
          通义千问 / 豆包等的兼容地址）。

    使用前提：需要设置环境变量 OPENAI_API_KEY。
    """

    # 用于引导 LLM 输出结构化 JSON 的提示词
    _SYSTEM_PROMPT = (
        "你是一个严格但公正的 LLM 输出评测裁判。"
        "请阅读「题目」「参考答案」「模型输出」，判断模型输出的正确性，"
        "并只返回一个 JSON 对象，格式如下：\n"
        '{"score": 0到1之间的小数, "reason": "简短理由"}\n'
        "不要输出 JSON 以外的任何内容。"
    )

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: int = 30,
    ) -> None:
        """初始化裁判。

        参数：
            api_key  : API 密钥。若不传，则从环境变量 OPENAI_API_KEY 读取
            base_url : OpenAI 兼容接口的基地址
            model    : 使用的模型名
            timeout  : 请求超时（秒）
        """
        import os

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _build_user_prompt(self, prompt: str, reference: str, model_output: str) -> str:
        """拼装给裁判的完整输入。"""
        return (
            f"【题目】\n{prompt}\n\n"
            f"【参考答案】\n{reference}\n\n"
            f"【模型输出】\n{model_output}\n"
        )

    def judge(self, prompt: str, reference: str, model_output: str) -> JudgeVerdict:
        """调用 LLM 裁判打分。"""
        if not self.api_key:
            raise ValueError("缺少 API Key：请设置环境变量 OPENAI_API_KEY 或传入 api_key")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(prompt, reference, model_output)},
            ],
            "temperature": 0,  # 0 表示尽量确定性，评测场景更稳定
        }

        # 直接使用 requests.post，能看到完整的请求 / 响应过程
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()  # 非 2xx 会抛异常

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # 尝试解析 JSON，失败时回退为字符串
        try:
            parsed = json.loads(content)
            score = float(parsed.get("score", 0.0))
            reason = str(parsed.get("reason", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            score, reason = 0.0, content

        # 分数归一化到 0~1
        score = max(0.0, min(1.0, score))
        return JudgeVerdict(score=score, reason=reason, raw=data)
