"""评测指标模块 —— 全部用纯 Python 实现，便于理解每个指标的数学原理。

在 LLM 评测中，指标用来「量化」模型输出与参考答案有多接近。
常见指标按任务分类：

    分类任务：
        - exact_match  : 精确匹配（输出是否完全等于参考答案）
        - accuracy     : 准确率（预测正确的比例）
        - label_precision_recall_f1 : 标签级别的精确率 / 召回率 / F1

    生成任务（摘要 / 翻译 / 问答）：
        - token_f1     : 词级别的 F1（看模型命中参考中多少关键词）
        - rouge_l      : 基于最长公共子序列的召回率 / F1（摘要常用）
        - bleu         : 基于 n-gram 精确率的分数（翻译常用）
        - cosine_similarity : 基于 TF-IDF 的余弦相似度（语义接近程度）

提示：这里每个指标都刻意写成了「无第三方依赖」的版本，方便你逐步调试、
逐行读懂。理解之后，再对照 rouge-score / sacrebleu 等官方库即可。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

# ---------------------------------------------------------------------------
# 基础工具：文本规范化 与 分词
# ---------------------------------------------------------------------------

# 匹配英文单词 / 数字 / 中文字符（中文按单字切分）
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def normalize_text(text: str) -> str:
    """把文本规范化为小写、去掉首尾空白。

    统一大小写可以让「匹配」不受大小写影响，是评测前最常见的预处理。
    """
    return text.strip().lower()


def tokenize(text: str) -> List[str]:
    """把一段文本切分成 token 列表。

    规则：英文按单词、数字切分，中文按单字切分。
    例如 "Hello 世界 123" -> ["hello", "世", "界", "123"]
    """
    return _TOKEN_RE.findall(normalize_text(text))


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    """生成 n-gram 序列。

    例如 tokens=["a","b","c"], n=2 -> [("a","b"), ("b","c")]
    """
    if n <= 0 or n > len(tokens):
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# 分类任务指标
# ---------------------------------------------------------------------------


def exact_match(pred: str, reference: str, normalize: bool = True) -> bool:
    """精确匹配：模型输出与参考答案是否完全一致。

    参数：
        normalize : 是否先做规范化（小写、去空白）。默认 True，更宽容。
    """
    if normalize:
        pred, reference = normalize_text(pred), normalize_text(reference)
    return pred == reference


def accuracy(preds: Sequence[str], refs: Sequence[str], normalize: bool = True) -> float:
    """准确率 = 预测正确的样本数 / 总样本数。

    适用于分类任务。参数 preds / refs 长度必须一致。
    """
    if len(preds) != len(refs):
        raise ValueError(f"预测与参考长度不一致: {len(preds)} vs {len(refs)}")
    if not preds:
        return 0.0
    correct = sum(exact_match(p, r, normalize) for p, r in zip(preds, refs))
    return correct / len(preds)


def label_precision_recall_f1(
    preds: Sequence[str], refs: Sequence[str]
) -> Dict[str, float]:
    """计算标签级别的精确率 / 召回率 / F1（micro 平均）。

    思路：
        - 先把所有标签去重，得到标签集合
        - TP = 预测为该标签且真实也是该标签的数量（按标签累加）
        - FP = 预测为该标签但真实不是的数量
        - FN = 真实是该标签但预测不是的数量
        - Precision = TP / (TP + FP)，Recall = TP / (TP + FN)，F1 = 调和平均

    micro 平均把每个「样本-标签」的判定等价看待，样本不均衡时更公平。
    """
    tp = fp = fn = 0
    for pred, ref in zip(preds, refs):
        p, r = normalize_text(pred), normalize_text(ref)
        if p == r:
            tp += 1
        else:
            fp += 1  # 预测错了（多算一个 FP）
            fn += 1  # 也漏掉了真实标签（多算一个 FN）
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# 生成任务指标
# ---------------------------------------------------------------------------


def token_f1(pred: str, reference: str) -> Dict[str, float]:
    """词级别的 F1（常用于问答 / 抽取类任务）。

    把模型输出和参考答案都切成 token 集合：
        - 命中数 = 两集合交集大小
        - Precision = 命中数 / 模型输出 token 数（模型输出里有多少是对的）
        - Recall    = 命中数 / 参考答案 token 数（参考答案里的关键词命中了多少）
        - F1        = 两者的调和平均

    相比精确匹配，它能给「部分正确」的输出一个合理的分数。
    """
    pred_tokens = Counter(tokenize(pred))
    ref_tokens = Counter(tokenize(reference))

    # 交集：每个 token 取两边出现次数的较小值（考虑到重复词）
    common = sum((pred_tokens & ref_tokens).values())
    total_pred = sum(pred_tokens.values())
    total_ref = sum(ref_tokens.values())

    precision = common / total_pred if total_pred else 0.0
    recall = common / total_ref if total_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """最长公共子序列（Longest Common Subsequence）长度。

    用动态规划求解，时间复杂度 O(m * n)。
    「子序列」不要求连续，但要求保持相对顺序 —— 这正是 ROUGE-L 的核心。
    """
    m, n = len(a), len(b)
    # dp[i][j] 表示 a 前 i 个、b 前 j 个元素的最长公共子序列长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(pred: str, reference: str) -> Dict[str, float]:
    """ROUGE-L 指标（基于最长公共子序列），常用于摘要评测。

    计算公式：
        R_lcs = LCS(pred, ref) / len(ref)    # 召回率：参考里有多少内容被覆盖
        P_lcs = LCS(pred, ref) / len(pred)   # 精确率：输出里有多少是有效的
        F     = 2 * P * R / (P + R)          # 调和平均

    LCS 不要求词连续，因此对「顺序略有变化但内容覆盖」的输出更宽容。
    """
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(reference)
    lcs = _lcs_length(pred_tokens, ref_tokens)

    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "lcs": lcs}


def _modified_precision(pred_tokens: Sequence[str], ref_tokens: Sequence[str], n: int) -> float:
    """计算 n-gram 的「修正精确率」（modified precision）。

    普通精确率的问题：模型若把同一个正确 n-gram 重复很多次，分数会虚高。
    修正做法：对每个 n-gram，命中的次数「封顶」为它在参考中出现的次数。

    公式：P_n = sum(min(count_clip(pred, g), count_ref(g))) / sum(count_pred(g))
    """
    pred_ngrams = Counter(_ngrams(pred_tokens, n))
    ref_ngrams = Counter(_ngrams(ref_tokens, n))

    clipped = 0
    for gram, count in pred_ngrams.items():
        # 命中次数不能超过参考中该 n-gram 的出现次数
        clipped += min(count, ref_ngrams.get(gram, 0))

    total = sum(pred_ngrams.values())
    return clipped / total if total else 0.0


def bleu(pred: str, reference: str, max_n: int = 4, weights: Sequence[float] | None = None) -> float:
    """BLEU 指标（基于 n-gram 精确率），常用于翻译评测。

    计算步骤：
        1. 对 n = 1..max_n，计算 n-gram 的修正精确率 p_n
        2. 对 p_1..p_n 取几何平均（加权）
        3. 施加「长度惩罚」BP：模型输出太短时惩罚
            若 c > r: BP = 1
            若 c <= r: BP = exp(1 - r/c)
        4. BLEU = BP * exp( sum(w_n * log(p_n)) )

    返回 0~1 之间的分数（通常再乘 100 表示百分比）。
    """
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(reference)

    if not pred_tokens:
        return 0.0

    if weights is None:
        # 默认每个 n-gram 等权重
        weights = [1.0 / max_n] * max_n

    # 1. 计算各阶 n-gram 的修正精确率
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p_n = _modified_precision(pred_tokens, ref_tokens, n)
        # 若某阶精确率为 0，为避免 log(0)，用一个极小值代替；
        # 实际工程里常见做法是给 0 的阶赋 0，这里用平滑处理更稳健。
        log_sum += weights[n - 1] * math.log(p_n if p_n > 0 else 1e-12)

    # 2. 长度惩罚
    c, r = len(pred_tokens), len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0

    # 3. 组合
    return bp * math.exp(log_sum)


# ---------------------------------------------------------------------------
# 语义相似度（可选进阶）
# ---------------------------------------------------------------------------


def _tf(tokens: Sequence[str]) -> Dict[str, float]:
    """词频（Term Frequency）：每个词出现的频率。"""
    counter = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {word: count / total for word, count in counter.items()}


def _idf(documents: Sequence[Sequence[str]]) -> Dict[str, float]:
    """逆文档频率（Inverse Document Frequency）：越常见的词权重越低。

    公式：idf(w) = log((1 + N) / (1 + df(w))) + 1
    其中 N 是文档总数，df(w) 是包含该词的文档数。
    """
    n = len(documents)
    df: Counter = Counter()
    for doc in documents:
        df.update(set(doc))  # 每个文档里同一个词只算一次

    idf = {}
    for word, freq in df.items():
        idf[word] = math.log((1 + n) / (1 + freq)) + 1
    return idf


def cosine_similarity(text1: str, text2: str) -> float:
    """基于 TF-IDF 的余弦相似度，衡量两段文本的语义接近程度。

    步骤：
        1. 分别分词
        2. 计算 TF-IDF 权重向量
        3. 求两向量的余弦夹角 cos = (A·B) / (|A| * |B|)

    返回 0~1，越接近 1 表示越相似。
    """
    t1, t2 = tokenize(text1), tokenize(text2)
    if not t1 or not t2:
        return 0.0

    idf = _idf([t1, t2])
    tf1, tf2 = _tf(t1), _tf(t2)

    # 只用两段文本共同出现的词即可，其余词点积为 0
    common_words = set(tf1) & set(tf2)
    dot = sum(tf1[w] * idf[w] * tf2[w] * idf[w] for w in common_words)

    # 向量模长（用所有词计算）
    norm1 = math.sqrt(sum((tf1[w] * idf[w]) ** 2 for w in tf1))
    norm2 = math.sqrt(sum((tf2[w] * idf[w]) ** 2 for w in tf2))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
