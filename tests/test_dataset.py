"""评测数据集加载的测试。

验证 load_dataset 能正确解析 JSON，并正确识别任务类型。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval.dataset import EvalSample, TaskType, load_dataset, save_dataset


class TestLoadDataset:
    """数据集加载。"""

    def test_load_from_json(self, samples):
        # conftest 提供的 fixture：从 data/samples.json 加载
        assert isinstance(samples, list)
        assert len(samples) == 8
        assert all(isinstance(s, EvalSample) for s in samples)

    def test_task_type_parsing(self, samples):
        # 前 4 条是分类，后 4 条是生成
        by_type = {TaskType.CLASSIFICATION: 0, TaskType.GENERATION: 0}
        for s in samples:
            by_type[s.task_type] += 1
        assert by_type[TaskType.CLASSIFICATION] == 4
        assert by_type[TaskType.GENERATION] == 4

    def test_fields_present(self, samples):
        # 每条样本都必须有 id、prompt、reference
        for s in samples:
            assert s.id
            assert s.prompt
            assert s.reference

    def test_unknown_task_type_defaults_to_generation(self, tmp_path: Path):
        # 未知任务类型应回退为 generation，而不是抛异常
        data = [
            {
                "id": "x-1",
                "prompt": "p",
                "reference": "r",
                "model_output": "o",
                "task_type": "unknown-type",
            }
        ]
        f = tmp_path / "tmp.json"
        f.write_text(
            '[\n  {"id": "x-1", "prompt": "p", "reference": "r", '
            '"model_output": "o", "task_type": "unknown-type"}\n]',
            encoding="utf-8",
        )
        loaded = load_dataset(f)
        assert loaded[0].task_type == TaskType.GENERATION


class TestSaveDataset:
    """数据集保存（回写）。"""

    def test_round_trip(self, samples, tmp_path: Path):
        # 保存后重新加载，字段应一致
        out = tmp_path / "out.json"
        save_dataset(samples, out)
        reloaded = load_dataset(out)
        assert len(reloaded) == len(samples)
        assert reloaded[0].id == samples[0].id
        assert reloaded[0].reference == samples[0].reference
