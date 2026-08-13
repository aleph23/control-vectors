"""Tests for DatasetManager — stdlib only, no torch."""

import json
import pytest

from dataset_manager import DatasetManager


class TestDatasetManager:
    def test_class_names_with_baseline(self, tmp_data_dir):
        stems = str(tmp_data_dir / "prompt_stems.json")
        conts = str(tmp_data_dir / "continuations" / "test_continuations.json")
        prompts = str(tmp_data_dir / "writing_prompts.txt")

        dm = DatasetManager(stems, conts, prompts, num_prompt_samples=12)

        assert dm.class_names == ["baseline", "class_a", "class_b"]
        assert dm.get_num_classes() == 3

    def test_class_names_without_baseline(self, tmp_data_dir):
        stems = str(tmp_data_dir / "prompt_stems.json")
        conts = str(tmp_data_dir / "continuations" / "test_continuations.json")
        prompts = str(tmp_data_dir / "writing_prompts.txt")

        dm = DatasetManager(stems, conts, prompts, num_prompt_samples=12, use_baseline_class=False)

        assert dm.class_names == ["class_a", "class_b"]
        assert dm.get_num_classes() == 2

    def test_num_samples_is_total(self, tmp_data_dir):
        """--num-samples is a total, divided evenly across classes."""
        stems = str(tmp_data_dir / "prompt_stems.json")
        conts = str(tmp_data_dir / "continuations" / "test_continuations.json")
        prompts = str(tmp_data_dir / "writing_prompts.txt")

        dm = DatasetManager(stems, conts, prompts, num_prompt_samples=12)

        assert dm.get_total_samples() == 12
        # 3 classes * 4 samples each = 12
        for dataset in dm.datasets:
            assert len(dataset) == 4

    def test_num_prompt_samples_too_small(self, tmp_data_dir):
        stems = str(tmp_data_dir / "prompt_stems.json")
        conts = str(tmp_data_dir / "continuations" / "test_continuations.json")
        prompts = str(tmp_data_dir / "writing_prompts.txt")

        with pytest.raises(ValueError):
            DatasetManager(stems, conts, prompts, num_prompt_samples=2)  # 2 / 3 = 0 per class

    def test_triplet_invariant(self, tmp_data_dir):
        """For each sample index i, the writing prompt is identical across all classes."""
        stems = str(tmp_data_dir / "prompt_stems.json")
        conts = str(tmp_data_dir / "continuations" / "test_continuations.json")
        prompts = str(tmp_data_dir / "writing_prompts.txt")

        dm = DatasetManager(stems, conts, prompts, num_prompt_samples=30)

        num_classes = dm.get_num_classes()
        samples_per_class = 30 // num_classes

        for i in range(samples_per_class):
            prompts_at_i = [dm.datasets[c][i][1] for c in range(num_classes)]
            assert len(set(prompts_at_i)) == 1, f"Sample {i}: writing prompts differ across classes"

    def test_missing_pre_key(self, tmp_data_dir):
        stems = tmp_data_dir / "bad_stems.json"
        stems.write_text(json.dumps({"post": ["an author"]}))
        conts = str(tmp_data_dir / "continuations" / "test_continuations.json")
        prompts = str(tmp_data_dir / "writing_prompts.txt")

        with pytest.raises(ValueError, match="must contain 'pre' and 'post'"):
            DatasetManager(str(stems), conts, prompts, num_prompt_samples=12)

    def test_missing_classes_key(self, tmp_data_dir):
        stems = str(tmp_data_dir / "prompt_stems.json")
        conts_file = tmp_data_dir / "bad_conts.json"
        conts_file.write_text(json.dumps({"data": [["a", "b"]]}))
        prompts = str(tmp_data_dir / "writing_prompts.txt")

        with pytest.raises(ValueError, match="Invalid or no data"):
            DatasetManager(stems, str(conts_file), prompts, num_prompt_samples=12)

    def test_missing_data_key(self, tmp_data_dir):
        stems = str(tmp_data_dir / "prompt_stems.json")
        conts_file = tmp_data_dir / "bad_conts.json"
        conts_file.write_text(json.dumps({"classes": ["a", "b"]}))
        prompts = str(tmp_data_dir / "writing_prompts.txt")

        with pytest.raises(ValueError, match="Invalid or no data"):
            DatasetManager(stems, str(conts_file), prompts, num_prompt_samples=12)

    def test_file_not_found(self, tmp_data_dir):
        stems = str(tmp_data_dir / "nonexistent.json")
        conts = str(tmp_data_dir / "continuations" / "test_continuations.json")
        prompts = str(tmp_data_dir / "writing_prompts.txt")

        with pytest.raises(FileNotFoundError):
            DatasetManager(stems, conts, prompts, num_prompt_samples=12)

    def test_malformed_json(self, tmp_data_dir):
        stems = tmp_data_dir / "bad.json"
        stems.write_text("not json")
        conts = str(tmp_data_dir / "continuations" / "test_continuations.json")
        prompts = str(tmp_data_dir / "writing_prompts.txt")

        with pytest.raises(ValueError, match="Failed to decode JSON"):
            DatasetManager(str(stems), conts, prompts, num_prompt_samples=12)