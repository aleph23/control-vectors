"""Tests for cache validation and fail-fast behaviour (Ticket 04).

Needs torch.
"""

import os
import pytest

torch = pytest.importorskip("torch")

from unittest.mock import MagicMock, patch


class TestCacheValidation:
    def _make_data_manager(self, tmp_path, dataset_manager):
        """Create a HiddenStateDataManager with a patched __init__ that skips
        model loading."""
        from hidden_state_data_manager import HiddenStateDataManager
        dm = object.__new__(HiddenStateDataManager)
        dm.model_handler = None
        dm.dataset_hidden_states = []
        dm.batch_size = 1
        dm.precision = "orig"
        dm.attn = "none"
        return dm

    def _make_dataset_manager(self, num_classes=3):
        dm_mgr = MagicMock()
        dm_mgr.get_num_classes.return_value = num_classes
        return dm_mgr

    def test_valid_cache_passes_validation(self, tmp_path):
        from hidden_state_data_manager import HiddenStateDataManager

        dm = object.__new__(HiddenStateDataManager)
        dm_mgr = self._make_dataset_manager(num_classes=2)

        # Build a well-formed cache: 2 classes, 3 samples each, 2 layers, feature dim 4
        data = []
        for _ in range(2):
            class_samples = []
            for _ in range(3):
                sample = [torch.randn(4) for _ in range(2)]
                class_samples.append(sample)
            data.append(class_samples)

        cache_path = tmp_path / "valid.pt"
        torch.save(data, cache_path)

        result = dm._load_and_validate_cache(str(cache_path), dm_mgr)
        assert result is not None
        assert len(result) == 2

    def test_empty_cache_rejected(self, tmp_path):
        from hidden_state_data_manager import HiddenStateDataManager

        dm = object.__new__(HiddenStateDataManager)
        dm_mgr = self._make_dataset_manager()

        cache_path = tmp_path / "empty.pt"
        torch.save([], cache_path)

        result = dm._load_and_validate_cache(str(cache_path), dm_mgr)
        assert result is None

    def test_wrong_class_count_rejected(self, tmp_path):
        from hidden_state_data_manager import HiddenStateDataManager

        dm = object.__new__(HiddenStateDataManager)
        dm_mgr = self._make_dataset_manager(num_classes=3)

        # Cache has 2 classes, expected 3
        data = []
        for _ in range(2):
            class_samples = [[torch.randn(4) for _ in range(2)] for _ in range(3)]
            data.append(class_samples)

        cache_path = tmp_path / "wrong_classes.pt"
        torch.save(data, cache_path)

        result = dm._load_and_validate_cache(str(cache_path), dm_mgr)
        assert result is None

    def test_ragged_per_class_sample_counts_rejected(self, tmp_path):
        from hidden_state_data_manager import HiddenStateDataManager

        dm = object.__new__(HiddenStateDataManager)
        dm_mgr = self._make_dataset_manager(num_classes=2)

        # Class 0 has 3 samples, class 1 has 2
        data = [
            [[torch.randn(4) for _ in range(2)] for _ in range(3)],
            [[torch.randn(4) for _ in range(2)] for _ in range(2)],
        ]

        cache_path = tmp_path / "ragged.pt"
        torch.save(data, cache_path)

        result = dm._load_and_validate_cache(str(cache_path), dm_mgr)
        assert result is None

    def test_inconsistent_layer_counts_rejected(self, tmp_path):
        from hidden_state_data_manager import HiddenStateDataManager

        dm = object.__new__(HiddenStateDataManager)
        dm_mgr = self._make_dataset_manager(num_classes=2)

        # Class 0 sample 1 has 2 layers, sample 2 has 3 layers
        data = []
        for _ in range(2):
            class_samples = [
                [torch.randn(4) for _ in range(2)],
                [torch.randn(4) for _ in range(3)],  # different layer count
            ]
            data.append(class_samples)

        cache_path = tmp_path / "bad_layers.pt"
        torch.save(data, cache_path)

        result = dm._load_and_validate_cache(str(cache_path), dm_mgr)
        assert result is None

    def test_weights_only_true(self, tmp_path):
        """torch.load must be called with weights_only=True."""
        from hidden_state_data_manager import HiddenStateDataManager

        dm = object.__new__(HiddenStateDataManager)
        dm_mgr = self._make_dataset_manager(num_classes=2)

        data = []
        for _ in range(2):
            class_samples = [[torch.randn(4) for _ in range(2)] for _ in range(3)]
            data.append(class_samples)

        cache_path = tmp_path / "valid.pt"
        torch.save(data, cache_path)

        with patch("torch.load", wraps=torch.load) as mock_load:
            dm._load_and_validate_cache(str(cache_path), dm_mgr)
            mock_load.assert_called_once()
            _, kwargs = mock_load.call_args
            assert kwargs.get("weights_only") is True, "torch.load not called with weights_only=True"