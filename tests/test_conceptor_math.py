"""Tests for conceptor math functions.

Needs torch. Pure-function tests; does not instantiate ConceptorAnalyzer.
"""

import pytest

torch = pytest.importorskip("torch")

from conceptor_analyzer import (
    compute_conceptor,
    reconstruct_conceptor,
    compute_optimal_low_rank_conceptor,
    compute_conceptor_low_rank,
    compute_low_rank_conceptor_automatic,
    ConceptorAnalyzer,
    ConceptorRepresentation,
)


class TestComputeConceptor:
    def test_returns_symmetric_matrix(self):
        X = torch.randn(200, 32)
        C = compute_conceptor(X, aperture=0.1)
        assert C.shape == (32, 32)
        assert torch.allclose(C, C.T, atol=1e-5)

    def test_eigenvalues_in_01(self):
        X = torch.randn(200, 32)
        C = compute_conceptor(X, aperture=0.1)
        eigenvalues = torch.linalg.eigvalsh(C)
        assert torch.all(eigenvalues >= 0.0)
        assert torch.all(eigenvalues < 1.0)

    def test_larger_aperture_pushes_eigenvalues_toward_one(self):
        X = torch.randn(200, 32)
        C_small = compute_conceptor(X, aperture=0.01)
        C_large = compute_conceptor(X, aperture=1.0)
        eig_small = torch.linalg.eigvalsh(C_small)
        eig_large = torch.linalg.eigvalsh(C_large)
        # Larger aperture should give larger eigenvalues (monotonic)
        assert torch.all(eig_large >= eig_small - 1e-6)


class TestReconstructConceptor:
    def test_round_trips_optimal_low_rank(self):
        X = torch.randn(200, 32)
        C = compute_conceptor(X, aperture=0.1)
        U, s = compute_optimal_low_rank_conceptor(C, thres=1e-4)
        C_reconstructed = reconstruct_conceptor(U, s)
        # Should be close to the original
        diff = torch.norm(C - C_reconstructed) / torch.norm(C)
        assert diff < 0.1, f"Reconstruction error {diff:.4f} too large"


class TestLowRankHelpers:
    def test_manual_low_rank_shapes(self):
        X = torch.randn(200, 32)
        U, s = compute_conceptor_low_rank(X, aperture=0.1, rank=10)
        assert U.shape == (32, 10)
        assert s.shape == (10,)

    def test_automatic_low_rank_shapes(self):
        X = torch.randn(200, 32)
        U, s = compute_low_rank_conceptor_automatic(X, aperture=0.1, variance_thres=0.99)
        assert U.shape[0] == 32
        assert U.shape[1] == s.shape[0]
        assert U.shape[1] <= 32  # k <= d

    def test_optimal_low_rank_shapes(self):
        X = torch.randn(200, 32)
        C = compute_conceptor(X, aperture=0.1)
        U, s = compute_optimal_low_rank_conceptor(C, thres=1e-4)
        assert U.shape[0] == 32
        assert U.shape[1] == s.shape[0]


class TestConceptorAnalyzerInit:
    def _make_data_manager(self, num_layers=4, num_classes=3, hidden_dim=16):
        """Build a minimal stub HiddenStateDataManager."""
        from unittest.mock import MagicMock

        dm = MagicMock()
        dm.get_num_layers.return_value = num_layers
        dm.get_num_dataset_types.return_value = num_classes

        # Return small synthetic datasets for each layer
        sample_count = 10
        datasets = []
        for _ in range(num_classes):
            datasets.append(torch.randn(sample_count, hidden_dim))
        dm.get_datasets.return_value = datasets

        return dm

    def test_non_positive_aperture_raises(self):
        dm = self._make_data_manager()
        with pytest.raises(ValueError, match="Aperture must be positive"):
            ConceptorAnalyzer(
                hidden_state_data_manager=dm,
                skip_early_layers=0,
                skip_late_layers=0,
                aperture=0.0,
                center="none",
                lora=False,
                lora_method=None,
                rank=None,
                variance_thres=None,
                thres=None,
            )

    def test_invalid_center_raises(self):
        dm = self._make_data_manager()
        with pytest.raises(ValueError, match="Invalid center"):
            ConceptorAnalyzer(
                hidden_state_data_manager=dm,
                skip_early_layers=0,
                skip_late_layers=0,
                aperture=0.1,
                center="invalid",
                lora=False,
                lora_method=None,
                rank=None,
                variance_thres=None,
                thres=None,
            )

    def test_lora_without_method_raises(self):
        dm = self._make_data_manager()
        with pytest.raises(ValueError, match="lora_method must be one of"):
            ConceptorAnalyzer(
                hidden_state_data_manager=dm,
                skip_early_layers=0,
                skip_late_layers=0,
                aperture=0.1,
                center="none",
                lora=True,
                lora_method=None,
                rank=None,
                variance_thres=None,
                thres=None,
            )

    def test_manual_lora_without_rank_raises(self):
        dm = self._make_data_manager()
        with pytest.raises(ValueError, match="'rank' must be specified"):
            ConceptorAnalyzer(
                hidden_state_data_manager=dm,
                skip_early_layers=0,
                skip_late_layers=0,
                aperture=0.1,
                center="none",
                lora=True,
                lora_method="manual",
                rank=None,
                variance_thres=None,
                thres=None,
            )

    def test_automatic_lora_without_variance_thres_raises(self):
        dm = self._make_data_manager()
        with pytest.raises(ValueError, match="'variance_thres' must be specified"):
            ConceptorAnalyzer(
                hidden_state_data_manager=dm,
                skip_early_layers=0,
                skip_late_layers=0,
                aperture=0.1,
                center="none",
                lora=True,
                lora_method="automatic",
                rank=None,
                variance_thres=None,
                thres=None,
            )

    def test_optimal_lora_without_thres_raises(self):
        dm = self._make_data_manager()
        with pytest.raises(ValueError, match="'thres' must be specified"):
            ConceptorAnalyzer(
                hidden_state_data_manager=dm,
                skip_early_layers=0,
                skip_late_layers=0,
                aperture=0.1,
                center="none",
                lora=True,
                lora_method="optimal",
                rank=None,
                variance_thres=None,
                thres=None,
            )

    def test_skipping_all_layers_raises(self):
        dm = self._make_data_manager(num_layers=4)
        with pytest.raises(ValueError, match="Skipping all layers"):
            ConceptorAnalyzer(
                hidden_state_data_manager=dm,
                skip_early_layers=2,
                skip_late_layers=2,
                aperture=0.1,
                center="none",
                lora=False,
                lora_method=None,
                rank=None,
                variance_thres=None,
                thres=None,
            )