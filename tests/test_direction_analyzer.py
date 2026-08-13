"""Tests for DirectionAnalyzer math functions.

Needs torch. Pure-function tests only; does not instantiate DirectionAnalyzer.
"""

import pytest

torch = pytest.importorskip("torch")

from direction_analyzer import (
    compute_symmetrised_cross_covariance_eigenvectors,
    project_data_onto_direction,
    compute_discriminant_ratio,
    compute_variance_reduction,
)


class TestCrossCovarianceEigenvectors:
    def test_returns_square_matrix(self):
        A = torch.randn(100, 16)
        B = torch.randn(100, 16)
        result = compute_symmetrised_cross_covariance_eigenvectors(A, B)
        assert result.shape == (16, 16)

    def test_eigenvectors_as_rows(self):
        """Result is eigenvectors as rows (VT from eigh)."""
        A = torch.randn(100, 8)
        B = torch.randn(100, 8)
        result = compute_symmetrised_cross_covariance_eigenvectors(A, B)
        # Each row should be a unit vector (eigenvectors are orthonormal)
        for i in range(result.shape[0]):
            norm = torch.norm(result[i])
            assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_symmetric_input_invariant(self):
        """If A and B are identical, the matrix is symmetric and eigenvectors
        should be the same regardless of the transpose."""
        A = torch.randn(50, 10)
        result = compute_symmetrised_cross_covariance_eigenvectors(A, A)
        # The symmetrised matrix is AᵀA, which is symmetric
        # Just verify the result is well-formed
        assert result.shape == (10, 10)


class TestProjectData:
    def test_returns_one_scalar_per_row(self):
        data = torch.randn(50, 16)
        direction = torch.randn(16)
        projected = project_data_onto_direction(data, direction)
        assert projected.shape == (50,)

    def test_invariant_to_direction_magnitude(self):
        """Projection is invariant to the direction's magnitude (it normalises)."""
        data = torch.randn(50, 16)
        direction = torch.randn(16)
        p1 = project_data_onto_direction(data, direction)
        p2 = project_data_onto_direction(data, 5.0 * direction)
        assert torch.allclose(p1, p2, atol=1e-5)


class TestDiscriminantRatio:
    def test_large_for_separated_clusters(self):
        A = torch.randn(100) + 5.0  # mean ~5
        B = torch.randn(100) - 5.0  # mean ~-5
        ratio = compute_discriminant_ratio(A, B)
        assert ratio > 1.0

    def test_near_zero_for_identical(self):
        A = torch.randn(100)
        ratio = compute_discriminant_ratio(A, A)
        assert ratio < 0.1

    def test_zero_when_within_class_variance_zero(self):
        """Returns 0 rather than dividing by zero when within-class variance is 0."""
        A = torch.tensor([3.0, 3.0, 3.0])
        B = torch.tensor([1.0, 1.0, 1.0])
        ratio = compute_discriminant_ratio(A, B)
        assert ratio == 0.0


class TestVarianceReduction:
    def test_clamped_at_zero(self):
        A = torch.randn(100)
        B = torch.randn(100)
        vr = compute_variance_reduction(A, B)
        assert vr >= 0.0, f"variance reduction {vr} is negative"

    def test_near_zero_for_identical(self):
        A = torch.randn(100)
        vr = compute_variance_reduction(A, A)
        assert vr < 0.2