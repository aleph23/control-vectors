"""Tests for corrected batched hidden-state extraction (Ticket 03).

Needs torch. Proves left-padding fix by comparing _generate_batch vs _generate.
"""

import pytest

torch = pytest.importorskip("torch")

from unittest.mock import MagicMock, PropertyMock


def _make_mock_model_handler(num_layers=4, hidden_dim=16):
    """Build a fake ModelHandler whose model.generate returns deterministic
    per-position hidden states encoding (row, absolute_position, layer)."""
    handler = MagicMock()

    # tokenizer
    handler.tokenizer.pad_token_id = 0
    handler.tokenizer.eos_token_id = 2

    # model device
    handler.model.device = torch.device("cpu")

    def fake_generate(tokens, **kwargs):
        batch_size = tokens.size(0)
        seq_len = tokens.size(1)
        # Build hidden states: one generation step, (num_layers+1) layers
        # each layer: (batch, seq_len, hidden_dim)
        # Value = float(row * 1000 + pos * 10 + layer) so we can detect
        # which position was read.
        hidden_states = []
        for layer in range(num_layers + 1):
            hs = torch.zeros(batch_size, seq_len, hidden_dim)
            for row in range(batch_size):
                for pos in range(seq_len):
                    hs[row, pos, :] = float(row * 1000 + pos * 10 + layer)
            hidden_states.append(hs)
        result = MagicMock()
        result.hidden_states = [tuple(hidden_states)]
        return result

    handler.model.generate = fake_generate
    return handler


class TestBatchedExtraction:
    def test_batch_equals_single(self):
        """_generate_batch(batch) must equal [_generate(t) for t in batch]."""
        from hidden_state_data_manager import HiddenStateDataManager

        handler = _make_mock_model_handler(num_layers=4, hidden_dim=16)
        # Create a HiddenStateDataManager without running __init__
        dm = object.__new__(HiddenStateDataManager)
        dm.model_handler = handler
        dm.batch_size = 4

        # Three token tensors of different lengths
        tokens = [
            torch.tensor([[1, 2, 3, 4, 5]]),       # len 5
            torch.tensor([[1, 2, 3]]),              # len 3
            torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), # len 7
        ]

        batch_result = dm._generate_batch(tokens)
        single_results = [dm._generate(t) for t in tokens]

        assert len(batch_result) == len(single_results)
        for i, (batch_deltas, single_deltas) in enumerate(zip(batch_result, single_results)):
            assert len(batch_deltas) == len(single_deltas), f"Sample {i}: delta count mismatch"
            for j, (bd, sd) in enumerate(zip(batch_deltas, single_deltas)):
                assert torch.allclose(bd, sd), (
                    f"Sample {i}, delta {j}: batch vs single mismatch\n"
                    f"batch: {bd[:3]}\nsingle: {sd[:3]}"
                )

    def test_return_structure(self):
        """Each returned sample is a list of num_layers deltas."""
        from hidden_state_data_manager import HiddenStateDataManager

        handler = _make_mock_model_handler(num_layers=4, hidden_dim=16)
        dm = object.__new__(HiddenStateDataManager)
        dm.model_handler = handler
        dm.batch_size = 2

        tokens = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 2, 3, 4]]),
        ]

        result = dm._generate_batch(tokens)
        assert len(result) == 2
        for sample in result:
            assert isinstance(sample, list)
            assert len(sample) == 4  # num_layers deltas = 4 (5 hidden states - 1)

    def test_short_sequence_reads_real_final_token(self):
        """A short sequence in a batch must extract the real final token, not a pad slot."""
        from hidden_state_data_manager import HiddenStateDataManager

        num_layers = 3
        handler = _make_mock_model_handler(num_layers=num_layers, hidden_dim=8)
        dm = object.__new__(HiddenStateDataManager)
        dm.model_handler = handler
        dm.batch_size = 2

        # Short sequence (length 2) and long sequence (length 5)
        tokens = [
            torch.tensor([[1, 2]]),
            torch.tensor([[1, 2, 3, 4, 5]]),
        ]

        # Single-sample results for comparison
        short_single = dm._generate(tokens[0])

        # The batch result for the short sequence should match the single result
        batch_result = dm._generate_batch(tokens)
        short_from_batch = batch_result[0]

        for j, (bd, sd) in enumerate(zip(short_from_batch, short_single)):
            assert torch.allclose(bd, sd), (
                f"Short sequence delta {j}: batch result differs from single.\n"
                f"batch: {bd[:3]}\nsingle: {sd[:3]}"
            )