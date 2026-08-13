"""Tests for the CLI argument parser — stdlib only.

Guards the import with pytest.importorskip("torch") so the suite degrades
cleanly on a torch-less host.
"""

import pytest

torch = pytest.importorskip("torch")

from create_control_vectors import parse_args


class TestParseArgs:
    """Test the argument parser from create_control_vectors."""

    _BASE = [
        "--model", "/path/to/model",
        "--outpath", "output_prefix_",
        "--prompts", "data/prompt_stems.json",
        "--continuations", "data/continuations.json",
        "--writing-prompts-file", "data/writing_prompts.txt",
    ]

    def test_required_flags(self):
        """Omitting any required flag raises SystemExit."""
        required = ["--model", "--outpath", "--prompts", "--continuations", "--writing-prompts-file"]
        for flag in required:
            # Build argv without this flag
            argv = self._BASE[:]
            idx = argv.index(flag)
            argv.pop(idx)  # remove flag
            argv.pop(idx)  # remove its value
            with pytest.raises(SystemExit):
                parse_args(argv)

    def test_full_valid_parse(self):
        args = parse_args(self._BASE)
        assert args.model == "/path/to/model"
        assert args.outpath == "output_prefix_"
        assert args.prompts == "data/prompt_stems.json"
        assert args.continuations == "data/continuations.json"
        assert args.writing_prompts_file == "data/writing_prompts.txt"

    def test_defaults_match_expected(self):
        """Pin the intentional defaults — diverged from upstream."""
        args = parse_args(self._BASE)
        assert args.num_samples == 10000
        assert args.use_system_prompt is True
        assert args.skip_early_layers == 1
        assert args.skip_late_layers == 1
        assert args.gate == 0.3
        assert args.conceptors is False
        assert args.conceptor_aperture == 0.1
        assert args.center == "none"
        assert args.lora is False
        assert args.lora_method is None
        assert args.rank is None
        assert args.variance_thres is None
        assert args.thres is None
        assert args.batch_size == 1
        assert args.precision == "orig"
        assert args.attn == "none"

    def test_no_use_system_prompt(self):
        argv = self._BASE + ["--no-use-system-prompt"]
        args = parse_args(argv)
        assert args.use_system_prompt is False

    def test_invalid_precision(self):
        argv = self._BASE + ["--precision", "invalid"]
        with pytest.raises(SystemExit):
            parse_args(argv)

    def test_invalid_center(self):
        argv = self._BASE + ["--conceptors", "--center", "invalid"]
        with pytest.raises(SystemExit):
            parse_args(argv)

    def test_invalid_lora_method(self):
        argv = self._BASE + ["--conceptors", "--lora", "--lora-method", "invalid"]
        with pytest.raises(SystemExit):
            parse_args(argv)

    def test_invalid_attn(self):
        argv = self._BASE + ["--attn", "invalid"]
        with pytest.raises(SystemExit):
            parse_args(argv)

    def test_all_documented_attributes_present(self):
        """Every documented attribute is present on the Namespace."""
        args = parse_args(self._BASE)
        expected_attrs = [
            "model", "outpath", "prompts", "continuations", "writing_prompts_file",
            "num_samples", "use_system_prompt", "skip_early_layers", "skip_late_layers",
            "gate", "conceptors", "conceptor_aperture", "center", "lora",
            "lora_method", "rank", "variance_thres", "thres", "batch_size",
            "precision", "attn",
        ]
        for attr in expected_attrs:
            assert hasattr(args, attr), f"Missing attribute: {attr}"