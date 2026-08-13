"""Shared fixtures for the test suite."""

import json
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with minimal valid data files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # prompt_stems.json
    stems = data_dir / "prompt_stems.json"
    stems.write_text(json.dumps({
        "pre": ["You are", "Act as"],
        "post": ["an author", "a writer"]
    }))

    # A continuations JSON with 2 classes
    continuations_dir = data_dir / "continuations"
    continuations_dir.mkdir()
    continuations = continuations_dir / "test_continuations.json"
    continuations.write_text(json.dumps({
        "classes": ["class_a", "class_b"],
        "data": [
            ["who writes simply", "who writes complexly"],
            ["who is direct", "who is indirect"]
        ]
    }))

    # writing_prompts.txt
    prompts = data_dir / "writing_prompts.txt"
    prompts.write_text("Write a story\nCompose a poem\nDraft an essay\n")

    return data_dir