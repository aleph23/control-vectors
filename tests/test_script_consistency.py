"""Flag consistency between the CLI parser, shell script, and README.

stdlib only — uses ast and regex, no imports of any repo module.
"""

import ast
import os
import re
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _extract_flag_names_from_parser(py_path):
    """Walk the AST of create_control_vectors.py and collect every --flag string
    literal passed to parser.add_argument."""
    with open(py_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match parser.add_argument(...)
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        # The first positional arg is the flag name
        if node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                if first.value.startswith("--"):
                    flags.add(first.value)
    return flags


def _extract_flag_names_from_shell(sh_path):
    """Extract every --flag token from the shell script."""
    if not os.path.exists(sh_path):
        return set()
    with open(sh_path, "r", encoding="utf-8") as f:
        text = f.read()
    return set(re.findall(r'--[a-z][a-z0-9-]*', text))


def _extract_flag_names_from_readme(md_path):
    """Extract every --flag token from README.md."""
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    return set(re.findall(r'--[a-z][a-z0-9-]*', text))


def _extract_continuation_paths_from_shell(sh_path):
    """Extract the continuation file paths from the shell script."""
    if not os.path.exists(sh_path):
        return set()
    with open(sh_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Match "$DATA/path/to/file.json"
    paths = re.findall(r'\$DATA/([a-z_/]+\.json)', text)
    return set(os.path.join(PROJECT_ROOT, "data", p) for p in paths)


# Upstream flag names that must NOT appear anywhere
_RETIRED_FLAGS = frozenset({
    "--model_id", "--output_path", "--prompt_stems_file",
    "--continuations_file", "--writing_prompts_file",
    "--num_prompt_samples", "--num_samples_per_class",
    "--use_separate_system_message", "--skip_begin_layers",
    "--skip_end_layers", "--discriminant_ratio_tolerance",
    "--quantization", "--use_bfloat16",
})


class TestFlagConsistency:
    def test_shell_flags_are_subset_of_parser(self):
        parser_flags = _extract_flag_names_from_parser(
            os.path.join(PROJECT_ROOT, "create_control_vectors.py"))
        shell_flags = _extract_flag_names_from_shell(
            os.path.join(PROJECT_ROOT, "create_all_control_vectors.sh"))

        unknown = shell_flags - parser_flags
        assert not unknown, f"Shell script uses flags not in the parser: {unknown}"

    def test_readme_flags_are_subset_of_parser(self):
        parser_flags = _extract_flag_names_from_parser(
            os.path.join(PROJECT_ROOT, "create_control_vectors.py"))
        readme_flags = _extract_flag_names_from_readme(
            os.path.join(PROJECT_ROOT, "README.md"))

        unknown = readme_flags - parser_flags
        assert not unknown, f"README uses flags not in the parser: {unknown}"

    def test_no_retired_flags_in_shell(self):
        shell_flags = _extract_flag_names_from_shell(
            os.path.join(PROJECT_ROOT, "create_all_control_vectors.sh"))
        found = shell_flags & _RETIRED_FLAGS
        assert not found, f"Shell script still uses retired upstream flags: {found}"

    def test_no_retired_flags_in_readme(self):
        readme_flags = _extract_flag_names_from_readme(
            os.path.join(PROJECT_ROOT, "README.md"))
        found = readme_flags & _RETIRED_FLAGS
        assert not found, f"README still uses retired upstream flags: {found}"

    def test_shell_continuation_paths_exist(self):
        paths = _extract_continuation_paths_from_shell(
            os.path.join(PROJECT_ROOT, "create_all_control_vectors.sh"))
        for path in paths:
            assert os.path.exists(path), f"Continuation file referenced in shell script does not exist: {path}"

    def test_no_nonexistent_references_in_shell(self):
        """Shell script must not reference the four removed continuation files."""
        sh_path = os.path.join(PROJECT_ROOT, "create_all_control_vectors.sh")
        with open(sh_path, "r", encoding="utf-8") as f:
            text = f.read()
        removed = [
            "optimism_vs_nihilism",
            "extreme_vs_tame",
            "vanilla_vs_deviant",
            "fantasy_vs_realistic",
        ]
        for name in removed:
            assert name not in text, f"Shell script still references removed file: {name}"