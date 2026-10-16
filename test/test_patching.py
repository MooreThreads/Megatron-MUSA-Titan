"""CPU-only tests; load helpers without importing the MUSA runtime."""

import runpy
import unittest
from contextlib import redirect_stdout
from functools import lru_cache
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
helpers = runpy.run_path(str(ROOT / "musa_titan/utils/patching.py"))
replace_attr = helpers["replace_attr"]
replace_import_source = helpers["replace_import_source"]
git_commit = helpers["git_commit"]


class PatchingTest(unittest.TestCase):
    def authored(self, repos=("megatron",)):
        return {
            "author": "Test Developer", "modified_date": "2026-09-09",
            "upstream": {repo: {
                "branch": "test-branch", "commit": "d" * 40,
                "commit_date": "2026-09-08T12:00:00+08:00",
            } for repo in repos},
        }

    def apply_patch(self, owner, name, replacement, **kwargs):
        return replace_attr(
            owner, name, replacement,
            reason="Test function replacement",
            commits={"megatron": "a" * 40},
            authored=kwargs.pop("authored", self.authored()),
            upstream_path=f"megatron.test.{name}",
            current_path="test/test_patching.py",
            commit_branch_verified=True,
            **kwargs,
        )

    def apply_source_patch(self, source, original, replacement, **kwargs):
        return replace_import_source(
            source,
            original,
            replacement,
            reason="Test import compatibility rewrite",
            commits={"megatron": "a" * 40},
            authored=kwargs.pop("authored", self.authored()),
            upstream_path="megatron.test.import",
            current_path="test/test_patching.py",
            commit_branch_verified=True,
            **kwargs,
        )

    def test_replacement_updates_known_consumers(self):
        original = lambda value: value + 1
        replacement = lambda value: value * 2
        owner = SimpleNamespace(step=original)
        consumer = SimpleNamespace(step=original)
        self.assertIs(self.apply_patch(owner, "step", replacement, aliases=(consumer,)), original)
        self.assertEqual(owner.step(3), 6)
        self.assertIs(consumer.step, replacement)

    def test_source_replacement_requires_one_exact_fragment(self):
        source = "from old.path import value\n"
        rewritten = self.apply_source_patch(
            source,
            "from old.path import value\n",
            "from new.path import value\n",
        )
        self.assertEqual(rewritten, "from new.path import value\n")

        with self.assertRaises(ValueError):
            self.apply_source_patch(source, "from missing import value\n", "replacement\n")

    def test_stale_alias_fails_before_any_assignment(self):
        original = lambda: None
        owner = SimpleNamespace(step=original)
        good = SimpleNamespace(step=original)
        stale = SimpleNamespace(step=lambda: None)
        output = StringIO()
        with redirect_stdout(output), self.assertRaises(ValueError):
            self.apply_patch(owner, "step", lambda: 1, aliases=(good, stale))
        self.assertIs(owner.step, original)
        self.assertIs(good.step, original)
        self.assertEqual(output.getvalue(), "")

    def test_missing_or_noncallable_target_is_rejected(self):
        owner = SimpleNamespace(step=42)
        with self.assertRaises(AttributeError):
            self.apply_patch(owner, "missing", lambda: None)
        with self.assertRaises(TypeError):
            self.apply_patch(owner, "step", lambda: None)
        self.assertEqual(owner.step, 42)

    def test_outer_cache_preserves_method_and_avoids_32_entry_thrashing(self):
        calls = []

        class Embedding:
            @lru_cache(maxsize=32)
            def forward(self, length):
                calls.append(length)
                return length * 2

        original = Embedding.forward
        self.apply_patch(Embedding, "forward", lru_cache(maxsize=64)(original))
        layers = [Embedding() for _ in range(40)]
        for _ in range(2):
            self.assertEqual([layer.forward(128) for layer in layers], [256] * 40)
        self.assertEqual(len(calls), 40)
        self.assertIs(Embedding.forward.__wrapped__, original)

    def test_rl_log_includes_all_requested_provenance(self):
        owner = SimpleNamespace(step=lambda: None)
        output = StringIO()
        with redirect_stdout(output):
            replace_attr(
                owner, "step", lambda: 1,
                reason="Synchronize actor weights",
                commits={"megatron": "a" * 40, "slime": "b" * 40, "miles": "c" * 40},
                authored=self.authored(("megatron", "slime", "miles")),
                upstream_path="slime.actor.step",
                current_path="musa_titan/rl/slime_musa/actor.py",
                commit_branch_verified=True,
            )
        for value in ("slime.actor.step", "Synchronize actor weights",
                      "megatron@" + "a" * 40, "slime@" + "b" * 40,
                      "miles@" + "c" * 40, "musa_titan/rl/slime_musa/actor.py"):
            self.assertIn(value, output.getvalue())
        for value in ("author: Test Developer", "modified_date: 2026-09-09",
                      "branch=test-branch", "commit=" + "d" * 40,
                      "commit_date=2026-09-08T12:00:00+08:00", "runtime:"):
            self.assertIn(value, output.getvalue())

    def test_incomplete_authored_metadata_fails_before_replacement(self):
        invalid = [{}]
        for field in ("author", "modified_date"):
            record = self.authored()
            record[field] = " "
            invalid.append(record)
        for field in ("branch", "commit", "commit_date"):
            record = self.authored()
            del record["upstream"]["megatron"][field]
            invalid.append(record)
        invalid.append(self.authored(("slime",)))
        for record in invalid:
            with self.subTest(record=record):
                original = lambda: None
                owner = SimpleNamespace(step=original)
                output = StringIO()
                with redirect_stdout(output), self.assertRaises(ValueError):
                    self.apply_patch(owner, "step", lambda: 1, authored=record)
                self.assertIs(owner.step, original)
                self.assertEqual(output.getvalue(), "")

    def test_metadata_is_mandatory(self):
        owner = SimpleNamespace(step=lambda: None)
        with self.assertRaises(TypeError):
            replace_attr(owner, "step", lambda: 1)

    def test_git_commit_reads_checkout_head_and_caches_it(self):
        git_commit.cache_clear()
        with patch("subprocess.check_output", return_value="a" * 40 + "\n") as run:
            self.assertEqual(git_commit(__file__), "a" * 40)
            self.assertEqual(git_commit(__file__), "a" * 40)
            run.assert_called_once_with(
                ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                text=True,
            )
        git_commit.cache_clear()


if __name__ == "__main__":
    unittest.main()
