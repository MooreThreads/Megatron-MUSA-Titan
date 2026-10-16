"""Explicit function replacements with a readable audit log."""

import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def git_commit(source_file):
    """Read HEAD from an upstream current_path file's checkout, once per file."""
    return subprocess.check_output(
        ["git", "-C", str(Path(source_file).resolve().parent), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def replace_attr(owner, 
            name, 
            replacement, 
            *, 
            reason, 
            commits, 
            authored, 
            upstream_path, 
            current_path, 
            aliases=(), 
            commit_branch_verified=False):
    """Replace a callable and log its development baseline and runtime commits.

    authored records author, modified_date date, and upstream branch/commit/commit_date.
    Keep it fixed in current_path; commits describes the environment running the patch.
    """

    # 1. Verify provenance before changing anything. Never infer the development
    # baseline from runtime HEAD: the patch may have been written much earlier.
    if not reason.strip() or not commits or not all(commits.values()):
        raise ValueError("Patch reason and upstream commits are required")
    if not upstream_path.endswith("." + name):
        raise ValueError("upstream_path must be the full dotted path of the replaced function")
    if not current_path or Path(current_path).is_absolute() or ".." in Path(current_path).parts:
        raise ValueError("current_path must be a file path relative to this repository")
    if not all(authored.get(field, "").strip() for field in ("author", "modified_date")):
        raise ValueError("authored requires author and modified_date date")
    upstream = authored.get("upstream", {})
    if commit_branch_verified:
        if upstream.keys() != commits.keys():
            raise ValueError("authored.upstream and commits must list the same frameworks")
        for repo, revision in upstream.items():
            if not all(revision.get(field, "").strip() for field in ("branch", "commit", "commit_date")):
                raise ValueError(f"{repo}: branch, commit, and commit_date are required")

    # 2. Verify the original and every explicit import alias first, so a stale
    # alias cannot leave the function only partially replaced.
    original = getattr(owner, name)
    if not callable(original) or not callable(replacement):
        raise TypeError(f"{name}: expected callable upstream_path and replacement")

    aliases = tuple(aliases)
    for alias in aliases:
        if getattr(alias, name) is not original:
            raise ValueError(f"{name}: alias no longer references the original")

    # 3. Replace the function and its known aliases with the same callable.
    setattr(owner, name, replacement)
    for alias in aliases:
        setattr(alias, name, replacement)

    # 4. Print only after replacement succeeds. Separate the recorded baseline
    # from runtime versions; neither is a guarantee of compatibility.
    revisions = ", ".join(f"{repo}@{commit}" for repo, commit in commits.items())
    baseline = "\n".join(
        f"    {repo}: branch={revision['branch']}, commit={revision['commit']}, "
        f"commit_date={revision['commit_date']}"
        for repo, revision in upstream.items()
    )
    print(
        f"[MUSA patch] {upstream_path}\n"
        f"  reason: {reason}\n"
        f"  current_path: {current_path}\n"
        f"  author: {authored['author']}\n"
        f"  modified_date: {authored['modified_date']}\n"
        f"  authored against:\n{baseline}\n"
        f"  runtime: {revisions}",
        flush=True,
    )
    return original


def replace_import_source(
    source,
    original_source,
    replacement_source,
    *,
    reason,
    commits,
    authored,
    upstream_path,
    current_path,
    expected_count=1,
    commit_branch_verified=False,
):
    """Replace an exact import-related source fragment and log its provenance.

    This is the source-level counterpart to :func:`replace_attr`.  It is
    intended for narrowly scoped compatibility rewrites that keep upstream
    source files unchanged, such as making an optional import tolerate a
    dependency-version difference.

    ``original_source`` must occur exactly ``expected_count`` times.  Requiring
    an exact match prevents a source rewrite from silently doing nothing when
    the supported upstream version changes.
    """
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if not isinstance(original_source, str) or not original_source:
        raise ValueError("original_source must be a non-empty string")
    if not isinstance(replacement_source, str) or not replacement_source:
        raise ValueError("replacement_source must be a non-empty string")
    if original_source == replacement_source:
        raise ValueError("original_source and replacement_source must differ")
    if not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("expected_count must be a positive integer")

    # Keep the same provenance checks as replace_attr.  Source rewrites are
    # version-sensitive and must identify both their authored baseline and the
    # runtime checkout on which they are applied.
    if not reason.strip() or not commits or not all(commits.values()):
        raise ValueError("Patch reason and upstream commits are required")
    if not isinstance(upstream_path, str) or not upstream_path.strip():
        raise ValueError("upstream_path is required for a source rewrite")
    if not current_path or Path(current_path).is_absolute() or ".." in Path(current_path).parts:
        raise ValueError("current_path must be a file path relative to this repository")
    if not all(authored.get(field, "").strip() for field in ("author", "modified_date")):
        raise ValueError("authored requires author and modified_date date")
    upstream = authored.get("upstream", {})
    if commit_branch_verified:
        if upstream.keys() != commits.keys():
            raise ValueError("authored.upstream and commits must list the same frameworks")
        for repo, revision in upstream.items():
            if not all(revision.get(field, "").strip() for field in ("branch", "commit", "commit_date")):
                raise ValueError(f"{repo}: branch, commit, and commit_date are required")

    occurrence_count = source.count(original_source)
    if occurrence_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} occurrence(s) of the source fragment, "
            f"found {occurrence_count}"
        )

    rewritten = source.replace(original_source, replacement_source)
    revisions = ", ".join(f"{repo}@{commit}" for repo, commit in commits.items())
    baseline = "\n".join(
        f"    {repo}: branch={revision['branch']}, commit={revision['commit']}, "
        f"commit_date={revision['commit_date']}"
        for repo, revision in upstream.items()
    )
    print(
        f"[MUSA source patch] {upstream_path}\n"
        f"  reason: {reason}\n"
        f"  current_path: {current_path}\n"
        f"  author: {authored['author']}\n"
        f"  modified_date: {authored['modified_date']}\n"
        f"  authored against:\n{baseline}\n"
        f"  runtime: {revisions}",
        flush=True,
    )
    return rewritten
