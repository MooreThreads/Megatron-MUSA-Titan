# Writing maintainable patches

## Utility imports

The package is now `musa_titan`. External Megatron and TransformerEngine
integrations must also update their imports:

```python
from musa_titan.utils.mem_utils import MemMonitor
from musa_titan.utils.utils import get_iteration_id
from musa_titan.utils.patching import git_commit, replace_attr
```

Memory helpers are implemented directly in `musa_titan.utils.mem_utils`.
The debug utility module still shares the implementation and mutable state in
`musa_titan.megatron.utils`. The `utils` package does not import these modules
eagerly, avoiding circular imports while initializing Megatron.

## Function replacements

Keep each change close to the code it affects:

- Route all new or migrated Megatron, slime, and Miles function replacements
  through `replace_attr()`, including wrappers. Legacy direct assignments still need migration.
- Record the upstream source commit beside copied code.
- Explain the behavioral difference in one sentence.
- Prefer wrapping the original for logging, caching, or input/output adaptation.
- For copied functions, mark the changed blocks and retain upstream copyright notices.

Use ordinary Python imports and the small `replace_attr` helper:

```python
from functools import wraps
from musa_titan.utils.patching import git_commit, replace_attr
from megatron.training import training

original = training.train_step

@wraps(original)
def train_step(*args, **kwargs):
    with my_profiler():
        return original(*args, **kwargs)

replace_attr(
    training, "train_step", train_step,
    reason="Add profiling around the training step",
    commits={"megatron": git_commit(training.__file__)},
    authored={
        "author": "Your Name",
        "modified_date": "YYYY-MM-DD",
        "upstream": {"megatron": {
            "branch": "development branch",
            "commit": "full upstream commit SHA",
            "commit_date": "ISO 8601 committer date with timezone",
        }},
    },
    upstream_path="megatron.training.training.train_step",
    current_path="musa_titan/pretrain/training.py",
    commit_branch_verified=True,
)
```

For a function imported elsewhere with `from ... import ...`, list the known
alias modules explicitly with `aliases=(consumer_module,)`. They must still
reference the original function. Apply patches before other consumers import
the function; existing unlisted references are not updated.

Every call requires a reason, runtime commits, `authored`, the full dotted target path,
and the patch file's path relative to this repository. The helper prints these
fields after a successful replacement. It checks target existence, callability,
and alias identity, but does not verify upstream compatibility.

Keep `authored` as a literal development record, not a runtime Git lookup.
`author` identifies the patch developer, not the upstream commit author;
`modified_date` is the date of the last patch logic change, not the execution date.
For each framework, record the development branch, full commit SHA, and
committer date (with timezone). In that upstream checkout, obtain them with:

```bash
git branch --show-current
git show -s --format='%H %cI' HEAD
```

For detached HEAD, record `detached HEAD` as the branch. Update this record
when changing or rebasing the patch logic. `authored.upstream` and `commits`
must list the same frameworks when `commit_branch_verified=True`. This flag
checks metadata completeness, not actual compatibility. Required fields are checked for presence, not
SHA/date format. The two legacy examples explicitly use `unknown` until their
history is verified; new patches should provide real values.

For RL, include Megatron and every participating RL framework in `commits`:

```python
commits={
    "megatron": git_commit(megatron_source.__file__),
    "slime": git_commit(slime_source.__file__),
    # Include Miles too when it participates in this run.
}
```

Use an imported source module from each checkout. `git_commit()` reads the
current checkout's HEAD once per source file; it does not read a recommended
commit from documentation. For installations without Git metadata, pass the
actual build commit explicitly. Missing Git metadata raises an error.

The logged commit identifies the current checkout, not the version the patch
was originally written against, and does not describe uncommitted edits.
Keep the original source reference beside copied code. Logs are emitted in
each process that applies the patch.

When upgrading upstream, compare the recorded source commit with the new
implementation and run a behavior test for the change. A matching signature
alone does not establish compatibility. For large internal changes, a small
commit on an upstream fork may be easier to maintain than a copied function.

The example patches currently cite `fdfcef87` only as the repository baseline
from the README, not as a verified source or tested revision for each function.
Replace that note with a permalink to the actual source commit after review.
