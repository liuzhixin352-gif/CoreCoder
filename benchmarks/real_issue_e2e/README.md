# Real-World Issue Repair E2E Evaluation

This evaluation tests whether CoreCoder can repair real historical GitHub issues
in external open-source repositories.

Each case starts from a repository revision before the official fix. The original
issue behavior is reproduced before repair, CoreCoder performs the repair, and the
same issue-specific verification is executed afterward.

## Results

| Repository | Issue | Pre-fix Commit | Before | After |
|---|---|---|---|---|
| Flask | #5258 | `b7c1290528f907c9f41afcdfd33a2227c73e26d3` | FAIL | PASS |
| Requests | #6711 | `6360477c52303c9445b45fa8744b02d05a2f0905` | FAIL | PASS |
| pytest | #14189 | `ced9022c0ca87ae2a0a604c68d2e1c462f8a5c6f` | FAIL | PASS |

All three selected smoke-test cases achieved reproducible
**Fail → Repair → Pass** issue-level validation.

This is a small-scale real-world E2E smoke test rather than a statistical benchmark.
It demonstrates issue-level repair capability on these selected cases and should not
be interpreted as a general 100% repair success rate.

## Evaluation Method

For each case:

1. Start from a repository revision before the official fix.
2. Reproduce the original issue behavior.
3. Let CoreCoder diagnose and repair the issue.
4. Execute the same issue-specific verification after repair.
5. Run targeted repository tests where applicable.

A case is considered an issue-level repair success when the original failing behavior
changes from **FAIL to PASS** after the CoreCoder repair.

## Issue Repair vs. Full Workflow Completion

Issue-level repair correctness and full workflow completion are evaluated separately.

A repair can correctly solve the target issue while the generic validation guardrail
still blocks later workflow stages because of unrelated baseline failures or
repository-specific test configuration differences.

Therefore, these cases demonstrate **Issue Repair E2E** capability rather than
successful completion of the entire Commit → Push → PR → CI workflow.

## Cases

- [Flask #5258](flask_5258.md)
- [Requests #6711](requests_6711.md)
- [pytest #14189](pytest_14189.md)