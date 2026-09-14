# pytest #14189

## Issue

pytest #14189 is a historical issue involving nested use of `caplog.filtering()`.

When the same logging filter is used in nested contexts, the inner context may
remove the filter too early, causing a later log record to be captured incorrectly.

## Evaluation Setup

Pre-fix commit:

`ced9022c0ca87ae2a0a604c68d2e1c462f8a5c6f`

The evaluation starts from a historical revision before the official fix.

## Before Repair

The original issue behavior was reproduced with an external regression test:

```python
def test_caplog_fixture(caplog):
    import logging

    def no_capture_filter(log_record):
        return False

    with caplog.filtering(no_capture_filter):
        logging.warning("Will not be captured")

        with caplog.filtering(no_capture_filter):
            logging.warning("Will also not be captured")

        logging.warning("Will incorrectly be captured")

    assert caplog.records == []
```

Before the CoreCoder repair, this test reported:

```text
1 failed
```

The final warning was incorrectly present in `caplog.records`.

Result: **FAIL**

## CoreCoder Repair

CoreCoder modified:

- `src/_pytest/logging.py`
- `testing/logging/test_fixture.py`
- `changelog/14189.bugfix.rst`

The repair prevents an already-installed logging filter from being removed
prematurely by an inner nested `caplog.filtering()` context.

## After Repair

The same external regression test was executed after the repair.

Observed result:

```text
1 passed in 0.02s
```

Result: **PASS**

## Targeted Validation

Relevant tests after the repair:

- `testing/logging/test_fixture.py`: **29 passed**
- `testing/logging/`: **86 passed**

The new regression test also failed when the repair was removed and passed again
when the repair was applied.

## Generic Validation Note

The generic post-repair validation attempted:

```bash
python -m pytest tests -q
```

However, the pytest repository uses `testing/` rather than `tests/` as its main
test directory.

The generic validation therefore returned a usage error. This was a
repository-specific validation configuration mismatch rather than a failure of
the issue repair itself.

## Conclusion

For pytest #14189, the original issue-specific behavior changed from:

**FAIL -> CoreCoder Repair -> PASS**

This provides issue-level evidence that the selected historical bug was repaired.