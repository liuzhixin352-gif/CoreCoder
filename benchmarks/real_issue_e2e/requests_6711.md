# Requests #6711

## Issue

Requests #6711 is a historical issue involving preservation of a double leading
slash in request paths, which can affect services such as AWS S3 presigned URLs.

The affected behavior normalized a path beginning with `//` into `/`, which could
invalidate request signatures.

## Evaluation Setup

Pre-fix commit:

`6360477c52303c9445b45fa8744b02d05a2f0905`

The evaluation starts from a historical revision before the official fix.

## Before Repair

The original issue behavior was reproduced with:

```bash
python -c "import requests; a=requests.adapters.HTTPAdapter(); p=requests.Request(method='GET', url='http://127.0.0.1:10000//v:h').prepare(); print(a.request_url(p, {}))"
```

Observed result:

```text
/v:h
```

Expected behavior:

```text
//v:h
```

Result: **FAIL**

## CoreCoder Repair

CoreCoder modified:

- `src/requests/adapters.py`
- `tests/test_adapters.py`

The repair removes the normalization that collapsed a double leading slash into
a single slash.

## After Repair

The same issue-specific reproduction command was executed after the repair.

Observed result:

```text
//v:h
```

Result: **PASS**

## Targeted Validation

Relevant tests after the repair:

- `tests/test_adapters.py`: **1 passed**
- `tests/test_utils.py`: **219 passed, 1 skipped**

## Full-Suite Validation Note

The broader validation reported:

- **411 passed**
- **2 failed**
- **201 errors**

The remaining failures and errors were also present independently of the CoreCoder
repair and were attributable to pre-existing fixture, environment, and
network-related test issues.

The generic validation guardrail therefore blocked later workflow stages even
though the target Requests #6711 behavior had been repaired successfully.

## Conclusion

For Requests #6711, the original issue-specific behavior changed from:

**FAIL -> CoreCoder Repair -> PASS**

This provides issue-level evidence that the selected historical bug was repaired.