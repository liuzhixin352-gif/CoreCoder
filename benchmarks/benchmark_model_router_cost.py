"""Deterministic model-router cost-efficiency benchmark.

This benchmark does not call any model API and does not measure model quality.
It exercises the real CoreCoder CapabilityModelRouter against a frozen,
synthetic workload, using CoreCoder's built-in pricing table to estimate cost.

Comparison:
- baseline: route every call to the flagship model
- routed: use capability requirements + per-call remaining-cost budget to
  select the highest-priority eligible model

All workload token counts and capability assumptions are benchmark inputs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from corecoder.llm import estimate_cost_usd
from corecoder.model_catalog import ModelCatalog, ModelProfile
from corecoder.model_router import CapabilityModelRouter, RouteRequest


BASELINE_MODEL = "gpt-5.4"

CATALOG = ModelCatalog(
    [
        ModelProfile(
            name="gpt-5.4",
            capabilities=frozenset(
                {
                    "text",
                    "plan",
                    "review",
                    "summarize",
                    "code_edit",
                    "deep_reasoning",
                }
            ),
        ),
        ModelProfile(
            name="gpt-5.4-mini",
            capabilities=frozenset(
                {
                    "text",
                    "plan",
                    "review",
                    "summarize",
                }
            ),
        ),
        ModelProfile(
            name="gpt-5.4-nano",
            capabilities=frozenset(
                {
                    "text",
                    "summarize",
                }
            ),
        ),
    ]
)

# Expensive-to-cheap priority order is intentional. The router may only move
# to a cheaper model when capability and remaining-cost constraints require it.
ROUTER = CapabilityModelRouter(
    [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ],
    CATALOG,
)


@dataclass(frozen=True)
class WorkloadClass:
    name: str
    count: int
    prompt_tokens: int
    completion_tokens: int
    required_capabilities: frozenset[str]
    per_call_budget_usd: float


WORKLOAD = (
    WorkloadClass(
        name="planner",
        count=25,
        prompt_tokens=2_500,
        completion_tokens=500,
        required_capabilities=frozenset({"plan"}),
        per_call_budget_usd=0.010,
    ),
    WorkloadClass(
        name="coder",
        count=25,
        prompt_tokens=5_000,
        completion_tokens=1_200,
        required_capabilities=frozenset(
            {"code_edit", "deep_reasoning"}
        ),
        per_call_budget_usd=0.040,
    ),
    WorkloadClass(
        name="reviewer",
        count=25,
        prompt_tokens=3_000,
        completion_tokens=600,
        required_capabilities=frozenset({"review"}),
        per_call_budget_usd=0.010,
    ),
    WorkloadClass(
        name="summarizer",
        count=25,
        prompt_tokens=6_000,
        completion_tokens=500,
        required_capabilities=frozenset({"summarize"}),
        per_call_budget_usd=0.005,
    ),
)


def cost(
    model: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    value = estimate_cost_usd(
        model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    if value is None:
        raise RuntimeError(f"pricing unavailable for {model}")
    return value


def main() -> int:
    baseline_total = 0.0
    routed_total = 0.0
    total_calls = 0
    selected_models: Counter[str] = Counter()
    failures: list[str] = []

    print("Model Router Cost Efficiency Benchmark")
    print("======================================")
    print(
        "protocol: deterministic 100-call synthetic workload, "
        "real CapabilityModelRouter, built-in estimated pricing"
    )
    print(
        "note: no model API calls; cost is estimated, "
        "not provider billing or quality measurement"
    )
    print()

    for item in WORKLOAD:
        baseline_unit = cost(
            BASELINE_MODEL,
            prompt_tokens=item.prompt_tokens,
            completion_tokens=item.completion_tokens,
        )

        request = RouteRequest(
            required_capabilities=item.required_capabilities,
            remaining_cost_usd=item.per_call_budget_usd,
            estimated_prompt_tokens=item.prompt_tokens,
            estimated_completion_tokens=item.completion_tokens,
        )

        decision = ROUTER.select(request)
        routed_unit = cost(
            decision.model,
            prompt_tokens=item.prompt_tokens,
            completion_tokens=item.completion_tokens,
        )

        if not CATALOG.supports(
            decision.model,
            item.required_capabilities,
        ):
            failures.append(
                f"{item.name}: selected model does not satisfy capabilities"
            )

        if routed_unit > item.per_call_budget_usd:
            failures.append(
                f"{item.name}: routed estimated cost exceeds budget"
            )

        baseline_total += baseline_unit * item.count
        routed_total += routed_unit * item.count
        total_calls += item.count
        selected_models[decision.model] += item.count

        capabilities = ",".join(
            sorted(item.required_capabilities)
        )

        print(
            f"{item.name:<11} calls={item.count:<3} "
            f"tokens={item.prompt_tokens}+{item.completion_tokens:<4} "
            f"caps={capabilities:<24} "
            f"budget=${item.per_call_budget_usd:.3f}"
        )
        print(
            f"  baseline {BASELINE_MODEL:<13} "
            f"${baseline_unit:.6f}/call"
        )
        print(
            f"  routed   {decision.model:<13} "
            f"${routed_unit:.6f}/call"
        )

    print()
    print("Summary")
    print("-------")
    print(f"calls:              {total_calls}")
    print(f"baseline_model:     {BASELINE_MODEL}")
    print(f"baseline_est_cost:  ${baseline_total:.6f}")
    print(f"routed_est_cost:    ${routed_total:.6f}")

    savings = baseline_total - routed_total
    reduction = (
        savings / baseline_total * 100
        if baseline_total
        else 0.0
    )

    print(f"estimated_savings:  ${savings:.6f}")
    print(f"cost_reduction:     {reduction:.1f}%")
    print(
        "selected_models:    "
        + ", ".join(
            f"{model}={count}"
            for model, count in sorted(selected_models.items())
        )
    )
    print(
        "capability_contract: "
        f"{'PASS' if not failures else 'FAIL'}"
    )
    print(
        "budget_contract:     "
        f"{'PASS' if not failures else 'FAIL'}"
    )

    if failures:
        print()
        print("Failures")
        print("--------")
        for failure in failures:
            print(f"- {failure}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
