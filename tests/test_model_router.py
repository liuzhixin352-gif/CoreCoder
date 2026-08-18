"""Tests for model-routing primitives."""

import pytest

from corecoder.model_catalog import (
    ModelCatalog,
    ModelProfile,
)
from corecoder.model_router import (
    CapabilityModelRouter,
    NoEligibleModelError,
    RouteRequest,
    StaticModelRouter,
    NoRoleRouteError,
    RoleModelRouter,
)

def test_static_router_selects_primary_model():
    router = StaticModelRouter(
        [
            "gpt-5.5",
            "gpt-5.4-mini",
        ]
    )

    decision = router.select(
        RouteRequest(),
    )

    assert decision.model == "gpt-5.5"
    assert decision.fallback_index == 0
    assert decision.is_fallback is False


def test_static_router_selects_next_fallback():
    router = StaticModelRouter(
        [
            "gpt-5.5",
            "gpt-5.4-mini",
            "gpt-4.1-mini",
        ]
    )

    decision = router.fallback(
        RouteRequest(),
        "gpt-5.5",
        RuntimeError("temporary failure"),
    )

    assert decision is not None
    assert decision.model == "gpt-5.4-mini"
    assert decision.fallback_index == 1
    assert decision.is_fallback is True


def test_static_router_walks_fallback_chain():
    router = StaticModelRouter(
        [
            "primary",
            "fallback-1",
            "fallback-2",
        ]
    )

    first = router.fallback(
        RouteRequest(),
        "primary",
        RuntimeError("failure"),
    )
    assert first is not None

    second = router.fallback(
        RouteRequest(),
        first.model,
        RuntimeError("failure"),
    )

    assert second is not None
    assert second.model == "fallback-2"
    assert second.fallback_index == 2


def test_static_router_returns_none_after_last_model():
    router = StaticModelRouter(
        [
            "primary",
            "fallback",
        ]
    )

    decision = router.fallback(
        RouteRequest(),
        "fallback",
        RuntimeError("failure"),
    )

    assert decision is None


def test_static_router_rejects_unknown_current_model():
    router = StaticModelRouter(
        [
            "primary",
            "fallback",
        ]
    )

    with pytest.raises(
        ValueError,
        match="current_model is not configured",
    ):
        router.fallback(
            RouteRequest(),
            "unknown",
            RuntimeError("failure"),
        )


def test_static_router_rejects_empty_model_list():
    with pytest.raises(
        ValueError,
        match="at least one model",
    ):
        StaticModelRouter([])


@pytest.mark.parametrize(
    "models",
    [
        ["primary", ""],
        ["primary", "   "],
    ],
)
def test_static_router_rejects_empty_model_names(
    models,
):
    with pytest.raises(
        ValueError,
        match="model names must be non-empty",
    ):
        StaticModelRouter(models)


def test_static_router_rejects_duplicate_models():
    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        StaticModelRouter(
            [
                "primary",
                "primary",
            ]
        )


def test_route_request_preserves_role():
    request = RouteRequest(
        role="reviewer",
    )

    assert request.role == "reviewer"

def test_route_request_preserves_required_capabilities():
    request = RouteRequest(
        role="coder",
        required_capabilities=frozenset(
            {
                "coding",
                "tool_calling",
            }
        ),
    )

    assert request.required_capabilities == frozenset(
        {
            "coding",
            "tool_calling",
        }
    )

def test_capability_router_selects_first_eligible_model():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="cheap",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
            ModelProfile(
                name="primary",
                capabilities=frozenset(
                    {
                        "coding",
                        "tool_calling",
                    }
                ),
            ),
            ModelProfile(
                name="fallback",
                capabilities=frozenset(
                    {
                        "coding",
                        "tool_calling",
                    }
                ),
            ),
        ]
    )

    router = CapabilityModelRouter(
        [
            "cheap",
            "primary",
            "fallback",
        ],
        catalog,
    )

    decision = router.select(
        RouteRequest(
            required_capabilities=frozenset(
                {
                    "coding",
                    "tool_calling",
                }
            ),
        )
    )

    assert decision.model == "primary"
    assert decision.fallback_index == 0
    assert decision.is_fallback is False

def test_capability_router_uses_first_model_without_requirements():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="first",
            ),
            ModelProfile(
                name="second",
            ),
        ]
    )

    router = CapabilityModelRouter(
        [
            "first",
            "second",
        ],
        catalog,
    )

    decision = router.select(
        RouteRequest(),
    )

    assert decision.model == "first"
    assert decision.fallback_index == 0

def test_capability_router_rejects_request_without_eligible_model():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="text-only",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            )
        ]
    )

    router = CapabilityModelRouter(
        [
            "text-only",
        ],
        catalog,
    )

    with pytest.raises(
        NoEligibleModelError,
        match="tool_calling",
    ):
        router.select(
            RouteRequest(
                required_capabilities=frozenset(
                    {
                        "tool_calling",
                    }
                ),
            )
        )

def test_capability_router_fallback_skips_incompatible_models():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="primary",
                capabilities=frozenset(
                    {
                        "coding",
                        "tool_calling",
                    }
                ),
            ),
            ModelProfile(
                name="incompatible",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
            ModelProfile(
                name="fallback",
                capabilities=frozenset(
                    {
                        "coding",
                        "tool_calling",
                    }
                ),
            ),
        ]
    )

    router = CapabilityModelRouter(
        [
            "primary",
            "incompatible",
            "fallback",
        ],
        catalog,
    )

    request = RouteRequest(
        required_capabilities=frozenset(
            {
                "coding",
                "tool_calling",
            }
        ),
    )

    decision = router.fallback(
        request,
        "primary",
        RuntimeError(
            "temporary failure"
        ),
    )

    assert decision is not None
    assert decision.model == "fallback"
    assert decision.fallback_index == 1
    assert decision.is_fallback is True

def test_capability_router_returns_none_after_last_eligible_model():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="primary",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
            ModelProfile(
                name="other",
                capabilities=frozenset(
                    {
                        "vision",
                    }
                ),
            ),
        ]
    )

    router = CapabilityModelRouter(
        [
            "primary",
            "other",
        ],
        catalog,
    )

    decision = router.fallback(
        RouteRequest(
            required_capabilities=frozenset(
                {
                    "coding",
                }
            ),
        ),
        "primary",
        RuntimeError(
            "temporary failure"
        ),
    )

    assert decision is None

def test_capability_router_rejects_unknown_current_model():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="primary",
            )
        ]
    )

    router = CapabilityModelRouter(
        [
            "primary",
        ],
        catalog,
    )

    with pytest.raises(
        ValueError,
        match="current_model is not configured",
    ):
        router.fallback(
            RouteRequest(),
            "unknown",
            RuntimeError(
                "failure"
            ),
        )

def test_capability_router_rejects_incompatible_current_model():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="coding-only",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
            ModelProfile(
                name="tool-model",
                capabilities=frozenset(
                    {
                        "tool_calling",
                    }
                ),
            ),
        ]
    )

    router = CapabilityModelRouter(
        [
            "coding-only",
            "tool-model",
        ],
        catalog,
    )

    with pytest.raises(
        ValueError,
        match="does not satisfy",
    ):
        router.fallback(
            RouteRequest(
                required_capabilities=frozenset(
                    {
                        "tool_calling",
                    }
                ),
            ),
            "coding-only",
            RuntimeError(
                "failure"
            ),
        )

def test_capability_router_rejects_model_missing_from_catalog():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="known",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="Unknown model profile",
    ):
        CapabilityModelRouter(
            [
                "known",
                "missing",
            ],
            catalog,
        )

def test_capability_router_skips_model_that_exceeds_remaining_cost():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="gpt-4o",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
            ModelProfile(
                name="gpt-4o-mini",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
        ]
    )

    router = CapabilityModelRouter(
        [
            "gpt-4o",
            "gpt-4o-mini",
        ],
        catalog,
    )

    decision = router.select(
        RouteRequest(
            required_capabilities=frozenset(
                {
                    "coding",
                }
            ),
            remaining_cost_usd=0.001,
            estimated_prompt_tokens=1000,
            estimated_completion_tokens=500,
        )
    )

    assert decision.model == "gpt-4o-mini"
    assert decision.fallback_index == 0

def test_capability_router_preserves_priority_when_primary_fits_budget():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="gpt-4o",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
            ModelProfile(
                name="gpt-4o-mini",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
        ]
    )

    router = CapabilityModelRouter(
        [
            "gpt-4o",
            "gpt-4o-mini",
        ],
        catalog,
    )

    decision = router.select(
        RouteRequest(
            required_capabilities=frozenset(
                {
                    "coding",
                }
            ),
            remaining_cost_usd=0.01,
            estimated_prompt_tokens=1000,
            estimated_completion_tokens=500,
        )
    )

    assert decision.model == "gpt-4o"

def test_capability_router_skips_unpriced_model_with_cost_constraint():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="custom-unpriced-model",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
            ModelProfile(
                name="gpt-4o-mini",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
        ]
    )

    router = CapabilityModelRouter(
        [
            "custom-unpriced-model",
            "gpt-4o-mini",
        ],
        catalog,
    )

    decision = router.select(
        RouteRequest(
            required_capabilities=frozenset(
                {
                    "coding",
                }
            ),
            remaining_cost_usd=0.001,
            estimated_prompt_tokens=1000,
            estimated_completion_tokens=500,
        )
    )

    assert decision.model == "gpt-4o-mini"

def test_capability_router_rejects_when_no_model_fits_remaining_cost():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="gpt-4o",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
            ModelProfile(
                name="gpt-4o-mini",
                capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
            ),
        ]
    )

    router = CapabilityModelRouter(
        [
            "gpt-4o",
            "gpt-4o-mini",
        ],
        catalog,
    )

    with pytest.raises(
        NoEligibleModelError,
    ):
        router.select(
            RouteRequest(
                required_capabilities=frozenset(
                    {
                        "coding",
                    }
                ),
                remaining_cost_usd=0.0001,
                estimated_prompt_tokens=1000,
                estimated_completion_tokens=500,
            )
        )

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("remaining_cost_usd", -0.01),
        ("estimated_prompt_tokens", -1),
        ("estimated_completion_tokens", -1),
    ],
)
def test_route_request_rejects_negative_budget_inputs(
    field,
    value,
):
    kwargs = {
        field: value,
    }

    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        RouteRequest(
            **kwargs,
        )

def test_role_router_selects_different_models_for_roles():
    router = RoleModelRouter(
        {
            "planner": StaticModelRouter(
                [
                    "planner-primary",
                    "planner-fallback",
                ]
            ),
            "coder": StaticModelRouter(
                [
                    "coder-primary",
                    "coder-fallback",
                ]
            ),
            "reviewer": StaticModelRouter(
                [
                    "reviewer-primary",
                    "reviewer-fallback",
                ]
            ),
        }
    )

    assert router.select(
        RouteRequest(
            role="planner",
        )
    ).model == "planner-primary"

    assert router.select(
        RouteRequest(
            role="coder",
        )
    ).model == "coder-primary"

    assert router.select(
        RouteRequest(
            role="reviewer",
        )
    ).model == "reviewer-primary"

def test_role_router_uses_role_specific_fallback_chain():
    router = RoleModelRouter(
        {
            "coder": StaticModelRouter(
                [
                    "coder-primary",
                    "coder-fallback",
                ]
            ),
            "reviewer": StaticModelRouter(
                [
                    "reviewer-primary",
                    "reviewer-fallback",
                ]
            ),
        }
    )

    decision = router.fallback(
        RouteRequest(
            role="reviewer",
        ),
        "reviewer-primary",
        RuntimeError(
            "temporary failure"
        ),
    )

    assert decision is not None
    assert decision.model == "reviewer-fallback"
    assert decision.fallback_index == 1

def test_role_router_rejects_unconfigured_role():
    router = RoleModelRouter(
        {
            "coder": StaticModelRouter(
                [
                    "coder-model",
                ]
            )
        }
    )

    with pytest.raises(
        NoRoleRouteError,
        match="reviewer",
    ):
        router.select(
            RouteRequest(
                role="reviewer",
            )
        )

def test_role_router_uses_default_router_for_unconfigured_role():
    router = RoleModelRouter(
        {},
        default_router=StaticModelRouter(
            [
                "default-model",
            ]
        ),
    )

    decision = router.select(
        RouteRequest(
            role="custom-role",
        )
    )

    assert decision.model == "default-model"

def test_role_router_uses_default_router_without_role():
    router = RoleModelRouter(
        {
            "coder": StaticModelRouter(
                [
                    "coder-model",
                ]
            )
        },
        default_router=StaticModelRouter(
            [
                "default-model",
            ]
        ),
    )

    decision = router.select(
        RouteRequest(),
    )

    assert decision.model == "default-model"