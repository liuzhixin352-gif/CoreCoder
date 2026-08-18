"""Tests for model metadata and capability catalog."""

import pytest

from corecoder.model_catalog import (
    ModelCatalog,
    ModelProfile,
)


def test_model_profile_preserves_metadata():
    profile = ModelProfile(
        name="coder-model",
        capabilities=frozenset(
            {
                "coding",
                "tool_calling",
            }
        ),
    )

    assert profile.name == "coder-model"
    assert profile.capabilities == frozenset(
        {
            "coding",
            "tool_calling",
        }
    )


def test_model_profile_supports_required_capabilities():
    profile = ModelProfile(
        name="coder-model",
        capabilities=frozenset(
            {
                "coding",
                "tool_calling",
            }
        ),
    )

    assert profile.supports(
        frozenset(
            {
                "coding",
            }
        )
    )

    assert profile.supports(
        frozenset(
            {
                "coding",
                "tool_calling",
            }
        )
    )


def test_model_profile_rejects_missing_capability():
    profile = ModelProfile(
        name="coder-model",
        capabilities=frozenset(
            {
                "coding",
            }
        ),
    )

    assert not profile.supports(
        frozenset(
            {
                "vision",
            }
        )
    )


def test_model_profile_uses_existing_pricing_table():
    profile = ModelProfile(
        name="gpt-4o-mini",
    )

    assert profile.pricing == (
        0.15,
        0.6,
    )


def test_model_profile_returns_none_for_unknown_pricing():
    profile = ModelProfile(
        name="custom-model",
    )

    assert profile.pricing is None


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
    ],
)
def test_model_profile_rejects_empty_name(
    name,
):
    with pytest.raises(
        ValueError,
        match="model name must be non-empty",
    ):
        ModelProfile(
            name=name,
        )


def test_model_profile_rejects_empty_capability():
    with pytest.raises(
        ValueError,
        match="capability names must be non-empty",
    ):
        ModelProfile(
            name="model",
            capabilities=frozenset(
                {
                    "coding",
                    "",
                }
            ),
        )


def test_model_catalog_returns_profile():
    profile = ModelProfile(
        name="coder-model",
        capabilities=frozenset(
            {
                "coding",
            }
        ),
    )

    catalog = ModelCatalog(
        [
            profile,
        ]
    )

    assert catalog.get(
        "coder-model"
    ) is profile


def test_model_catalog_checks_capabilities():
    catalog = ModelCatalog(
        [
            ModelProfile(
                name="coder-model",
                capabilities=frozenset(
                    {
                        "coding",
                        "tool_calling",
                    }
                ),
            )
        ]
    )

    assert catalog.supports(
        "coder-model",
        frozenset(
            {
                "tool_calling",
            }
        ),
    )


def test_model_catalog_rejects_duplicate_models():
    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        ModelCatalog(
            [
                ModelProfile(
                    name="same-model",
                ),
                ModelProfile(
                    name="same-model",
                ),
            ]
        )


def test_model_catalog_rejects_unknown_model():
    catalog = ModelCatalog([])

    with pytest.raises(
        ValueError,
        match="Unknown model profile",
    ):
        catalog.get(
            "missing-model"
        )