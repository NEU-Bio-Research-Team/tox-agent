"""What the Settings page may offer (I13).

The page carried its own list. It offered OpenAI, Anthropic and Google Gemini
with an empty default base URL and the field described as optional; the probe
then refused every one of them with `capability probing requires an explicit
base_url`. A user could fill the form in correctly and still never get a
working connection, because two of the three named providers speak a wire
format this control plane has no adapter for.

The rule the fix established: a provider may be listed only if this code can
talk to it, and a listed provider either names a base URL that works or says
the deployment must supply one. These are the assertions of that rule — I13
was closed with a source change and a UI change and nothing checking either,
which the regression-guard registry is what surfaced.
"""
from __future__ import annotations

import pytest

from toxagent.connections import providers
from toxagent.domain.runtime import AuthMode


def test_every_offered_provider_can_be_reached_without_the_user_guessing():
    """The exact defect: listed, form completable, probe impossible."""
    for spec in providers.SUPPORTED:
        resolved = spec.resolve_base_url(None)
        assert resolved is not None or spec.default_base_url is None, spec.id
        if resolved is None:
            # Only the self-hosted entry may need one, and it has to say so
            # rather than leave the field looking optional.
            assert "required" in spec.note.lower(), spec.id
        else:
            assert resolved.startswith("https://"), spec.id


def test_a_supplied_base_url_wins_and_blank_input_is_not_a_base_url():
    spec = providers.get("openai")
    assert spec.resolve_base_url("https://gw.example.com/v1") == "https://gw.example.com/v1"
    assert spec.resolve_base_url("   ") == spec.default_base_url
    assert spec.resolve_base_url(None) == spec.default_base_url


def test_nothing_is_offered_that_this_control_plane_cannot_speak():
    """One adapter exists, so one protocol may appear in the catalogue."""
    protocols = {spec.protocol for spec in providers.SUPPORTED}
    assert protocols == {providers.Protocol.OPENAI_CHAT_COMPLETIONS}


@pytest.mark.parametrize("provider_id", ["anthropic", "gemini"])
def test_the_two_that_were_offered_falsely_are_refused_with_the_reason(provider_id):
    """Refused, not silently absent: the user asked for a reason and gets one,
    including the way to use that model through a gateway."""
    with pytest.raises(providers.UnsupportedProvider) as excinfo:
        providers.get(provider_id)
    assert excinfo.value.provider_id == provider_id
    assert "OpenAI-compatible" in excinfo.value.reason
    assert provider_id not in {spec.id for spec in providers.SUPPORTED}


def test_an_unknown_provider_names_what_is_available():
    with pytest.raises(providers.UnsupportedProvider) as excinfo:
        providers.get("not-a-provider")
    assert "openai" in excinfo.value.reason


def test_the_catalogue_tells_the_ui_when_it_must_ask_for_a_base_url():
    """`base_url_required` is what stops the form describing a mandatory
    field as optional — the half of the defect the user actually saw."""
    catalogue = {entry["provider_id"]: entry for entry in providers.catalogue()}
    assert catalogue["openai"]["base_url_required"] is False
    assert catalogue["openai"]["default_base_url"] == "https://api.openai.com/v1"
    assert catalogue["openai_compatible"]["base_url_required"] is True
    assert catalogue["openai_compatible"]["default_base_url"] is None
    assert set(catalogue) == {spec.id for spec in providers.SUPPORTED}
    for entry in catalogue.values():
        assert entry["auth_modes"], entry["provider_id"]
        assert all(mode in {m.value for m in AuthMode} for mode in entry["auth_modes"])
