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
    """Every listed provider's wire format has a probe that speaks it.

    The rule I13 established, restated now that a second protocol exists: a
    provider may be listed only if this code can talk to it. Anthropic was
    refused until the Messages-API probe was written; Gemini still is.
    """
    from toxagent.connections.probe import ProtocolProbe

    spoken = set(ProtocolProbe()._probes)
    for spec in providers.SUPPORTED:
        assert spec.protocol in spoken, spec.id


def test_gemini_is_offered_as_google_and_the_old_name_says_so():
    """Gemini was refused for lack of an adapter. It is now offered under
    OpenCode's own provider id, and the name a user might type points there."""
    spec = providers.get("google")
    assert spec.runtime_provider_id == "google"
    assert spec.default_base_url.endswith("/v1beta/openai")
    with pytest.raises(providers.UnsupportedProvider) as excinfo:
        providers.get("gemini")
    assert "'google'" in excinfo.value.reason


@pytest.mark.parametrize("provider_id", ["claude_pro", "claude_code"])
def test_a_claude_subscription_is_refused_and_says_to_use_a_key(provider_id):
    with pytest.raises(providers.UnsupportedProvider) as excinfo:
        providers.get(provider_id)
    assert "API key" in excinfo.value.reason


def test_every_runtime_backed_provider_names_an_opencode_provider_or_is_custom():
    """The runtime must know which OpenCode provider a connection becomes."""
    for spec in providers.SUPPORTED:
        if spec.default_base_url is None:
            assert spec.runtime_provider_id is None, spec.id
        else:
            assert spec.runtime_provider_id, spec.id


def test_chatgpt_sign_in_is_listed_only_where_the_deployment_can_run_it():
    def modes(chatgpt):
        entry = next(e for e in providers.catalogue(chatgpt=chatgpt) if e["provider_id"] == "openai")
        return entry["auth_modes"], entry["chatgpt_oauth"]

    offered, oauth = modes({"available": True, "reason": "local"})
    assert "chatgpt_subscription" in offered and oauth["available"] is True
    refused, oauth = modes({"available": False, "reason": "hosted without opt-in"})
    assert "chatgpt_subscription" not in refused
    assert oauth == {"available": False, "reason": "hosted without opt-in"}
    assert "chatgpt_subscription" not in modes(None)[0]


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
