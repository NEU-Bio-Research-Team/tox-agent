"""DSH and Codex are spikes, and the code has to keep saying so (K09).

ADR 0004 once asserted a `dsh` adapter "written against the pinned version
0.1.1-rc.2... covered by contract suites marked live_runtime". No such adapter
existed; the DSH side of that table was enum and config scaffolding nothing
implemented. ADR 0007 corrected it, ADR 0008 recorded Codex the same way, and
both are marked as spikes with no adapter.

An ADR is prose, and prose does not notice when an adapter appears beside it
and starts taking runs. K09's closing criterion is that neither is promoted
before the auth, tool-surface, cancel, recovery and eval-matrix evidence
exists — so promotion has to be a change this test fails on, forcing the ADR
and the evidence to move together, rather than something that can happen
quietly in a directory listing.

What this does *not* assert is that DSH is a bad idea or that the scaffolding
should go. `RuntimeKind.DSH` and the `dsh_*` settings are deliberate: they say
what a future adapter would target. Declaring an intention is not the same as
claiming support.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from toxagent.config import RuntimeSettings
from toxagent.domain.runtime import RuntimeKind

ADAPTERS = Path(__import__("toxagent").__file__).resolve().parent / "harness" / "adapters"
ADR_DIR = Path(__file__).resolve().parents[2] / "docs" / "adr"

#: Runtime kinds an adapter ships for. Everything else in `RuntimeKind` is a
#: declared target, not a supported one.
IMPLEMENTED = {"opencode_v1", "scripted"}

EXPERIMENTAL_ADRS = {
    "dsh": "0007-dsh-conformance-spike.md",
    "codex": "0008-codex-runtime-spike.md",
}


def adapter_modules() -> set[str]:
    return {p.stem for p in ADAPTERS.glob("*.py") if not p.stem.startswith("_")}


def test_the_only_adapters_that_ship_are_the_two_that_are_supported():
    """A new adapter file is a promotion, whatever the ADR still says."""
    assert adapter_modules() == IMPLEMENTED, (
        "an adapter appeared or disappeared; if this is a promotion, the ADR, "
        "the conformance evidence and this list move together"
    )


@pytest.mark.parametrize("runtime,adr_name", sorted(EXPERIMENTAL_ADRS.items()))
def test_each_experimental_runtime_has_an_adr_that_still_calls_it_one(runtime, adr_name):
    adr = ADR_DIR / adr_name
    assert adr.is_file(), f"{runtime}: {adr_name} is gone"
    status = next(
        line for line in adr.read_text().splitlines() if line.startswith("**Status:**")
    )
    assert "experimental" in status or "spike" in status, (
        f"{runtime}: {adr_name} no longer marks it a spike — promoting needs the "
        "auth, tool-surface, cancel, recovery and eval-matrix evidence, not an edit here"
    )


def test_no_experimental_runtime_has_an_adapter_behind_it():
    for runtime in EXPERIMENTAL_ADRS:
        assert not any(module.startswith(runtime) for module in adapter_modules()), runtime


def test_the_dsh_scaffolding_is_a_declared_target_not_a_claim_of_support():
    """The settings name a version an adapter would target. Keeping them is
    fine; what is not fine is a reader taking them for a shipped integration,
    which is what ADR 0004 did before its correction."""
    settings = RuntimeSettings()
    assert RuntimeKind.DSH.value == "dsh"
    assert settings.dsh_version, "the pinned target version is the point of the scaffolding"
    assert settings.kind == "none", "no runtime is configured by default"
