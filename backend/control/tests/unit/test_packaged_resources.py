"""Files the running code reads must resolve from the installed package.

The src layout move (I22) silently repointed ``profiles_dir`` at
``src/agent_profiles``, a directory that has never existed. Nothing failed at
import; the first agent run failed instead, and the image failed the same way
as the checkout because Compose sets no ``TOXAGENT_PROFILES_DIR`` override.
These assertions fail on the *resolution rule*, not on one path string, so a
future relocation cannot reintroduce the same class of bug.
"""
from __future__ import annotations

from toxagent.config import PACKAGE_ROOT, SERVICE_ROOT, Settings


def test_package_root_is_the_toxagent_package_not_the_src_directory():
    assert PACKAGE_ROOT.name == "toxagent"
    assert (PACKAGE_ROOT / "__init__.py").is_file()


def test_service_root_is_where_mutable_state_lives_not_a_shipped_directory():
    # backend/control in a checkout, /app in the image. It must not be the
    # package (mutable state inside an installed package is unwritable in a
    # wheel install) and must not be ``src`` (which no install layout has).
    assert SERVICE_ROOT != PACKAGE_ROOT
    assert SERVICE_ROOT.name != "src"


def test_default_profiles_dir_exists_and_holds_the_pinned_opencode_profile():
    settings = Settings.from_env()
    assert settings.profiles_dir.is_dir(), settings.profiles_dir
    assert (settings.profiles_dir / "opencode" / "toxagent.json").is_file()


def test_default_object_store_dir_is_service_local_not_inside_the_package():
    settings = Settings.from_env()
    assert PACKAGE_ROOT not in settings.object_store_dir.parents


def test_every_runtime_resource_the_package_reads_is_declared_as_package_data():
    # Keeping this list next to the assertion is the point: adding a resource
    # without adding it to pyproject.toml's package-data breaks non-editable
    # installs only, which no other test in this suite exercises. Read as text
    # rather than TOML — tomllib is 3.11+, and this package supports 3.10.
    pyproject = (SERVICE_ROOT / "pyproject.toml").read_text()
    assert "[tool.setuptools.package-data]" in pyproject
    assert '"agent_profiles/**/*"' in pyproject
    assert '"predictor/contract_snapshot.json"' in pyproject
