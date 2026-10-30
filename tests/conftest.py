"""Shared pytest setup for the self-contained public test bundle."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


TEST_PROJECT_ROOT = Path(__file__).resolve().parent / "fixtures" / "project"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def use_public_project_fixtures():
    """Resolve repository-relative test data from the sanitized fixture tree.

    Production defaults intentionally continue to use ``data/`` relative to the
    caller's working directory. Tests switch their working directory instead of
    changing those public interfaces, which keeps a core-only Git checkout
    reproducible without publishing retrieved third-party source bodies.
    """

    # The development archive keeps the full local datasets and should continue
    # validating them. A core-only public checkout omits those directories and
    # therefore switches to the project-authored fixture mirror.
    has_development_data = (REPOSITORY_ROOT / "data").is_dir() and (
        REPOSITORY_ROOT / "experiments"
    ).is_dir()
    if has_development_data:
        yield
        return

    if not TEST_PROJECT_ROOT.is_dir():
        raise RuntimeError(f"Missing public test fixture root: {TEST_PROJECT_ROOT}")
    previous = Path.cwd()
    os.chdir(TEST_PROJECT_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)
