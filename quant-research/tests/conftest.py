import copy
from pathlib import Path

import pytest

from quant_research.config import load_config
from quant_research.fixtures import create_fixture


@pytest.fixture(scope="session")
def base_snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("market")
    return create_fixture(root / "source", root / "snapshot", stocks=6)


@pytest.fixture
def snapshot(base_snapshot):
    return copy.deepcopy(base_snapshot)


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / "configs/research.toml")

