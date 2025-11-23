import pytest
from pathlib import Path


def pytest_sessionstart(session):
    """Create test output directories before running tests."""
    plots_dir = Path(__file__).parent / 'plots'
    plots_dir.mkdir(exist_ok=True)


@pytest.fixture
def plots_dir():
    """Provide path to plots directory for tests."""
    return Path(__file__).parent / 'plots'
