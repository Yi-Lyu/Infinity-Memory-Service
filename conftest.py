import pytest
import pytest_asyncio

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as async"
    )

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()