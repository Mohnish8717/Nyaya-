"""Shared fixtures.

`client_session` gives a started session and a client already carrying its
bearer token, which is what most endpoint tests need.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import sessions
from app.main import app
from tests.support import AuthedClient, start


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    """Rate counters are in the shared store, so they outlive a run. Without
    this, a suite that runs twice in a minute inherits the first run's spent
    budget and fails in ways that have nothing to do with the code under test.
    """
    sessions.LIMITER.clear()
    yield


@pytest.fixture
def client_session():
    with TestClient(app) as inner:
        client = AuthedClient(inner)
        session_id, _ = start(client)
        yield client, session_id
