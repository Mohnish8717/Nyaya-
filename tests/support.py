"""Test client that carries the session's bearer token.

A session id is not a credential any more: every authenticated endpoint needs
the token issued at creation. Tests should not repeat that header twenty
times, and should not be able to forget it and pass by accident, so the
wrapper attaches it to every request once the session exists.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


class AuthedClient:
    def __init__(self, inner: TestClient) -> None:
        self._inner = inner
        self._token = ""

    def set_token(self, token: str) -> None:
        self._token = token

    def _prepare(self, kwargs: dict) -> dict:
        if self._token and not kwargs.get("headers"):
            kwargs["headers"] = {"Authorization": f"Bearer {self._token}"}
        return kwargs

    def post(self, url: str, **kwargs):
        return self._inner.post(url, **self._prepare(kwargs))

    def get(self, url: str, **kwargs):
        return self._inner.get(url, **self._prepare(kwargs))

    def delete(self, url: str, **kwargs):
        return self._inner.delete(url, **self._prepare(kwargs))


def start(client: AuthedClient, language: str = "en-IN") -> tuple[str, dict]:
    """Create a session and arm the client with its token."""
    response = client.post("/api/session", json={"language": language, "consent_confirmed": True})
    assert response.status_code == 200, response.text
    data = response.json()
    client.set_token(data["session_token"])
    return data["session_id"], data


def new_client() -> AuthedClient:
    return AuthedClient(TestClient(app))
