"""The installable-app surface: manifest, service worker, icons.

These are the parts that make it an app rather than a page, and every one of
them fails silently in a browser when it is wrong — a manifest with a bad MIME
type is ignored, a service worker outside its scope never registers, and a
missing 192px icon removes the install prompt. So they are asserted here.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_the_manifest_is_served_with_the_manifest_content_type(client):
    response = client.get("/manifest.webmanifest")
    assert response.status_code == 200
    # A manifest served as application/json is ignored by browsers.
    assert response.headers["content-type"].startswith("application/manifest+json")


def test_the_manifest_declares_an_installable_app(client):
    manifest = json.loads(client.get("/manifest.webmanifest").text)
    assert manifest["start_url"] == "/"
    assert manifest["scope"] == "/"
    assert manifest["display"] == "standalone"
    assert manifest["theme_color"].startswith("#")
    sizes = {icon["sizes"] for icon in manifest["icons"]}
    # Chrome requires both of these before it will offer installation.
    assert {"192x192", "512x512"} <= sizes
    assert any(icon.get("purpose") == "maskable" for icon in manifest["icons"])


def test_the_service_worker_is_served_from_the_root_with_scope(client):
    response = client.get("/sw.js")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    # A worker at /static/ could not control the app; this header widens it.
    assert response.headers.get("service-worker-allowed") == "/"
    # A cached worker can pin an old shell forever.
    assert "no-cache" in response.headers.get("cache-control", "")


def test_both_app_shell_routes_answer_head(client):
    for path in ("/manifest.webmanifest", "/sw.js"):
        assert client.head(path).status_code == 200


def test_the_declared_icons_exist_and_are_png(client):
    manifest = json.loads(client.get("/manifest.webmanifest").text)
    for icon in manifest["icons"]:
        response = client.get(icon["src"])
        assert response.status_code == 200, icon["src"]
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_page_links_the_manifest_and_installs_meta(client):
    page = client.get("/").text
    assert 'rel="manifest"' in page
    assert 'viewport-fit=cover' in page
    assert 'apple-mobile-web-app-capable' in page
    assert 'id="mobileNav"' in page
    assert 'id="installButton"' in page


def test_the_content_security_policy_allows_the_worker(client):
    policy = client.get("/").headers["content-security-policy"]
    assert "worker-src 'self'" in policy
    assert "manifest-src 'self'" in policy
