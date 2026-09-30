"""Auth lifecycle: register, login, refresh, me, password change."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import register


def test_register_returns_user_and_tokens(client: TestClient) -> None:
    account = register(client)
    assert account["user"]["email"] == "ada@example.com"
    assert account["tokens"]["token_type"] == "bearer"
    assert account["tokens"]["access_token"]


def test_duplicate_email_conflicts(client: TestClient) -> None:
    register(client)
    response = client.post(
        "/api/v1/auth/register",
        json={"name": "Ada", "email": "ada@example.com", "password": "analytical1"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "conflict"


def test_login_and_me(client: TestClient) -> None:
    register(client)
    login = client.post("/api/v1/auth/login", json={"email": "ada@example.com", "password": "analytical1"})
    assert login.status_code == 200
    headers = {"Authorization": f"Bearer {login.json()['tokens']['access_token']}"}
    me = client.get("/api/v1/users/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["name"] == "Ada Lovelace"


def test_login_wrong_password(client: TestClient) -> None:
    register(client)
    response = client.post("/api/v1/auth/login", json={"email": "ada@example.com", "password": "wrongpass1"})
    assert response.status_code == 401


def test_refresh_rotates_token(client: TestClient) -> None:
    account = register(client)
    response = client.post("/api/v1/auth/refresh", json={"refresh_token": account["tokens"]["refresh_token"]})
    assert response.status_code == 200
    new_tokens = response.json()["tokens"]
    assert new_tokens["refresh_token"] != account["tokens"]["refresh_token"]
    # Old refresh token is revoked.
    again = client.post("/api/v1/auth/refresh", json={"refresh_token": account["tokens"]["refresh_token"]})
    assert again.status_code == 401


def test_me_requires_auth(client: TestClient) -> None:
    assert client.get("/api/v1/users/me").status_code == 401
    assert client.get("/api/v1/users/me", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_logout_revokes_refresh(client: TestClient) -> None:
    account = register(client)
    response = client.post("/api/v1/auth/logout", json={"refresh_token": account["tokens"]["refresh_token"]})
    assert response.status_code == 200
    assert response.json()["revoked"] == 1
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": account["tokens"]["refresh_token"]}).status_code == 401


def test_health_is_public(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}
