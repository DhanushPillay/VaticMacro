from app.factory import create_app


def test_health_ok() -> None:
    app = create_app()
    c = app.test_client()
    r = c.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"


def test_sandbox_clips_and_validates() -> None:
    app = create_app()
    c = app.test_client()
    # extreme out-of-range must be 422
    r = c.post("/api/predictive-sandbox", json={"wpi_pct_1m": 999})
    assert r.status_code in (400, 422)
    # valid minimal payload returns prediction
    r = c.post(
        "/api/predictive-sandbox",
        json={
            "WPIATT01INM661N_pct_1m": 1.2,
            "INTDSRINM193N_pct_1m": 0,
            "DEXINUS_pct_1m": 0.2,
            "Average of DCOILBRENTEU_pct_1m": -2.0,
        },
    )
    assert r.status_code == 200
    j = r.get_json()
    assert "prediction" in j


def test_refresh_requires_token_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("REFRESH_TOKEN", "secret123")
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("legacy_app", pathlib.Path("app.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # patch env before exec; module reads REFRESH_TOKEN on request, not import
    spec.loader.exec_module(mod)  # type: ignore
    mod.app.config["TESTING"] = True
    c = mod.app.test_client()
    r = c.post("/api/refresh")
    assert r.status_code == 401
