from pathlib import Path

from src.data.db import _load_db_config


def test_load_db_config_from_dotenv_whitespace(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "  DB_HOST= localhost",
                "DB_PORT=5433",
                "DB_USER=kekoa",
                "DB_PASSWORD= goatez ",
                "DB_NAME=mlb_fantasy",
                "SCHEMA=production",
            ]
        ),
        encoding="utf-8",
    )

    for key in ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME", "SCHEMA"]:
        monkeypatch.delenv(key, raising=False)

    cfg = _load_db_config(path=tmp_path / "missing.yaml", env_path=env_path)

    assert cfg["host"] == "localhost"
    assert cfg["port"] == 5433
    assert cfg["user"] == "kekoa"
    assert cfg["password"] == "goatez"
    assert cfg["dbname"] == "mlb_fantasy"
    assert cfg["schema"] == "production"


def test_env_vars_override_dotenv_values(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DB_HOST=from_file",
                "DB_PORT=5433",
                "DB_USER=file_user",
                "DB_PASSWORD=file_pw",
                "DB_NAME=file_db",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DB_HOST", "from_env")
    monkeypatch.setenv("DB_USER", "env_user")
    monkeypatch.setenv("DB_PASSWORD", "env_pw")
    monkeypatch.setenv("DB_NAME", "env_db")
    monkeypatch.setenv("DB_PORT", "6432")

    cfg = _load_db_config(path=tmp_path / "missing.yaml", env_path=env_path)

    assert cfg["host"] == "from_env"
    assert cfg["user"] == "env_user"
    assert cfg["password"] == "env_pw"
    assert cfg["dbname"] == "env_db"
    assert cfg["port"] == 6432
