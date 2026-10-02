"""Smoke test for the Flask app:  python tests/smoke_test_app.py

Uses Flask's test client (no server needed) and a throwaway database, so it never
touches real history. Covers: every page, every model, input validation, and the
privacy behaviour (per-visitor history, cookie flags, tampered cookies, automatic
deletion of old rows, migration of an older database)."""
import os
import sqlite3
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp())
os.environ["DATABASE_URL"] = f"sqlite:///{(TMP / 'test.db').as_posix()}"

from sqlalchemy import create_engine, inspect  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app import app  # noqa: E402
from src import db  # noqa: E402

HIGH = {"age": 67, "sex": "Male", "cp": "asymptomatic", "exang": "True", "oldpeak": 2.5, "thalch": 110,
        "trestbps": 160, "chol": 280, "fbs": "True", "restecg": "lv hypertrophy"}
LOW = {"age": 35, "sex": "Female", "cp": "atypical angina", "exang": "False", "oldpeak": 0, "thalch": 185,
       "trestbps": 115, "chol": 180, "fbs": "False", "restecg": "normal", "height_cm": 168, "weight_kg": 82}


def history(client):
    return client.get("/history").get_data(as_text=True)


def test_pages_and_models():
    c = app.test_client()
    for url in ["/", "/assess", "/how-it-works", "/history", "/reports/confusion_matrices.png"]:
        assert c.get(url).status_code == 200, url
    assert c.get("/reports/../app.py").status_code == 404
    assert c.get("/reports/secret.txt").status_code == 404

    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        for label, body in [("high", HIGH), ("low", LOW)]:
            r = c.post("/api/predict", json={**body, "model": name})
            assert r.status_code == 200, r.get_data(as_text=True)
            d = r.get_json()
            print(f"{name:20s} {label:4s} -> {d['percent']:3d}%  flagged={d['high_risk']}")
            assert 0 <= d["percent"] <= 100 and d["model"] == name
    assert c.post("/api/predict", json={**HIGH, "model": "XGBoost"}).get_json()["high_risk"]
    assert not c.post("/api/predict", json={**LOW, "model": "XGBoost"}).get_json()["high_risk"]
    assert c.post("/api/predict", json=LOW).get_json()["bmi"] == 29.1  # 82 / 1.68^2
    assert c.post("/api/predict", json={"age": 50, "sex": "Male", "cp": "non-anginal", "exang": "False"}).status_code == 200

    bad = c.post("/api/predict", json={"age": 5, "sex": "x", "cp": "nope", "chol": "abc"})
    assert bad.status_code == 400
    assert {"age", "sex", "cp", "exang", "chol"} <= set(bad.get_json()["errors"])


def test_malformed_requests_are_rejected_not_crashing():
    c = app.test_client()
    ok = {"age": 50, "sex": "Male", "cp": "non-anginal", "exang": "False"}
    for body in [[1, 2], "text", {**ok, "sex": ["Male"]}, {**ok, "age": True}, {**ok, "oldpeak": True},
                 {**ok, "age": "nan"}, {**ok, "chol": "inf"}, {**ok, "cp": {"x": 1}}]:
        r = c.post("/api/predict", json=body)
        assert r.status_code == 400, (body, r.status_code)
    r = c.post("/api/predict", data="not json", content_type="application/json")
    assert r.status_code == 400
    r = c.post("/api/predict", json={**ok, "model": ["XGBoost"]})   # bad model name -> default model
    assert r.status_code == 200 and r.get_json()["model"] == "XGBoost"
    r = c.post("/api/predict", json={**ok, "pad": "x" * 20000})      # over the size limit
    assert r.status_code == 413


def test_blank_fields_carry_no_signal():
    """Leaving an optional field blank must give exactly the same answer as typing in the
    typical training value. If missingness itself were a feature (it reveals the hospital
    in this dataset), skipping a field would move the estimate on its own."""
    import numpy as np
    from src.predict import load_artifacts, predict_proba
    base = {"age": 50, "sex": "Male", "cp": "non-anginal", "exang": "False", "trestbps": 130.0, "chol": 220.0,
            "fbs": "False", "restecg": "normal", "thalch": 150.0, "oldpeak": 1.0}
    for name, art in load_artifacts().items():
        assert "calibrator" in art, f"{name} has no calibrator"
        fill = {}
        for tname, pipe, cols in art["model"].named_steps["prep"].transformers_:
            if tname in ("num", "cat"):
                fill.update(zip(cols, pipe.named_steps["impute"].statistics_))
        for field in ["trestbps", "chol", "fbs", "restecg", "thalch", "oldpeak"]:
            blank, typed = dict(base), dict(base)
            blank[field] = np.nan
            typed[field] = fill[field] if isinstance(fill[field], str) else float(fill[field])
            assert abs(predict_proba(art, blank) - predict_proba(art, typed)) < 1e-12, (name, field)


def test_history_is_private():
    a, b = app.test_client(), app.test_client()
    assert "No assessments yet" in history(a)          # new visitor: empty
    a.post("/api/predict", json={**HIGH, "model": "XGBoost"})
    a.post("/api/predict", json={**LOW, "model": "Random Forest"})
    assert "Random Forest" in history(a) and "XGBoost" in history(a)
    assert "No assessments yet" in history(b), "visitor B can see visitor A's history"

    b.post("/api/predict", json={**LOW, "model": "Logistic Regression"})
    assert "Logistic Regression" in history(b) and "Logistic Regression" not in history(a)

    b.post("/history/clear")                            # B clearing must not touch A
    assert "No assessments yet" in history(b)
    assert "Random Forest" in history(a)
    a.post("/history/clear")
    assert "No assessments yet" in history(a)

    anon = app.test_client()                            # no cookie: clear is a harmless no-op
    assert anon.post("/history/clear").status_code in (200, 302)


def test_cookie_flags_and_tampering():
    c = app.test_client()
    first = c.post("/api/predict", json=LOW).headers.getlist("Set-Cookie")
    assert len(first) == 1 and first[0].startswith("pc_visitor=")
    assert "HttpOnly" in first[0] and "SameSite=Lax" in first[0] and "Secure" not in first[0], first[0]
    assert c.post("/api/predict", json=LOW).headers.getlist("Set-Cookie") == []   # not re-issued

    secure = app.test_client().post("/api/predict", json=LOW, headers={"X-Forwarded-Proto": "https"})
    cookie = secure.headers.getlist("Set-Cookie")[0]
    assert "Secure" in cookie and "SameSite=None" in cookie and "HttpOnly" in cookie, cookie

    t = app.test_client()                               # tampered / malformed cookie values are ignored
    for bad in ["../../etc", "x", "a" * 80, "' OR 1=1 --"]:
        t.set_cookie("pc_visitor", bad)
        assert "No assessments yet" in history(t), bad
        r = t.post("/api/predict", json=LOW)
        assert r.status_code == 200 and r.headers.getlist("Set-Cookie"), bad   # gets a fresh valid id


def test_old_rows_are_deleted():
    with Session(db.engine) as s:
        s.add(db.Prediction(visitor_id="oldvisitor_abcdefghij", created_at=db._utcnow() - timedelta(days=db.HISTORY_DAYS + 1),
                            inputs={"age": 40}, probability=0.5, threshold=0.4, flagged_high_risk=True, model_name="XGBoost"))
        s.commit()
        assert s.query(db.Prediction).filter_by(visitor_id="oldvisitor_abcdefghij").count() == 1
    app.test_client().post("/api/predict", json=LOW)    # any new prediction triggers the purge
    with Session(db.engine) as s:
        assert s.query(db.Prediction).filter_by(visitor_id="oldvisitor_abcdefghij").count() == 0


def test_migration_of_old_database():
    path = TMP / "old.db"
    con = sqlite3.connect(path)                         # the schema from before visitor ids existed
    con.execute("CREATE TABLE predictions (id INTEGER PRIMARY KEY, created_at DATETIME, inputs JSON, probability FLOAT,"
                " threshold FLOAT, flagged_high_risk BOOLEAN, model_name VARCHAR(50))")
    con.execute("INSERT INTO predictions VALUES (1, '2026-01-01 00:00:00', '{}', 0.5, 0.4, 1, 'XGBoost')")
    con.commit(); con.close()
    eng = create_engine(f"sqlite:///{path.as_posix()}")
    db.init_db(eng)
    assert "visitor_id" in {c["name"] for c in inspect(eng).get_columns("predictions")}
    with eng.connect() as conn:
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM predictions").scalar() == 1   # old row kept
    db.init_db(eng)                                     # running it twice is harmless


if __name__ == "__main__":
    for t in [test_pages_and_models, test_malformed_requests_are_rejected_not_crashing,
              test_blank_fields_carry_no_signal, test_history_is_private, test_cookie_flags_and_tampering,
              test_old_rows_are_deleted, test_migration_of_old_database]:
        t()
        print("PASS", t.__name__)
    print("OK")
