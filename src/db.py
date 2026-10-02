"""SQLite storage via SQLAlchemy.

Two tables:
  patients     the cleaned UCI records (loaded once by `python -m src.load_db`)
  predictions  predictions made in the web app, each tied to an anonymous
               visitor id (a random value kept in the visitor's browser cookie),
               so people only ever see and delete their own history.
               Rows are deleted automatically after HISTORY_DAYS days.

To use PostgreSQL later, set DATABASE_URL, e.g.
  postgresql+psycopg2://user:password@localhost:5432/heart
(and `pip install psycopg2-binary`). Nothing else needs to change.
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import (JSON, Boolean, DateTime, Float, Integer, String, create_engine, inspect,
                        select, text)
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from src.preprocess import ROOT

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{(ROOT / 'heart.db').as_posix()}")
HISTORY_DAYS = int(os.environ.get("HISTORY_DAYS", 7))
engine = create_engine(DATABASE_URL)


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(primary_key=True)
    hospital: Mapped[str] = mapped_column(String(30))
    age: Mapped[int] = mapped_column(Integer)
    sex: Mapped[str] = mapped_column(String(10))
    cp: Mapped[str] = mapped_column(String(30))
    trestbps: Mapped[float | None] = mapped_column(Float)
    chol: Mapped[float | None] = mapped_column(Float)
    fbs: Mapped[str | None] = mapped_column(String(5))
    restecg: Mapped[str | None] = mapped_column(String(30))
    thalch: Mapped[float | None] = mapped_column(Float)
    exang: Mapped[str | None] = mapped_column(String(5))
    oldpeak: Mapped[float | None] = mapped_column(Float)
    slope: Mapped[str | None] = mapped_column(String(30))
    ca: Mapped[float | None] = mapped_column(Float)
    thal: Mapped[str | None] = mapped_column(String(30))
    has_disease: Mapped[bool] = mapped_column(Boolean)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    visitor_id: Mapped[str | None] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: _utcnow())
    inputs: Mapped[dict] = mapped_column(JSON)
    probability: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    flagged_high_risk: Mapped[bool] = mapped_column(Boolean)
    model_name: Mapped[str] = mapped_column(String(50))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)  # stored as naive UTC


_ready = False


def _create_and_migrate(eng):
    Base.metadata.create_all(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("predictions")}
    if "visitor_id" not in cols:
        with eng.begin() as conn:
            conn.execute(text("ALTER TABLE predictions ADD COLUMN visitor_id VARCHAR(32)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_predictions_visitor_id "
                              "ON predictions (visitor_id)"))


def init_db(eng=None):
    """Create missing tables and add `visitor_id` to an older predictions table.

    Several server workers (gunicorn) can start at the same moment on a fresh
    database; if one creates a table while another is checking for it, the
    second gets "already exists". Retrying a couple of times resolves that."""
    global _ready
    if eng is None:
        if _ready:
            return
        eng = engine
    for attempt in range(3):
        try:
            _create_and_migrate(eng)
            break
        except DBAPIError:
            if attempt == 2:
                raise
            time.sleep(0.2 * (attempt + 1))
    if eng is engine:
        _ready = True


def load_patients(df: pd.DataFrame):
    """Replace the patients table with the cleaned dataframe."""
    init_db()
    rows = df.rename(columns={"dataset": "hospital", "target": "has_disease"}).copy()
    rows["has_disease"] = rows["has_disease"].astype(bool)
    rows = rows.astype(object).where(rows.notna(), None)
    with Session(engine) as s:
        s.query(Patient).delete()
        s.add_all(Patient(**r) for r in rows.to_dict("records"))
        s.commit()
        return s.query(Patient).count()


def log_prediction(visitor_id: str, inputs: dict, probability: float, threshold: float, model_name: str):
    init_db()
    clean = {k: (None if pd.isna(v) else v) for k, v in inputs.items()}
    with Session(engine) as s:
        # keep health inputs only for a limited time
        s.query(Prediction).filter(Prediction.created_at < _utcnow() - timedelta(days=HISTORY_DAYS)).delete()
        s.add(Prediction(visitor_id=visitor_id, inputs=json.loads(json.dumps(clean, default=str)),
                         probability=float(probability), threshold=float(threshold),
                         flagged_high_risk=bool(probability >= threshold), model_name=model_name))
        s.commit()


def clear_predictions(visitor_id: str | None) -> int:
    """Delete one visitor's predictions. Without a visitor id, deletes nothing."""
    if not visitor_id:
        return 0
    init_db()
    with Session(engine) as s:
        n = s.query(Prediction).filter(Prediction.visitor_id == visitor_id).delete()
        s.commit()
    return n


def recent_predictions(visitor_id: str | None, limit=50) -> pd.DataFrame:
    """This visitor's most recent predictions. Without a visitor id: nothing."""
    if not visitor_id:
        return pd.DataFrame()
    init_db()
    with Session(engine) as s:
        rows = s.scalars(select(Prediction).where(Prediction.visitor_id == visitor_id)
                         .order_by(Prediction.id.desc()).limit(limit)).all()
    return pd.DataFrame([{
        "time (UTC)": r.created_at.strftime("%b %d, %H:%M"),
        "model": r.model_name,
        "risk": f"{r.probability:.0%}",
        "high risk?": "Yes" if r.flagged_high_risk else "No",
        **r.inputs} for r in rows])
