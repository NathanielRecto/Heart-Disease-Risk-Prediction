"""Flask web app:  python app.py   then open http://127.0.0.1:5000"""
import os
import re
import secrets

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

from src.db import HISTORY_DAYS, clear_predictions, init_db, log_prediction, recent_predictions
from src.predict import best_artifact, explain, guidance, load_artifacts, parse_inputs, predict_proba
from src.preprocess import ROOT, load_data

app = Flask(__name__)
init_db()  # create tables / upgrade an older database before serving requests
# Behind a hosting proxy (Hugging Face, Cloud Run) the app sees plain HTTP; trust the proxy's
# X-Forwarded-Proto so it knows the visitor is really on HTTPS.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1)
REPORT_IMAGES = {"confusion_matrices.png", "precision_recall.png"}

# History is private per visitor: a random anonymous id lives in an HttpOnly cookie.
VISITOR_COOKIE = "pc_visitor"
VISITOR_RE = re.compile(r"^[A-Za-z0-9_-]{16,32}$")
COOKIE_DAYS = 30


def current_visitor():
    """The visitor id from the cookie, or None (missing or malformed)."""
    value = request.cookies.get(VISITOR_COOKIE, "")
    return value if VISITOR_RE.match(value) else None


def set_visitor_cookie(response, visitor_id):
    # Over HTTPS use SameSite=None so the cookie also works when the app is embedded in an
    # iframe (as on huggingface.co); on plain HTTP (local use) Lax is the safe default.
    response.set_cookie(VISITOR_COOKIE, visitor_id, max_age=COOKIE_DAYS * 86400, httponly=True,
                        secure=request.is_secure, samesite="None" if request.is_secure else "Lax")
    return response


def model_cards():
    """Display info for every saved model (metrics are from the held-out test set)."""
    return [{
        "name": a["model_name"], "best": a["is_best"], "threshold": a["threshold"],
        "recall": a["test_recall"], "precision": a["test_precision"], "auc": a["test_roc_auc"],
    } for a in load_artifacts().values()]


@app.context_processor
def inject_globals():
    best = best_artifact()
    return {"history_days": HISTORY_DAYS, "best_model": {"name": best["model_name"], "recall": best["test_recall"],
                           "precision": best["test_precision"], "auc": best["test_roc_auc"],
                           "threshold": best["threshold"]}}


@app.get("/")
def home():
    return render_template("home.html", n_patients=len(load_data()))


@app.get("/assess")
def assess():
    return render_template("assess.html", models=model_cards())


@app.get("/how-it-works")
def how_it_works():
    return render_template("how.html", models=model_cards(), n_patients=len(load_data()))


@app.get("/history")
def history():
    df = recent_predictions(current_visitor(), limit=50)
    return render_template("history.html", rows=df.to_dict("records"), days=HISTORY_DAYS)


@app.post("/history/clear")
def history_clear():
    clear_predictions(current_visitor())  # only this visitor's rows
    return redirect(url_for("history"))


@app.get("/reports/<name>")
def report_image(name):
    if name not in REPORT_IMAGES:
        abort(404)
    return send_from_directory(ROOT / "reports", name)


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    row, extras, errors = parse_inputs(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    arts = load_artifacts()
    art = arts.get(payload.get("model")) or best_artifact()
    prob = predict_proba(art, row)
    high = prob >= art["threshold"]

    visitor = current_visitor()
    new_visitor = visitor is None
    if new_visitor:
        visitor = secrets.token_urlsafe(16)
    log_prediction(visitor, row, prob, art["threshold"], art["model_name"])
    response = jsonify({
        "probability": prob, "percent": round(prob * 100),
        "high_risk": bool(high), "threshold": art["threshold"],
        "model": art["model_name"], "model_recall": art["test_recall"],
        "bmi": extras["bmi"],
        "insights": explain(art, row, prob),
        "guidance": guidance(high, extras["bmi"]),
    })
    if new_visitor:
        set_visitor_cookie(response, visitor)
    return response


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)))
