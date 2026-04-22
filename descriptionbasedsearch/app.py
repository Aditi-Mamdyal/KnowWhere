"""
app.py — Flask Web Interface for KnowWhere Corporate Search System
===================================================================
Wraps the existing auth.py and searching.py backend into a simple
browser-accessible web app for Selenium testing.

Run with:
    python app.py

Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash
import os, sys

# Make sure imports find the project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auth import verify_login, create_user, delete_user, list_users, Session as AuthSession
from searching import run_search

app = Flask(__name__)
app.secret_key = "knowwhere_selenium_test_key"

# In-memory session store: flask session holds username + role
# We create an AuthSession on login and store it per user

_auth_sessions = {}   # username -> AuthSession


# ── HELPER ────────────────────────────────────────────────────────────────────

def get_auth_session():
    username = session.get("username")
    if not username:
        return None
    auth_sess = _auth_sessions.get(username)
    if auth_sess and auth_sess.is_valid():
        auth_sess.refresh()
        return auth_sess
    return None


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if get_auth_session():
        return redirect(url_for("search"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        result = verify_login(username, password)
        if result["success"]:
            session["username"] = result["username"]
            session["role"]     = result["role"]
            _auth_sessions[result["username"]] = AuthSession(
                result["username"], result["role"]
            )
            return redirect(url_for("search"))
        else:
            error = result["reason"]

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    username = session.get("username")
    if username in _auth_sessions:
        del _auth_sessions[username]
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("login"))


@app.route("/search", methods=["GET", "POST"])
def search():
    auth_sess = get_auth_session()
    if not auth_sess:
        flash("Please log in first.")
        return redirect(url_for("login"))

    results      = []
    query        = ""
    mode         = "documents"
    result_count = 0

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        mode  = request.form.get("mode", "documents")

        if query:
            raw = run_search(query, mode=mode, session=auth_sess)
            for score, path, rtype in raw[:10]:
                results.append({
                    "score": round(float(score), 4),
                    "path":  path,
                    "name":  os.path.basename(path),
                    "type":  rtype,
                })
            result_count = len(results)

    return render_template(
        "search.html",
        username=auth_sess.username,
        role=auth_sess.role,
        query=query,
        mode=mode,
        results=results,
        result_count=result_count,
    )


@app.route("/admin", methods=["GET", "POST"])
def admin():
    auth_sess = get_auth_session()
    if not auth_sess or not auth_sess.is_admin():
        flash("Admin access required.")
        return redirect(url_for("search"))

    message = None
    users   = list_users(auth_sess.username).get("users", [])

    if request.method == "POST":
        action = request.form.get("action")

        if action == "create":
            new_user  = request.form.get("new_username", "").strip()
            new_pass  = request.form.get("new_password", "").strip()
            new_role  = request.form.get("new_role", "user")
            res = create_user(auth_sess.username, new_user, new_pass, new_role)
            message = ("success", f"User '{new_user}' created.") if res["success"] \
                      else ("error", res["reason"])

        elif action == "delete":
            target = request.form.get("target_username", "").strip()
            res = delete_user(auth_sess.username, target)
            message = ("success", f"User '{target}' deleted.") if res["success"] \
                      else ("error", res["reason"])

        users = list_users(auth_sess.username).get("users", [])

    return render_template(
        "admin.html",
        username=auth_sess.username,
        users=users,
        message=message,
    )


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create templates folder if missing
    tmpl = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(tmpl, exist_ok=True)
    app.run(debug=True, port=5000)