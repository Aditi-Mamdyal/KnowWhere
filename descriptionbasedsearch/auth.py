"""
auth.py
=======
Handles all authentication for the corporate search system.

HOW IT WORKS:
- Credentials stored in data/users.json as SHA-256 hashed passwords
- No plain text passwords ever stored or transmitted
- On first run, a default admin account is created automatically
- Admin can create/delete employee accounts
- Session tracks login time — auto expires after SESSION_TIMEOUT minutes
- Every login attempt and search is logged to data/audit.log

ROLES:
- admin : can create/delete users, access search
- user  : can only access search

DEFAULT ADMIN (change after first login):
  username: admin
  password: admin123
"""

import os
import json
import hashlib
import logging
import time
from datetime import datetime

# -------- CONFIG --------
DATA_DIR        = "data"
USERS_FILE      = os.path.join(DATA_DIR, "users.json")
AUDIT_LOG_FILE  = os.path.join(DATA_DIR, "audit.log")
SESSION_TIMEOUT = 30 * 60   # 30 minutes in seconds

os.makedirs(DATA_DIR, exist_ok=True)

# -------- AUDIT LOGGER --------
# Separate logger just for audit trail — goes to file only, not console
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)

if not audit_logger.handlers:
    fh = logging.FileHandler(AUDIT_LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    audit_logger.addHandler(fh)
    audit_logger.propagate = False   # don't bubble up to root logger


# =============================================================================
# PASSWORD HASHING
# =============================================================================

def _hash_password(password: str) -> str:
    """
    SHA-256 hash of the password.
    We add a fixed salt prefix so even if someone gets the users.json,
    plain rainbow table attacks won't work directly.
    """
    salted = f"descSearch_salt_{password}_end"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()


# =============================================================================
# USER STORAGE
# =============================================================================

def _load_users() -> dict:
    """
    Load users from users.json.
    Structure:
    {
        "admin": {"password_hash": "abc123...", "role": "admin"},
        "john":  {"password_hash": "def456...", "role": "user"}
    }
    """
    if not os.path.exists(USERS_FILE):
        # First run — create default admin account
        _save_users({
            "admin": {
                "password_hash": _hash_password("admin123"),
                "role": "admin"
            }
        })
        print("[AUTH] First run detected. Default admin account created.")
        print("[AUTH] Username: admin | Password: admin123")
        print("[AUTH] Please change the password after first login.")

    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict):
    """Save users dict to users.json."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


# =============================================================================
# AUTH FUNCTIONS
# =============================================================================

def verify_login(username: str, password: str) -> dict:
    """
    Verify login credentials.

    Returns:
        {"success": True,  "role": "admin"/"user", "username": username}
        {"success": False, "reason": "..."}
    """
    username = username.strip().lower()

    if not username or not password:
        audit_logger.info(f"LOGIN FAILED | user='{username}' | reason=empty credentials")
        return {"success": False, "reason": "Username and password are required."}

    users = _load_users()

    if username not in users:
        audit_logger.info(f"LOGIN FAILED | user='{username}' | reason=user not found")
        return {"success": False, "reason": "Invalid username or password."}

    expected_hash = users[username]["password_hash"]
    actual_hash   = _hash_password(password)

    if expected_hash != actual_hash:
        audit_logger.info(f"LOGIN FAILED | user='{username}' | reason=wrong password")
        return {"success": False, "reason": "Invalid username or password."}

    role = users[username].get("role", "user")
    audit_logger.info(f"LOGIN SUCCESS | user='{username}' | role={role}")

    return {"success": True, "role": role, "username": username}


def change_password(username: str, old_password: str,
                    new_password: str) -> dict:
    """
    Allow a user to change their own password.

    Returns:
        {"success": True}  or  {"success": False, "reason": "..."}
    """
    username = username.strip().lower()
    users    = _load_users()

    if username not in users:
        return {"success": False, "reason": "User not found."}

    if users[username]["password_hash"] != _hash_password(old_password):
        return {"success": False, "reason": "Current password is incorrect."}

    if len(new_password) < 6:
        return {"success": False, "reason": "New password must be at least 6 characters."}

    users[username]["password_hash"] = _hash_password(new_password)
    _save_users(users)

    audit_logger.info(f"PASSWORD CHANGED | user='{username}'")
    return {"success": True}


# =============================================================================
# ADMIN FUNCTIONS (only callable when logged in as admin)
# =============================================================================

def create_user(admin_username: str, new_username: str,
                new_password: str, role: str = "user") -> dict:
    """
    Admin creates a new user account.

    Returns:
        {"success": True}  or  {"success": False, "reason": "..."}
    """
    users = _load_users()

    # verify the caller is actually an admin
    if users.get(admin_username, {}).get("role") != "admin":
        return {"success": False, "reason": "Only admins can create users."}

    new_username = new_username.strip().lower()

    if not new_username:
        return {"success": False, "reason": "Username cannot be empty."}

    if new_username in users:
        return {"success": False, "reason": f"User '{new_username}' already exists."}

    if len(new_password) < 6:
        return {"success": False, "reason": "Password must be at least 6 characters."}

    if role not in ("admin", "user"):
        role = "user"

    users[new_username] = {
        "password_hash": _hash_password(new_password),
        "role": role
    }
    _save_users(users)

    audit_logger.info(
        f"USER CREATED | by='{admin_username}' | new_user='{new_username}' | role={role}"
    )
    return {"success": True}


def delete_user(admin_username: str, target_username: str) -> dict:
    """
    Admin deletes a user account.
    Cannot delete yourself or the last admin.
    """
    users = _load_users()

    if users.get(admin_username, {}).get("role") != "admin":
        return {"success": False, "reason": "Only admins can delete users."}

    target_username = target_username.strip().lower()

    if target_username == admin_username:
        return {"success": False, "reason": "You cannot delete your own account."}

    if target_username not in users:
        return {"success": False, "reason": f"User '{target_username}' not found."}

    # prevent deleting the last admin
    remaining_admins = [
        u for u, d in users.items()
        if d.get("role") == "admin" and u != target_username
    ]
    if not remaining_admins and users[target_username].get("role") == "admin":
        return {"success": False, "reason": "Cannot delete the last admin account."}

    del users[target_username]
    _save_users(users)

    audit_logger.info(
        f"USER DELETED | by='{admin_username}' | deleted_user='{target_username}'"
    )
    return {"success": True}


def list_users(admin_username: str) -> dict:
    """Returns list of all users (admin only)."""
    users = _load_users()

    if users.get(admin_username, {}).get("role") != "admin":
        return {"success": False, "reason": "Only admins can list users."}

    user_list = [
        {"username": u, "role": d.get("role", "user")}
        for u, d in users.items()
    ]
    return {"success": True, "users": user_list}


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class Session:
    """
    Tracks who is logged in and when.
    Stored in memory only — destroyed when app closes.
    """

    def __init__(self, username: str, role: str):
        self.username   = username
        self.role       = role
        self._last_active = time.time()

    def is_valid(self) -> bool:
        """Returns False if session has been idle for SESSION_TIMEOUT seconds."""
        return (time.time() - self._last_active) < SESSION_TIMEOUT

    def refresh(self):
        """Call this on every user action to reset the timeout countdown."""
        self._last_active = time.time()

    def time_remaining(self) -> int:
        """Returns seconds remaining before session expires."""
        elapsed = time.time() - self._last_active
        return max(0, int(SESSION_TIMEOUT - elapsed))

    def is_admin(self) -> bool:
        return self.role == "admin"


# =============================================================================
# AUDIT LOGGING (called from searching.py and GUI)
# =============================================================================

def log_search(username: str, query: str, mode: str, result_count: int):
    """Log a search action to audit.log."""
    audit_logger.info(
        f"SEARCH | user='{username}' | mode={mode} | "
        f"query='{query}' | results={result_count}"
    )


def log_logout(username: str):
    """Log a logout event."""
    audit_logger.info(f"LOGOUT | user='{username}'")


# =============================================================================
# QUICK TEST — run this file directly to test auth system
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("AUTH SYSTEM TEST")
    print("=" * 50)

    # Test 1: wrong password
    result = verify_login("admin", "wrongpassword")
    print(f"\nTest 1 - Wrong password: {result}")

    # Test 2: correct login
    result = verify_login("admin", "admin123")
    print(f"Test 2 - Correct login:  {result}")

    # Test 3: create a new user
    result = create_user("admin", "john_doe", "pass123", role="user")
    print(f"Test 3 - Create user:    {result}")

    # Test 4: login as new user
    result = verify_login("john_doe", "pass123")
    print(f"Test 4 - New user login: {result}")

    # Test 5: wrong username
    result = verify_login("nobody", "pass123")
    print(f"Test 5 - Unknown user:   {result}")

    # Test 6: list users
    result = list_users("admin")
    print(f"Test 6 - List users:     {result}")

    # Test 7: session
    session = Session("admin", "admin")
    print(f"\nTest 7 - Session valid:  {session.is_valid()}")
    print(f"         Time remaining: {session.time_remaining()}s")

    # Test 8: change password
    result = change_password("john_doe", "pass123", "newpass456")
    print(f"Test 8 - Change pass:    {result}")

    # Test 9: login with new password
    result = verify_login("john_doe", "newpass456")
    print(f"Test 9 - New pass login: {result}")

    # Test 10: delete user
    result = delete_user("admin", "john_doe")
    print(f"Test 10 - Delete user:   {result}")

    print(f"\nCheck data/audit.log to see all logged events.")
    print("=" * 50)