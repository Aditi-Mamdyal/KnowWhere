"""
gui.py
======
Main GUI for the Corporate Document Search System.
Built with Tkinter — no extra GUI library needed.

SCREENS:
  1. Login Screen       — username + password
  2. Search Screen      — document / image / both search with results
  3. Admin Panel        — create/delete/list users (admin only)
  4. Change Password    — available to all users from settings

USAGE:
  python gui.py

HOW IT INTEGRATES:
  - Calls start_indexing_service() on startup (background indexing)
  - Calls run_search(query, mode, session) for search
  - Calls verify_login(), create_user(), delete_user() for auth
  - Calls stop_indexing_service() on window close
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import os
import sys

# -------- BACKEND IMPORTS --------
from auth import (
    verify_login, create_user, delete_user, list_users,
    change_password, log_logout, Session
)
from searching import run_search
from indexing_service import start_indexing_service, stop_indexing_service

# -------- THEME --------
BG_DARK      = "#0f1117"
BG_CARD      = "#1a1d27"
BG_INPUT     = "#252836"
ACCENT       = "#4f8ef7"
ACCENT_HOVER = "#3a7be0"
ACCENT_ADMIN = "#f7a44f"
TEXT_PRIMARY = "#e8eaf0"
TEXT_MUTED   = "#7a7f94"
TEXT_SUCCESS = "#4fba74"
TEXT_WARN    = "#f7c44f"
TEXT_ERROR   = "#f74f4f"
BORDER       = "#2e3146"
FONT_TITLE   = ("Georgia", 22, "bold")
FONT_SUB     = ("Georgia", 13)
FONT_BODY    = ("Consolas", 11)
FONT_SMALL   = ("Consolas", 9)
FONT_LABEL   = ("Consolas", 11, "bold")
FONT_BTN     = ("Consolas", 11, "bold")
RESULT_DOC   = "#4f8ef7"
RESULT_IMG   = "#a44ff7"

# =============================================================================
# HELPER WIDGETS
# =============================================================================

def styled_entry(parent, show=None, width=30):
    e = tk.Entry(
        parent, show=show, width=width,
        bg=BG_INPUT, fg=TEXT_PRIMARY,
        insertbackground=TEXT_PRIMARY,
        relief="flat", font=FONT_BODY,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT
    )
    return e

def styled_button(parent, text, command, color=ACCENT,
                  hover_color=ACCENT_HOVER, width=18):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=color, fg=TEXT_PRIMARY,
        activebackground=hover_color,
        activeforeground=TEXT_PRIMARY,
        relief="flat", font=FONT_BTN,
        cursor="hand2", width=width,
        pady=6
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=hover_color))
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    return btn

def label(parent, text, font=FONT_BODY, color=TEXT_PRIMARY, **kw):
    return tk.Label(
        parent, text=text, bg=BG_DARK,
        fg=color, font=font, **kw
    )

def card_label(parent, text, font=FONT_BODY, color=TEXT_PRIMARY, **kw):
    return tk.Label(
        parent, text=text, bg=BG_CARD,
        fg=color, font=font, **kw
    )

# =============================================================================
# SCREEN 1 — LOGIN
# =============================================================================

class LoginScreen(tk.Frame):

    def __init__(self, master, on_login_success):
        super().__init__(master, bg=BG_DARK)
        self.on_login_success = on_login_success
        self._build()

    def _build(self):
        # Center everything
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        card = tk.Frame(self, bg=BG_CARD, padx=48, pady=42,
                        highlightthickness=1,
                        highlightbackground=BORDER)
        card.grid(row=0, column=0)

        # Logo / title
        tk.Label(card, text="⬡", bg=BG_CARD, fg=ACCENT,
                 font=("Georgia", 36)).pack(pady=(0, 4))
        tk.Label(card, text="KnowWhere",
                 bg=BG_CARD, fg=TEXT_PRIMARY,
                 font=FONT_TITLE).pack()
        tk.Label(card, text="Corporate Document & Image Search",
                 bg=BG_CARD, fg=TEXT_MUTED,
                 font=FONT_SMALL).pack(pady=(2, 28))

        # Username
        tk.Label(card, text="USERNAME", bg=BG_CARD,
                 fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w")
        self.username_entry = styled_entry(card, width=32)
        self.username_entry.pack(pady=(2, 14), ipady=6, fill="x")

        # Password
        tk.Label(card, text="PASSWORD", bg=BG_CARD,
                 fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w")
        self.password_entry = styled_entry(card, show="●", width=32)
        self.password_entry.pack(pady=(2, 6), ipady=6, fill="x")

        # Error label (hidden until needed)
        self.error_label = tk.Label(
            card, text="", bg=BG_CARD,
            fg=TEXT_ERROR, font=FONT_SMALL
        )
        self.error_label.pack(pady=(0, 16))

        # Login button
        btn = styled_button(card, "LOGIN", self._attempt_login, width=32)
        btn.pack(ipady=2, fill="x")

        tk.Label(card, text="Contact IT admin if you need an account.",
                 bg=BG_CARD, fg=TEXT_MUTED,
                 font=FONT_SMALL).pack(pady=(18, 0))

        # Enter key triggers login
        self.password_entry.bind("<Return>", lambda e: self._attempt_login())
        self.username_entry.bind("<Return>", lambda e: self.password_entry.focus())

        # Auto-focus username
        self.username_entry.focus()

    def _attempt_login(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        result = verify_login(username, password)

        if result["success"]:
            session = Session(result["username"], result["role"])
            self.on_login_success(session)
        else:
            self.error_label.config(text=result["reason"])
            self.password_entry.delete(0, tk.END)
            self.password_entry.focus()

# =============================================================================
# SCREEN 2 — MAIN SEARCH
# =============================================================================

class SearchScreen(tk.Frame):

    def __init__(self, master, session, on_logout,
                 on_admin_panel, on_change_password,
                 status_var):
        super().__init__(master, bg=BG_DARK)
        self.session          = session
        self.on_logout        = on_logout
        self.on_admin_panel   = on_admin_panel
        self.on_change_password = on_change_password
        self.status_var       = status_var
        self.search_mode      = tk.StringVar(value="both")
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ---- TOP BAR ----
        topbar = tk.Frame(self, bg=BG_CARD,
                          highlightthickness=1,
                          highlightbackground=BORDER)
        topbar.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        topbar.columnconfigure(1, weight=1)

        tk.Label(topbar, text="⬡  KnowWhere",
                 bg=BG_CARD, fg=ACCENT,
                 font=("Georgia", 14, "bold"),
                 padx=20, pady=12).grid(row=0, column=0, sticky="w")

        # Status label in center
        tk.Label(topbar, textvariable=self.status_var,
                 bg=BG_CARD, fg=TEXT_MUTED,
                 font=FONT_SMALL).grid(row=0, column=1)

        # Right side — user info + buttons
        right = tk.Frame(topbar, bg=BG_CARD, padx=16)
        right.grid(row=0, column=2, sticky="e")

        role_color = ACCENT_ADMIN if self.session.is_admin() else TEXT_MUTED
        tk.Label(right,
                 text=f"  {self.session.username}  [{self.session.role}]",
                 bg=BG_CARD, fg=role_color,
                 font=FONT_SMALL).pack(side="left", padx=4)

        if self.session.is_admin():
            styled_button(
                right, "Admin Panel", self.on_admin_panel,
                color=ACCENT_ADMIN, hover_color="#e0933a", width=12
            ).pack(side="left", padx=4)

        styled_button(
            right, "Change Password", self.on_change_password,
            color=BG_INPUT, hover_color=BORDER, width=16
        ).pack(side="left", padx=4)

        styled_button(
            right, "Logout", self.on_logout,
            color="#3a2020", hover_color="#5a2e2e", width=8
        ).pack(side="left", padx=4)

        # ---- SEARCH AREA ----
        middle = tk.Frame(self, bg=BG_DARK, padx=32, pady=24)
        middle.grid(row=1, column=0, sticky="nsew")
        middle.columnconfigure(0, weight=1)
        middle.rowconfigure(2, weight=1)

        # Search bar
        search_bar = tk.Frame(middle, bg=BG_CARD,
                              highlightthickness=1,
                              highlightbackground=BORDER)
        search_bar.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        search_bar.columnconfigure(0, weight=1)

        self.query_entry = tk.Entry(
            search_bar, bg=BG_CARD, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="flat", font=("Georgia", 14),
            highlightthickness=0
        )
        self.query_entry.grid(row=0, column=0, sticky="ew",
                              padx=20, pady=14, ipady=4)
        self.query_entry.insert(0, "Describe what you're looking for...")
        self.query_entry.config(fg=TEXT_MUTED)

        def on_focus_in(e):
            if self.query_entry.get() == "Describe what you're looking for...":
                self.query_entry.delete(0, tk.END)
                self.query_entry.config(fg=TEXT_PRIMARY)

        def on_focus_out(e):
            if not self.query_entry.get():
                self.query_entry.insert(0, "Describe what you're looking for...")
                self.query_entry.config(fg=TEXT_MUTED)

        self.query_entry.bind("<FocusIn>", on_focus_in)
        self.query_entry.bind("<FocusOut>", on_focus_out)
        self.query_entry.bind("<Return>", lambda e: self._do_search())

        styled_button(
            search_bar, "Search", self._do_search, width=10
        ).grid(row=0, column=1, padx=8, pady=8)

        # Mode selector
        mode_frame = tk.Frame(middle, bg=BG_DARK)
        mode_frame.grid(row=1, column=0, sticky="w", pady=(0, 16))

        tk.Label(mode_frame, text="Search in:",
                 bg=BG_DARK, fg=TEXT_MUTED,
                 font=FONT_SMALL).pack(side="left", padx=(0, 10))

        for mode_val, mode_lbl in [
            ("both", "All"),
            ("documents", "Documents"),
            ("images", "Images")
        ]:
            rb = tk.Radiobutton(
                mode_frame,
                text=mode_lbl,
                variable=self.search_mode,
                value=mode_val,
                bg=BG_DARK, fg=TEXT_PRIMARY,
                selectcolor=BG_INPUT,
                activebackground=BG_DARK,
                activeforeground=ACCENT,
                font=FONT_SMALL,
                cursor="hand2"
            )
            rb.pack(side="left", padx=8)

        # Results area
        results_frame = tk.Frame(middle, bg=BG_CARD,
                                 highlightthickness=1,
                                 highlightbackground=BORDER)
        results_frame.grid(row=2, column=0, sticky="nsew")
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            bg=BG_CARD, fg=TEXT_PRIMARY,
            font=FONT_BODY,
            relief="flat",
            state="disabled",
            wrap="word",
            padx=20, pady=16,
            spacing1=4, spacing2=2
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")

        # Configure text tags for coloured output
        self.results_text.tag_config("strong",  foreground=TEXT_SUCCESS)
        self.results_text.tag_config("moderate", foreground=TEXT_WARN)
        self.results_text.tag_config("weak",     foreground=TEXT_MUTED)
        self.results_text.tag_config("doc_tag",  foreground=RESULT_DOC)
        self.results_text.tag_config("img_tag",  foreground=RESULT_IMG)
        self.results_text.tag_config("path",     foreground=TEXT_PRIMARY)
        self.results_text.tag_config("warning",  foreground=TEXT_ERROR)
        self.results_text.tag_config("header",   foreground=ACCENT,
                                     font=("Consolas", 11, "bold"))

        self._write_welcome()

    def _write_welcome(self):
        self._set_text_enabled()
        self.results_text.insert("end",
            "Welcome to KnowWhere\n\n", "header")
        self.results_text.insert("end",
            "Type a natural language description in the search bar above.\n"
            "Examples:\n"
            "  • budget report for Q3\n"
            "  • tictalk counselling booking website\n"
            "  • photo of team meeting\n"
            "  • excel sheet with student marks\n\n",
            "weak")
        self._set_text_disabled()

    def _set_text_enabled(self):
        self.results_text.config(state="normal")

    def _set_text_disabled(self):
        self.results_text.config(state="disabled")

    def _do_search(self):
        query = self.query_entry.get().strip()
        if not query or query == "Describe what you're looking for...":
            return

        if not self.session.is_valid():
            messagebox.showwarning(
                "Session Expired",
                "Your session has expired. Please log in again."
            )
            self.on_logout()
            return

        mode = self.search_mode.get()

        # Run search in background thread so GUI doesn't freeze
        self.status_var.set("Searching...")
        threading.Thread(
            target=self._run_search_thread,
            args=(query, mode),
            daemon=True
        ).start()

    def _run_search_thread(self, query, mode):
        try:
            results = run_search(query, mode=mode, session=self.session)
            # Update GUI from main thread
            self.after(0, self._display_results, query, mode, results)
        except Exception as e:
            self.after(0, self._display_error, str(e))

    def _display_results(self, query, mode, results):
        self.status_var.set("Status: Index up to date ✓")
        self._set_text_enabled()
        self.results_text.delete("1.0", "end")

        self.results_text.insert("end",
            f"Query: \"{query}\"  |  Mode: {mode}  "
            f"|  {len(results)} results\n\n", "header")

        if not results:
            self.results_text.insert("end",
                "No results found.\n", "warning")
        else:
            if results[0][0] < 0.3:
                self.results_text.insert("end",
                    "⚠  No strong matches found. "
                    "Showing closest results.\n\n", "warning")

            for i, (score, path, result_type) in enumerate(results[:5], 1):
                # Confidence
                if score >= 0.5:
                    conf, tag = "Strong  ", "strong"
                elif score >= 0.35:
                    conf, tag = "Moderate", "moderate"
                else:
                    conf, tag = "Weak    ", "weak"

                type_tag = "doc_tag" if result_type == "document" else "img_tag"
                type_lbl = "DOC" if result_type == "document" else "IMG"

                self.results_text.insert("end", f"  {i}. ", "weak")
                self.results_text.insert("end", f"[{conf}]", tag)
                self.results_text.insert("end", f" [{type_lbl}] ", type_tag)
                self.results_text.insert("end",
                    f"{score:.3f}  ", "weak")
                self.results_text.insert("end",
                    f"{os.path.basename(path)}\n", "path")
                self.results_text.insert("end",
                    f"       {path}\n\n", "weak")

        self._set_text_disabled()

    def _display_error(self, error_msg):
        self.status_var.set("Status: Error")
        self._set_text_enabled()
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end",
            f"Search error:\n{error_msg}\n", "warning")
        self._set_text_disabled()

# =============================================================================
# SCREEN 3 — ADMIN PANEL
# =============================================================================

class AdminPanel(tk.Toplevel):
    """Opens as a separate window on top of the main app."""

    def __init__(self, master, session):
        super().__init__(master)
        self.session = session
        self.title("Admin Panel — KnowWhere")
        self.geometry("560x520")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)
        self._build()
        self._refresh_user_list()

    def _build(self):
        tk.Label(self, text="Admin Panel",
                 bg=BG_DARK, fg=ACCENT_ADMIN,
                 font=FONT_TITLE).pack(pady=(24, 4))
        tk.Label(self, text="Manage employee accounts",
                 bg=BG_DARK, fg=TEXT_MUTED,
                 font=FONT_SMALL).pack(pady=(0, 20))

        # ---- Create User ----
        create_card = tk.LabelFrame(
            self, text="  Create New User  ",
            bg=BG_CARD, fg=ACCENT,
            font=FONT_LABEL,
            highlightthickness=1,
            highlightbackground=BORDER,
            bd=0, padx=20, pady=16
        )
        create_card.pack(fill="x", padx=24, pady=(0, 12))

        row1 = tk.Frame(create_card, bg=BG_CARD)
        row1.pack(fill="x", pady=4)
        tk.Label(row1, text="Username:", bg=BG_CARD,
                 fg=TEXT_MUTED, font=FONT_SMALL,
                 width=12, anchor="w").pack(side="left")
        self.new_username = styled_entry(row1, width=20)
        self.new_username.pack(side="left", padx=4, ipady=4)

        row2 = tk.Frame(create_card, bg=BG_CARD)
        row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Password:", bg=BG_CARD,
                 fg=TEXT_MUTED, font=FONT_SMALL,
                 width=12, anchor="w").pack(side="left")
        self.new_password = styled_entry(row2, show="●", width=20)
        self.new_password.pack(side="left", padx=4, ipady=4)

        row3 = tk.Frame(create_card, bg=BG_CARD)
        row3.pack(fill="x", pady=4)
        tk.Label(row3, text="Role:", bg=BG_CARD,
                 fg=TEXT_MUTED, font=FONT_SMALL,
                 width=12, anchor="w").pack(side="left")
        self.new_role = ttk.Combobox(
            row3, values=["user", "admin"],
            state="readonly", width=10,
            font=FONT_BODY
        )
        self.new_role.set("user")
        self.new_role.pack(side="left", padx=4)

        self.create_msg = tk.Label(
            create_card, text="", bg=BG_CARD,
            fg=TEXT_SUCCESS, font=FONT_SMALL
        )
        self.create_msg.pack(pady=(4, 0))

        styled_button(
            create_card, "Create Account",
            self._create_user,
            color=ACCENT_ADMIN, hover_color="#e0933a",
            width=20
        ).pack(pady=(8, 0))

        # ---- User List ----
        list_card = tk.LabelFrame(
            self, text="  Current Users  ",
            bg=BG_CARD, fg=ACCENT,
            font=FONT_LABEL,
            highlightthickness=1,
            highlightbackground=BORDER,
            bd=0, padx=20, pady=12
        )
        list_card.pack(fill="both", expand=True,
                       padx=24, pady=(0, 12))

        cols = ("Username", "Role")
        self.user_tree = ttk.Treeview(
            list_card, columns=cols,
            show="headings", height=6,
            selectmode="browse"
        )
        for col in cols:
            self.user_tree.heading(col, text=col)
            self.user_tree.column(col, width=200)
        self.user_tree.pack(fill="both", expand=True)

        styled_button(
            list_card, "Delete Selected",
            self._delete_user,
            color="#3a2020", hover_color="#5a2e2e",
            width=18
        ).pack(pady=(10, 0))

        self.delete_msg = tk.Label(
            list_card, text="", bg=BG_CARD,
            fg=TEXT_ERROR, font=FONT_SMALL
        )
        self.delete_msg.pack()

    def _refresh_user_list(self):
        for row in self.user_tree.get_children():
            self.user_tree.delete(row)
        result = list_users(self.session.username)
        if result["success"]:
            for u in result["users"]:
                self.user_tree.insert(
                    "", "end",
                    values=(u["username"], u["role"])
                )

    def _create_user(self):
        username = self.new_username.get().strip()
        password = self.new_password.get().strip()
        role     = self.new_role.get()

        result = create_user(
            self.session.username, username, password, role
        )
        if result["success"]:
            self.create_msg.config(
                text=f"✓ Account '{username}' created.",
                fg=TEXT_SUCCESS
            )
            self.new_username.delete(0, tk.END)
            self.new_password.delete(0, tk.END)
            self._refresh_user_list()
        else:
            self.create_msg.config(
                text=result["reason"], fg=TEXT_ERROR
            )

    def _delete_user(self):
        selected = self.user_tree.focus()
        if not selected:
            self.delete_msg.config(
                text="Select a user first.", fg=TEXT_WARN
            )
            return

        values   = self.user_tree.item(selected, "values")
        username = values[0]

        confirm = messagebox.askyesno(
            "Confirm Delete",
            f"Delete account '{username}'?\nThis cannot be undone."
        )
        if not confirm:
            return

        result = delete_user(self.session.username, username)
        if result["success"]:
            self.delete_msg.config(
                text=f"✓ '{username}' deleted.", fg=TEXT_SUCCESS
            )
            self._refresh_user_list()
        else:
            self.delete_msg.config(
                text=result["reason"], fg=TEXT_ERROR
            )

# =============================================================================
# SCREEN 4 — CHANGE PASSWORD
# =============================================================================

class ChangePasswordDialog(tk.Toplevel):

    def __init__(self, master, session):
        super().__init__(master)
        self.session = session
        self.title("Change Password")
        self.geometry("380x320")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)
        self._build()

    def _build(self):
        tk.Label(self, text="Change Password",
                 bg=BG_DARK, fg=ACCENT,
                 font=FONT_TITLE).pack(pady=(24, 4))
        tk.Label(self, text=f"Account: {self.session.username}",
                 bg=BG_DARK, fg=TEXT_MUTED,
                 font=FONT_SMALL).pack(pady=(0, 20))

        card = tk.Frame(self, bg=BG_CARD, padx=32, pady=24)
        card.pack(fill="x", padx=24)

        for lbl, attr, show in [
            ("Current password", "old_pass", "●"),
            ("New password",     "new_pass", "●"),
            ("Confirm new",      "conf_pass", "●"),
        ]:
            tk.Label(card, text=lbl.upper(), bg=BG_CARD,
                     fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w")
            entry = styled_entry(card, show=show, width=28)
            entry.pack(pady=(2, 12), ipady=4, fill="x")
            setattr(self, attr, entry)

        self.msg_label = tk.Label(
            card, text="", bg=BG_CARD,
            fg=TEXT_ERROR, font=FONT_SMALL
        )
        self.msg_label.pack()

        styled_button(
            card, "Update Password",
            self._update, width=24
        ).pack(fill="x", pady=(8, 0))

    def _update(self):
        old  = self.old_pass.get()
        new  = self.new_pass.get()
        conf = self.conf_pass.get()

        if new != conf:
            self.msg_label.config(
                text="New passwords do not match.", fg=TEXT_ERROR
            )
            return

        result = change_password(self.session.username, old, new)
        if result["success"]:
            self.msg_label.config(
                text="✓ Password updated successfully.", fg=TEXT_SUCCESS
            )
            self.after(1500, self.destroy)
        else:
            self.msg_label.config(
                text=result["reason"], fg=TEXT_ERROR
            )

# =============================================================================
# MAIN APP CONTROLLER
# =============================================================================

class App(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("KnowWhere — Corporate Search System")
        self.geometry("980x680")
        self.minsize(800, 560)
        self.configure(bg=BG_DARK)

        self.session         = None
        self.observer        = None
        self.stop_event      = None
        self.status_var      = tk.StringVar(value="Starting indexing service...")
        self.current_screen  = None

        # Start indexing service in background
        self._start_indexing()

        # Show login screen first
        self._show_login()

        # Clean shutdown on window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- INDEXING SERVICE ----

    def _start_indexing(self):
        def start():
            try:
                self.observer, self.stop_event = start_indexing_service(
                    status_callback=lambda msg: self.status_var.set(msg)
                )
            except Exception as e:
                self.status_var.set(f"Indexing error: {e}")

        threading.Thread(target=start, daemon=True).start()

    # ---- SCREEN SWITCHING ----

    def _clear_screen(self):
        if self.current_screen:
            self.current_screen.destroy()
            self.current_screen = None

    def _show_login(self):
        self._clear_screen()
        screen = LoginScreen(self, on_login_success=self._on_login_success)
        screen.pack(fill="both", expand=True)
        self.current_screen = screen

    def _show_search(self):
        self._clear_screen()
        screen = SearchScreen(
            self,
            session=self.session,
            on_logout=self._on_logout,
            on_admin_panel=self._open_admin_panel,
            on_change_password=self._open_change_password,
            status_var=self.status_var
        )
        screen.pack(fill="both", expand=True)
        self.current_screen = screen

    # ---- EVENT HANDLERS ----

    def _on_login_success(self, session):
        self.session = session
        self._show_search()

    def _on_logout(self):
        if self.session:
            log_logout(self.session.username)
        self.session = None
        self._show_login()

    def _open_admin_panel(self):
        if self.session and self.session.is_admin():
            AdminPanel(self, self.session)

    def _open_change_password(self):
        if self.session:
            ChangePasswordDialog(self, self.session)

    def _on_close(self):
        if self.session:
            log_logout(self.session.username)
        if self.observer and self.stop_event:
            threading.Thread(
                target=stop_indexing_service,
                args=(self.observer, self.stop_event),
                daemon=True
            ).start()
        self.destroy()

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()


