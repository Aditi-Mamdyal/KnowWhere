"""
gui_dark.py  —  KnowWhere  (Dark Theme)
FIXES:
  1. Change password: grab_set()+transient() so button always clickable
  2. Results are clickable — click path to open file
  3. os.chdir at startup fixes launch.bat indexing issue
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import os

# FIX 4: Set working directory to project folder so all relative paths work
# when launched via .bat or desktop shortcut (not just VS Code)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from auth import (verify_login, create_user, delete_user, list_users,
                  change_password, log_logout, Session)
from searching import run_search
from indexing_service import start_indexing_service, stop_indexing_service

# ── COLOURS ────────────────────────────────────────────────────────────────────
BG_DARK=  "#0f1117"; BG_CARD="#1a1d27"; BG_INPUT="#252836"
ACCENT=   "#4f8ef7"; ACCENT_HOVER="#3a7be0"; ACCENT_ADMIN="#f7a44f"
TEXT_PRIMARY="#e8eaf0"; TEXT_MUTED="#7a7f94"; TEXT_SUCCESS="#4fba74"
TEXT_WARN="#f7c44f"; TEXT_ERROR="#f74f4f"; BORDER="#2e3146"
RESULT_DOC="#4f8ef7"; RESULT_IMG="#a44ff7"; CLICKABLE="#4fba74"

FONT_TITLE=("Georgia",22,"bold"); FONT_SUB=("Georgia",13)
FONT_BODY=("Consolas",11); FONT_SMALL=("Consolas",9)
FONT_LABEL=("Consolas",11,"bold"); FONT_BTN=("Consolas",11,"bold")

# ── HELPERS ────────────────────────────────────────────────────────────────────
def styled_entry(parent, show=None, width=30):
    return tk.Entry(parent, show=show, width=width, bg=BG_INPUT, fg=TEXT_PRIMARY,
                    insertbackground=TEXT_PRIMARY, relief="flat", font=FONT_BODY,
                    highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT)

def styled_button(parent, text, command, color=ACCENT, hover_color=ACCENT_HOVER, width=18):
    btn = tk.Button(parent, text=text, command=command, bg=color, fg=TEXT_PRIMARY,
                    activebackground=hover_color, activeforeground=TEXT_PRIMARY,
                    relief="flat", font=FONT_BTN, cursor="hand2", width=width, pady=6)
    btn.bind("<Enter>", lambda e: btn.config(bg=hover_color))
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    return btn

# ── LOGIN ──────────────────────────────────────────────────────────────────────
class LoginScreen(tk.Frame):
    def __init__(self, master, on_login_success):
        super().__init__(master, bg=BG_DARK)
        self.on_login_success = on_login_success
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)
        card = tk.Frame(self, bg=BG_CARD, padx=48, pady=42,
                        highlightthickness=1, highlightbackground=BORDER)
        card.grid(row=0, column=0)

        tk.Label(card, text="⬡", bg=BG_CARD, fg=ACCENT, font=("Georgia",36)).pack(pady=(0,4))
        tk.Label(card, text="KnowWhere", bg=BG_CARD, fg=TEXT_PRIMARY, font=FONT_TITLE).pack()
        tk.Label(card, text="Corporate Document & Image Search",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(pady=(2,28))

        tk.Label(card, text="USERNAME", bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w")
        self.u = styled_entry(card, width=32)
        self.u.pack(pady=(2,14), ipady=6, fill="x")

        tk.Label(card, text="PASSWORD", bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w")
        self.p = styled_entry(card, show="●", width=32)
        self.p.pack(pady=(2,6), ipady=6, fill="x")

        self.err = tk.Label(card, text="", bg=BG_CARD, fg=TEXT_ERROR, font=FONT_SMALL)
        self.err.pack(pady=(0,16))

        styled_button(card, "LOGIN", self._login, width=32).pack(ipady=2, fill="x")
        tk.Label(card, text="Contact IT admin if you need an account.",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(pady=(18,0))

        self.p.bind("<Return>", lambda e: self._login())
        self.u.bind("<Return>", lambda e: self.p.focus())
        self.u.focus()

    def _login(self):
        r = verify_login(self.u.get().strip(), self.p.get().strip())
        if r["success"]:
            self.on_login_success(Session(r["username"], r["role"]))
        else:
            self.err.config(text=r["reason"])
            self.p.delete(0, tk.END); self.p.focus()

# ── SEARCH SCREEN ──────────────────────────────────────────────────────────────
class SearchScreen(tk.Frame):
    def __init__(self, master, session, on_logout, on_admin, on_chpw, status_var):
        super().__init__(master, bg=BG_DARK)
        self.session=session; self.on_logout=on_logout; self.on_admin=on_admin
        self.on_chpw=on_chpw; self.status_var=status_var
        self.mode=tk.StringVar(value="both")
        self._paths=[]   # parallel list of paths matching clickable tag ranges
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1); self.rowconfigure(1, weight=1)

        # topbar
        tb = tk.Frame(self, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        tb.grid(row=0, column=0, sticky="ew"); tb.columnconfigure(1, weight=1)
        tk.Label(tb, text="⬡  KnowWhere", bg=BG_CARD, fg=ACCENT,
                 font=("Georgia",14,"bold"), padx=20, pady=12).grid(row=0, column=0, sticky="w")
        tk.Label(tb, textvariable=self.status_var, bg=BG_CARD, fg=TEXT_MUTED,
                 font=FONT_SMALL).grid(row=0, column=1)

        right = tk.Frame(tb, bg=BG_CARD, padx=16); right.grid(row=0, column=2, sticky="e")
        rc = ACCENT_ADMIN if self.session.is_admin() else TEXT_MUTED
        tk.Label(right, text=f"  {self.session.username}  [{self.session.role}]",
                 bg=BG_CARD, fg=rc, font=FONT_SMALL).pack(side="left", padx=4)
        if self.session.is_admin():
            styled_button(right, "Admin Panel", self.on_admin,
                          color=ACCENT_ADMIN, hover_color="#e0933a", width=12).pack(side="left", padx=4)
        styled_button(right, "Change Password", self.on_chpw,
                      color=BG_INPUT, hover_color=BORDER, width=16).pack(side="left", padx=4)
        styled_button(right, "Logout", self.on_logout,
                      color="#3a2020", hover_color="#5a2e2e", width=8).pack(side="left", padx=4)

        # middle
        mid = tk.Frame(self, bg=BG_DARK, padx=32, pady=24)
        mid.grid(row=1, column=0, sticky="nsew")
        mid.columnconfigure(0, weight=1); mid.rowconfigure(2, weight=1)

        sb = tk.Frame(mid, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        sb.grid(row=0, column=0, sticky="ew", pady=(0,12)); sb.columnconfigure(0, weight=1)

        self.q = tk.Entry(sb, bg=BG_CARD, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                          relief="flat", font=("Georgia",14), highlightthickness=0)
        self.q.grid(row=0, column=0, sticky="ew", padx=20, pady=14, ipady=4)
        self.q.insert(0, "Describe what you're looking for..."); self.q.config(fg=TEXT_MUTED)

        def fi(e):
            if self.q.get()=="Describe what you're looking for...":
                self.q.delete(0,tk.END); self.q.config(fg=TEXT_PRIMARY)
        def fo(e):
            if not self.q.get():
                self.q.insert(0,"Describe what you're looking for..."); self.q.config(fg=TEXT_MUTED)
        self.q.bind("<FocusIn>",fi); self.q.bind("<FocusOut>",fo)
        self.q.bind("<Return>", lambda e: self._search())
        styled_button(sb, "Search", self._search, width=10).grid(row=0, column=1, padx=8, pady=8)

        mf = tk.Frame(mid, bg=BG_DARK); mf.grid(row=1, column=0, sticky="w", pady=(0,16))
        tk.Label(mf, text="Search in:", bg=BG_DARK, fg=TEXT_MUTED, font=FONT_SMALL).pack(side="left", padx=(0,10))
        for val,lbl in [("both","All"),("documents","Documents"),("images","Images")]:
            tk.Radiobutton(mf, text=lbl, variable=self.mode, value=val, bg=BG_DARK,
                           fg=TEXT_PRIMARY, selectcolor=BG_INPUT, activebackground=BG_DARK,
                           activeforeground=ACCENT, font=FONT_SMALL, cursor="hand2").pack(side="left", padx=8)

        rf = tk.Frame(mid, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        rf.grid(row=2, column=0, sticky="nsew"); rf.rowconfigure(0,weight=1); rf.columnconfigure(0,weight=1)

        self.txt = scrolledtext.ScrolledText(rf, bg=BG_CARD, fg=TEXT_PRIMARY, font=FONT_BODY,
                                             relief="flat", state="disabled", wrap="word",
                                             padx=20, pady=16, spacing1=4, spacing2=2, cursor="arrow")
        self.txt.grid(row=0, column=0, sticky="nsew")

        self.txt.tag_config("strong",   foreground=TEXT_SUCCESS)
        self.txt.tag_config("moderate", foreground=TEXT_WARN)
        self.txt.tag_config("weak",     foreground=TEXT_MUTED)
        self.txt.tag_config("doc_tag",  foreground=RESULT_DOC)
        self.txt.tag_config("img_tag",  foreground=RESULT_IMG)
        self.txt.tag_config("path",     foreground=TEXT_PRIMARY)
        self.txt.tag_config("warning",  foreground=TEXT_ERROR)
        self.txt.tag_config("header",   foreground=ACCENT, font=("Consolas",11,"bold"))
        # FIX 2: clickable paths
        self.txt.tag_config("clickable", foreground=CLICKABLE, underline=True)
        self.txt.tag_bind("clickable", "<Button-1>", self._click)
        self.txt.tag_bind("clickable", "<Enter>",  lambda e: self.txt.config(cursor="hand2"))
        self.txt.tag_bind("clickable", "<Leave>",  lambda e: self.txt.config(cursor="arrow"))

        self._welcome()

    def _welcome(self):
        self.txt.config(state="normal")
        self.txt.insert("end","Welcome to KnowWhere\n\n","header")
        self.txt.insert("end",
            "Type a natural language description in the search bar.\n"
            "Examples:\n  • budget report Q3\n  • tictalk counselling website\n"
            "  • photo of team meeting\n  • excel sheet with student marks\n\n"
            "Click any green path to open the file directly.\n","weak")
        self.txt.config(state="disabled")

    def _search(self):
        q = self.q.get().strip()
        if not q or q=="Describe what you're looking for...": return
        if not self.session.is_valid():
            messagebox.showwarning("Session Expired","Please log in again."); self.on_logout(); return
        self.status_var.set("Searching..."); self._paths=[]
        threading.Thread(target=self._thread, args=(q, self.mode.get()), daemon=True).start()

    def _thread(self, q, mode):
        try:
            res = run_search(q, mode=mode, session=self.session)
            self.after(0, self._show, q, mode, res)
        except Exception as e:
            self.after(0, self._err, str(e))

    def _show(self, q, mode, results):
        self.status_var.set("Status: Index up to date ✓")
        self._paths=[]
        self.txt.config(state="normal"); self.txt.delete("1.0","end")
        self.txt.insert("end", f"Query: \"{q}\"  |  Mode: {mode}  |  {len(results)} results\n\n","header")

        if not results:
            self.txt.insert("end","No results found.\n","warning")
        else:
            if results[0][0]<0.3:
                self.txt.insert("end","⚠  No strong matches. Showing closest results.\n\n","warning")
            self.txt.insert("end","  Click a green path to open the file.\n\n","weak")

            for i,(score,path,rtype) in enumerate(results[:10],1):
                conf,tag = ("Strong  ","strong") if score>=0.5 else \
                           ("Moderate","moderate") if score>=0.35 else ("Weak    ","weak")
                tlbl = "DOC" if rtype=="document" else "IMG"
                ttag = "doc_tag" if rtype=="document" else "img_tag"

                self.txt.insert("end",f"  {i}. ","weak")
                self.txt.insert("end",f"[{conf}]",tag)
                self.txt.insert("end",f" [{tlbl}] ",ttag)
                self.txt.insert("end",f"{score:.3f}  ","weak")
                self.txt.insert("end",f"{os.path.basename(path)}\n","path")
                self.txt.insert("end",f"       {path}\n\n","clickable")
                self._paths.append(path)

        self.txt.config(state="disabled")

    def _click(self, event):
        idx    = self.txt.index(f"@{event.x},{event.y}")
        ranges = self.txt.tag_ranges("clickable")
        pairs  = [(ranges[i],ranges[i+1]) for i in range(0,len(ranges),2)]
        for i,(s,e) in enumerate(pairs):
            if self.txt.compare(idx,">=",s) and self.txt.compare(idx,"<",e):
                if i < len(self._paths):
                    p = self._paths[i]
                    if os.path.exists(p):
                        try: os.startfile(p)
                        except Exception as ex: messagebox.showerror("Error",str(ex))
                    else:
                        messagebox.showwarning("Not Found",f"File not found:\n{p}")
                return

    def _err(self, msg):
        self.status_var.set("Status: Error")
        self.txt.config(state="normal"); self.txt.delete("1.0","end")
        self.txt.insert("end",f"Search error:\n{msg}\n","warning")
        self.txt.config(state="disabled")

# ── ADMIN PANEL ────────────────────────────────────────────────────────────────
class AdminPanel(tk.Toplevel):
    def __init__(self, master, session):
        super().__init__(master)
        self.session=session
        self.title("Admin Panel — KnowWhere"); self.geometry("560x520")
        self.configure(bg=BG_DARK); self.resizable(False,False)
        self.transient(master); self.grab_set()
        self._build(); self._refresh()

    def _build(self):
        tk.Label(self,text="Admin Panel",bg=BG_DARK,fg=ACCENT_ADMIN,font=FONT_TITLE).pack(pady=(24,4))
        tk.Label(self,text="Manage employee accounts",bg=BG_DARK,fg=TEXT_MUTED,font=FONT_SMALL).pack(pady=(0,20))

        cc=tk.LabelFrame(self,text="  Create New User  ",bg=BG_CARD,fg=ACCENT,font=FONT_LABEL,
                         highlightthickness=1,highlightbackground=BORDER,bd=0,padx=20,pady=16)
        cc.pack(fill="x",padx=24,pady=(0,12))

        for lt,attr,sh in [("Username:","new_u",None),("Password:","new_p","●")]:
            r=tk.Frame(cc,bg=BG_CARD); r.pack(fill="x",pady=4)
            tk.Label(r,text=lt,bg=BG_CARD,fg=TEXT_MUTED,font=FONT_SMALL,width=12,anchor="w").pack(side="left")
            e=styled_entry(r,show=sh,width=20); e.pack(side="left",padx=4,ipady=4); setattr(self,attr,e)

        r3=tk.Frame(cc,bg=BG_CARD); r3.pack(fill="x",pady=4)
        tk.Label(r3,text="Role:",bg=BG_CARD,fg=TEXT_MUTED,font=FONT_SMALL,width=12,anchor="w").pack(side="left")
        self.role=ttk.Combobox(r3,values=["user","admin"],state="readonly",width=10,font=FONT_BODY)
        self.role.set("user"); self.role.pack(side="left",padx=4)

        self.cmsg=tk.Label(cc,text="",bg=BG_CARD,fg=TEXT_SUCCESS,font=FONT_SMALL); self.cmsg.pack(pady=(4,0))
        styled_button(cc,"Create Account",self._create,color=ACCENT_ADMIN,hover_color="#e0933a",width=20).pack(pady=(8,0))

        lc=tk.LabelFrame(self,text="  Current Users  ",bg=BG_CARD,fg=ACCENT,font=FONT_LABEL,
                         highlightthickness=1,highlightbackground=BORDER,bd=0,padx=20,pady=12)
        lc.pack(fill="both",expand=True,padx=24,pady=(0,12))
        self.tree=ttk.Treeview(lc,columns=("Username","Role"),show="headings",height=6,selectmode="browse")
        for c in ("Username","Role"): self.tree.heading(c,text=c); self.tree.column(c,width=200)
        self.tree.pack(fill="both",expand=True)
        styled_button(lc,"Delete Selected",self._delete,color="#3a2020",hover_color="#5a2e2e",width=18).pack(pady=(10,0))
        self.dmsg=tk.Label(lc,text="",bg=BG_CARD,fg=TEXT_ERROR,font=FONT_SMALL); self.dmsg.pack()

    def _refresh(self):
        for r in self.tree.get_children(): self.tree.delete(r)
        res=list_users(self.session.username)
        if res["success"]:
            for u in res["users"]: self.tree.insert("","end",values=(u["username"],u["role"]))

    def _create(self):
        res=create_user(self.session.username,self.new_u.get().strip(),self.new_p.get().strip(),self.role.get())
        if res["success"]:
            self.cmsg.config(text=f"✓ '{self.new_u.get()}' created.",fg=TEXT_SUCCESS)
            self.new_u.delete(0,tk.END); self.new_p.delete(0,tk.END); self._refresh()
        else: self.cmsg.config(text=res["reason"],fg=TEXT_ERROR)

    def _delete(self):
        sel=self.tree.focus()
        if not sel: self.dmsg.config(text="Select a user first.",fg=TEXT_WARN); return
        uname=self.tree.item(sel,"values")[0]
        if not messagebox.askyesno("Confirm",f"Delete '{uname}'?\nCannot be undone."): return
        res=delete_user(self.session.username,uname)
        if res["success"]: self.dmsg.config(text=f"✓ '{uname}' deleted.",fg=TEXT_SUCCESS); self._refresh()
        else: self.dmsg.config(text=res["reason"],fg=TEXT_ERROR)

# ── CHANGE PASSWORD ────────────────────────────────────────────────────────────
class ChangePasswordDialog(tk.Toplevel):
    def __init__(self, master, session):
        super().__init__(master)
        self.session=session
        self.title("Change Password"); self.geometry("380x340")
        self.configure(bg=BG_DARK); self.resizable(False,False)
        # FIX 1: these two lines are the fix — dialog stays on top and focusable
        self.transient(master); self.grab_set()
        self._build()
        self.after(100, self.old.focus)

    def _build(self):
        tk.Label(self,text="Change Password",bg=BG_DARK,fg=ACCENT,font=FONT_TITLE).pack(pady=(24,4))
        tk.Label(self,text=f"Account: {self.session.username}",bg=BG_DARK,fg=TEXT_MUTED,font=FONT_SMALL).pack(pady=(0,20))
        card=tk.Frame(self,bg=BG_CARD,padx=32,pady=24); card.pack(fill="x",padx=24)

        for lt,attr in [("Current password","old"),("New password","new1"),("Confirm new","new2")]:
            tk.Label(card,text=lt.upper(),bg=BG_CARD,fg=TEXT_MUTED,font=FONT_SMALL).pack(anchor="w")
            e=styled_entry(card,show="●",width=28); e.pack(pady=(2,12),ipady=4,fill="x"); setattr(self,attr,e)

        self.new2.bind("<Return>", lambda e: self._submit())
        self.msg=tk.Label(card,text="",bg=BG_CARD,fg=TEXT_ERROR,font=FONT_SMALL); self.msg.pack()
        styled_button(card,"Update Password",self._submit,width=24).pack(fill="x",pady=(8,0))

    def _submit(self):
        o=self.old.get(); n=self.new1.get(); c=self.new2.get()
        if not o or not n: self.msg.config(text="All fields required.",fg=TEXT_ERROR); return
        if n!=c: self.msg.config(text="New passwords do not match.",fg=TEXT_ERROR); return
        res=change_password(self.session.username,o,n)
        if res["success"]:
            self.msg.config(text="✓ Password updated.",fg=TEXT_SUCCESS)
            self.after(1500,self.destroy)
        else: self.msg.config(text=res["reason"],fg=TEXT_ERROR)

# ── APP ────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KnowWhere — Corporate Search System")
        self.geometry("980x680"); self.minsize(800,560); self.configure(bg=BG_DARK)
        self.session=None; self.observer=None; self.stop_event=None
        self.status_var=tk.StringVar(value="Starting indexing service...")
        self.current_screen=None
        self._start_indexing(); self._show_login()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _start_indexing(self):
        def go():
            try:
                self.observer,self.stop_event=start_indexing_service(
                    status_callback=lambda m: self.status_var.set(m))
            except Exception as e: self.status_var.set(f"Indexing error: {e}")
        threading.Thread(target=go,daemon=True).start()

    def _clear(self):
        if self.current_screen: self.current_screen.destroy(); self.current_screen=None

    def _show_login(self):
        self._clear()
        s=LoginScreen(self, on_login_success=self._logged_in)
        s.pack(fill="both",expand=True); self.current_screen=s

    def _show_search(self):
        self._clear()
        s=SearchScreen(self, session=self.session, on_logout=self._logout,
                       on_admin=self._admin, on_chpw=self._chpw,
                       status_var=self.status_var)
        s.pack(fill="both",expand=True); self.current_screen=s

    def _logged_in(self,session): self.session=session; self._show_search()
    def _logout(self):
        if self.session: log_logout(self.session.username)
        self.session=None; self._show_login()
    def _admin(self):
        if self.session and self.session.is_admin(): AdminPanel(self,self.session)
    def _chpw(self):
        if self.session: ChangePasswordDialog(self,self.session)
    def _close(self):
        if self.session: log_logout(self.session.username)
        if self.observer and self.stop_event:
            threading.Thread(target=stop_indexing_service,
                             args=(self.observer,self.stop_event),daemon=True).start()
        self.destroy()

if __name__=="__main__":
    App().mainloop()