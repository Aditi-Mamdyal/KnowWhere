import os #to walk thru folders
import threading #we need this for two things: running the periodic scan in the background without freezing the GUI, and the Lock + Timer inside the debounce logic.
import subprocess #used to run the Windows command net use to detect if a drive is a network drive.
#verify subprocess
from watchdog.observers import Observer #uses the native Windows API ReadDirectoryChangesW which gets instant notifications from the OS itself.
from watchdog.observers.polling import PollingObserver #it manually checks the folder every N seconds, like a mini incremental scan of its own. like a fallback
from watchdog.events import FileSystemEventHandler #base class we inherit from

# -------- IMPORT EVERYTHING FROM YOUR EXISTING INDEXER --------
from indexing import incremental_index, should_skip, DOC_FOLDER, CHECK_INTERVAL


# =============================================================================
# PART 1 — WATCHDOG EVENT HANDLER
# =============================================================================
# This class tells watchdog WHAT TO DO when a file event fires.
# Watchdog calls on_created / on_modified / on_deleted / on_moved
# automatically whenever the OS reports a file system change.

class DocumentHandler(FileSystemEventHandler): #creating our own class that inherits from watchdog's FileSystemEventHandler. we override the specific methods we care about.

    def __init__(self, debounce_seconds=3, status_callback=None):
        self._debounce = debounce_seconds #stores how many seconds to wait after the last event before actually reindexing. 
        self._timer    = None #special thread that waits a specified amount of time and then calls a function. 
        self._lock     = threading.Lock()  # prevent race between timer resets. is a thread lock — watchdog fires events on its own internal thread, and multiple events can fire almost simultaneously, so we need a lock to prevent two events from both trying to reset the timer at the exact same millisecond, which could cause a race condition.
        self._callback = status_callback   # optional GUI label update function

    # ---- called when a NEW file appears in the watched folder ----
    def on_created(self, event):
        if event.is_directory:
            return
        if should_skip(os.path.basename(event.src_path)):
            return
        print(f"  [WATCH] New file: {event.src_path}")
        self._schedule_reindex()

    # ---- called when an EXISTING file is written to / saved ----
    def on_modified(self, event):
        if event.is_directory:
            return
        if should_skip(os.path.basename(event.src_path)):
            return
        print(f"  [WATCH] Modified: {event.src_path}")
        self._schedule_reindex()

    # ---- called when a file is DELETED ----
    def on_deleted(self, event):
        if event.is_directory:
            return
        if should_skip(os.path.basename(event.src_path)):
            return
        print(f"  [WATCH] Deleted: {event.src_path}")
        self._schedule_reindex()

    # ---- called when a file is RENAMED or MOVED ----
    def on_moved(self, event):
        if event.is_directory:
            return
        print(f"  [WATCH] Moved: {event.src_path} → {event.dest_path}")
        self._schedule_reindex()

    # ---- debounce logic ----
    def _schedule_reindex(self):
        # Every time an event fires, we cancel the previous pending timer
        # and start a fresh one. This means the actual reindex only runs
        # after things have been quiet for `debounce_seconds`.
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce, self._run_reindex)
            self._timer.start()

    def _run_reindex(self):
        if self._callback:
            self._callback("Status: Indexing changes...")
        try:
            incremental_index()
            print("  [WATCH] Re-index complete.")
            if self._callback:
                self._callback("Status: Index up to date ✓")
        except Exception as e:
            print(f"  [WATCH] Re-index failed: {e}")
            if self._callback:
                self._callback(f"Status: Error — {e}")


# =============================================================================
# PART 2 — DETECT IF THE FOLDER IS A NETWORK DRIVE
# =============================================================================
# Watchdog's native Observer uses Windows ReadDirectoryChangesW API.
# This works great for local drives but silently fails on:
#   - UNC paths like \\server\share
#   - Mapped network drives like Z:\
# For those we fall back to PollingObserver which manually checks for
# changes every N seconds — slower but works everywhere.

def is_network_path(path: str) -> bool:
    # UNC path — always a network path
    if path.startswith("\\\\"):
        return True

    drive = os.path.splitdrive(path)[0]   # e.g. "Z:"
    if not drive:
        return False

    # Ask Windows if this drive letter is a mapped network drive
    try:
        result = subprocess.run(
            f'net use {drive}',
            capture_output=True, text=True, shell=True
        )
        return result.returncode == 0   # returncode 0 = Windows recognises it as a network drive
    except Exception:
        return False   # if we can't tell, assume local and use native observer


# =============================================================================
# PART 3 — PERIODIC SAFETY SCAN (background thread)
# =============================================================================
# Watchdog is great but can miss events when:
#   - VPN drops and reconnects mid-session
#   - Antivirus briefly blocks ReadDirectoryChangesW
#   - Files are changed by a remote machine on a shared drive
#
# This function runs incremental_index() once every CHECK_INTERVAL seconds
# as a guaranteed safety net. It uses threading.Event.wait() instead of
# time.sleep() so it wakes up immediately when stop_event is set (clean shutdown).

def _periodic_scan_loop(stop_event: threading.Event, status_callback=None):
    while not stop_event.is_set():
        # Wait for CHECK_INTERVAL seconds OR until stop_event is set
        stop_event.wait(CHECK_INTERVAL)

        if stop_event.is_set():
            break   # app is closing — exit cleanly

        print("\n[PERIODIC] Safety scan running...")
        if status_callback:
            status_callback("Status: Periodic scan running...")
        try:
            incremental_index()
            print("[PERIODIC] Done.")
            if status_callback:
                status_callback("Status: Index up to date ✓")
        except Exception as e:
            print(f"[PERIODIC] Error: {e}")


# =============================================================================
# PART 4 — START / STOP THE FULL SERVICE
# =============================================================================
# This is the single function your GUI (or any other file) calls.
# It starts both watchdog AND the periodic safety scan together.
# Returns (observer, stop_event) so the caller can shut everything down cleanly.

def start_indexing_service(status_callback=None):
    """
    Start watchdog watcher + periodic safety scan.

    status_callback: optional function(str) — called with status messages.
                     Pass a lambda that updates a Tkinter label, for example.

    Returns:
        observer   — the watchdog Observer (call .stop() + .join() to shut down)
        stop_event — threading.Event (call .set() to stop the periodic scan)
    """

    # ---- Step 1: choose the right observer type ----
    handler = DocumentHandler(debounce_seconds=3, status_callback=status_callback)

    if is_network_path(DOC_FOLDER):
        # Poll every 30 seconds for network drives
        observer = PollingObserver(timeout=30)
        print(f"[SERVICE] Network drive detected — using PollingObserver (30s poll)")
        print(f"[SERVICE] Watching: {DOC_FOLDER}")
    else:
        observer = Observer()
        print(f"[SERVICE] Local drive detected — using native Observer")
        print(f"[SERVICE] Watching: {DOC_FOLDER}")

    # recursive=True means subfolders are watched too
    observer.schedule(handler, DOC_FOLDER, recursive=True)
    observer.start()

    # ---- Step 2: start periodic safety scan in a background thread ----
    stop_event = threading.Event()
    scan_thread = threading.Thread(
        target=_periodic_scan_loop,
        args=(stop_event, status_callback),
        daemon=True   # thread dies automatically if main process exits
    )
    scan_thread.start()

    print(f"[SERVICE] Periodic safety scan every {CHECK_INTERVAL}s also active.")

    if status_callback:
        status_callback("Status: Watching for changes...")

    return observer, stop_event


def stop_indexing_service(observer, stop_event):
    """
    Cleanly shut down both the watchdog observer and the periodic scan thread.
    Call this when the app window closes.
    """
    print("\n[SERVICE] Shutting down indexing service...")
    stop_event.set()      # wake up and exit the periodic scan loop
    observer.stop()       # tell watchdog to stop watching
    observer.join()       # wait for watchdog thread to fully finish
    print("[SERVICE] Indexing service stopped.")


# =============================================================================
# STANDALONE MODE — run this file directly to use without a GUI
# =============================================================================
# If you run `python indexing_service.py` directly (no GUI),
# this block starts the service and keeps it alive until Ctrl+C.

if __name__ == "__main__":
    import time

    # Run one full index on startup so the index is fresh immediately
    print("[STARTUP] Running initial index...")
    incremental_index()

    # Start the full service
    observer, stop_event = start_indexing_service()

    print("\n[STARTUP] Service running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_indexing_service(observer, stop_event)