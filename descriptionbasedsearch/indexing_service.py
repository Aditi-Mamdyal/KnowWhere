import os
import threading
import subprocess
import time
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

from indexing import (incremental_index, should_skip,
                      get_doc_folder, CHECK_INTERVAL)
from image_indexing import index_images

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ── SINGLE LOCK — only one indexing run at a time ─────────────────────────────
_index_lock = threading.Lock()


# ── UNIFIED INDEXING ──────────────────────────────────────────────────────────

def run_dual_index(status_callback=None):
    """
    Runs document + image indexing, then reloads the in-memory search index.
    Non-blocking acquire — if already running, skips silently.

    FIX: After indexing completes, reload_doc_index() is called so that
    searching.py's in-memory embeddings/metadata reflect the changes.
    Without this call, newly added, modified, or deleted files would never
    appear in search results until the process was restarted.
    """
    acquired = _index_lock.acquire(blocking=False)
    if not acquired:
        print("[SERVICE] Indexing already running — skipping.")
        return

    try:
        if status_callback:
            status_callback("Status: Indexing changes...")

        folder = get_doc_folder()
        if not folder or not os.path.isdir(folder):
            print("[SERVICE] No folder configured — skipping.")
            return

        print("\n[SERVICE] Updating text index...")
        incremental_index()

        print("[SERVICE] Updating image index...")
        index_images()

        # ── FIX: reload search module's in-memory index ───────────────────────
        # searching.py loads embeddings/metadata once at import time.
        # incremental_index() writes new data to disk but searching.py still
        # holds the old arrays in RAM.  We must tell it to re-read from disk.
        try:
            from searching import reload_doc_index
            reload_doc_index()
        except Exception as e:
            print(f"[SERVICE] Warning: could not reload search index: {e}")

        print("[SERVICE] All indices up to date.")
        if status_callback:
            status_callback("Status: Index up to date ✓")

    except Exception as e:
        print(f"[SERVICE] Indexing failed: {e}")
        if status_callback:
            status_callback(f"Status: Error — {e}")
    finally:
        _index_lock.release()


# ── WATCHDOG EVENT HANDLER ────────────────────────────────────────────────────

class DocumentHandler(FileSystemEventHandler):
    """
    Watchdog calls these methods when files change.
    Debounce prevents multiple rapid events from triggering multiple reindexes.
    """

    def __init__(self, debounce_seconds=5, status_callback=None):
        self._debounce = debounce_seconds
        self._timer    = None
        self._lock     = threading.Lock()
        self._callback = status_callback

    def _is_relevant(self, path: str) -> bool:
        filename = os.path.basename(path)
        ext      = os.path.splitext(filename)[1].lower()
        if not should_skip(filename):
            return True
        if ext in IMAGE_EXTS:
            return True
        return False

    def on_created(self, event):
        if event.is_directory: return
        if not self._is_relevant(event.src_path): return
        print(f"  [WATCH] New: {event.src_path}")
        self._schedule()

    def on_modified(self, event):
        if event.is_directory: return
        if not self._is_relevant(event.src_path): return
        print(f"  [WATCH] Modified: {event.src_path}")
        self._schedule()

    def on_deleted(self, event):
        if event.is_directory: return
        if not self._is_relevant(event.src_path): return
        print(f"  [WATCH] Deleted: {event.src_path}")
        self._schedule()

    def on_moved(self, event):
        if event.is_directory: return
        print(f"  [WATCH] Moved: {event.src_path} → {event.dest_path}")
        self._schedule()

    def _schedule(self):
        """Reset debounce timer on every event."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(
                self._debounce,
                lambda: threading.Thread(
                    target=run_dual_index,
                    args=(self._callback,),
                    daemon=True
                ).start()
            )
            self._timer.start()


# ── NETWORK DRIVE DETECTION ───────────────────────────────────────────────────

def is_network_path(path: str) -> bool:
    if path.startswith("\\\\"):
        return True
    drive = os.path.splitdrive(path)[0]
    if not drive:
        return False
    try:
        result = subprocess.run(f'net use {drive}',
                                capture_output=True, text=True, shell=True)
        return result.returncode == 0
    except Exception:
        return False


# ── PERIODIC SAFETY SCAN ─────────────────────────────────────────────────────

def _periodic_scan_loop(stop_event: threading.Event, status_callback=None):
    while not stop_event.is_set():
        stop_event.wait(CHECK_INTERVAL)
        if stop_event.is_set():
            break
        print("\n[PERIODIC] Safety scan running...")
        run_dual_index(status_callback)


# ── SERVICE CONTROL ───────────────────────────────────────────────────────────

def start_indexing_service(status_callback=None):
    """
    Step 1 — Run a full index first (blocking, in the calling thread).
    Step 2 — Only after that completes, start watchdog + periodic scan.

    This prevents watchdog from firing events during the initial scan
    which was causing duplicate indexing.

    Returns (observer, stop_event).
    """
    folder = get_doc_folder()
    if not folder or not os.path.isdir(folder):
        print("[SERVICE] No folder configured — service not started.")
        if status_callback:
            status_callback("Status: No folder configured")
        return None, threading.Event()

    # ── STEP 1: full index first ──────────────────────────────────────────────
    print("[SERVICE] Running initial index before starting watcher...")
    if status_callback:
        status_callback("Status: Building index...")

    run_dual_index(status_callback)   # blocks until done; also reloads search index

    # ── STEP 2: start watchdog ────────────────────────────────────────────────
    handler = DocumentHandler(debounce_seconds=5, status_callback=status_callback)

    if is_network_path(folder):
        observer = PollingObserver(timeout=30)
        print("[SERVICE] Network drive — PollingObserver active")
    else:
        observer = Observer()
        print("[SERVICE] Local drive — native Observer active")

    observer.schedule(handler, folder, recursive=True)
    observer.start()
    print(f"[SERVICE] Watching: {folder}")

    # ── STEP 3: start periodic safety scan ───────────────────────────────────
    stop_event  = threading.Event()
    scan_thread = threading.Thread(
        target=_periodic_scan_loop,
        args=(stop_event, status_callback),
        daemon=True
    )
    scan_thread.start()
    print(f"[SERVICE] Periodic safety scan every {CHECK_INTERVAL}s active.")

    if status_callback:
        status_callback("Status: Watching for changes ✓")

    return observer, stop_event


def stop_indexing_service(observer, stop_event):
    print("\n[SERVICE] Shutting down...")
    if stop_event:
        stop_event.set()
    if observer:
        observer.stop()
        observer.join()
    print("[SERVICE] Stopped.")


# ── STANDALONE MODE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    observer, stop_event = start_indexing_service()
    print(f"\n[STARTUP] Monitoring: {get_doc_folder()}")
    print("[STARTUP] Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_indexing_service(observer, stop_event)