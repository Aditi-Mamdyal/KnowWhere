import os
import threading
import subprocess
import time
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

# -------- IMPORT BOTH INDEXERS --------
from indexing import incremental_index, should_skip, DOC_FOLDER, CHECK_INTERVAL
from image_indexing import index_images, DOC_FOLDER as IMAGE_FOLDER

# Image extensions watchdog should care about
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# =============================================================================
# PART 1 — UNIFIED INDEXING
# =============================================================================

def run_dual_index(status_callback=None):
    """Runs both document and image indexing sequentially."""
    if status_callback:
        status_callback("Status: Indexing changes...")
    try:
        print("\n[SERVICE] Phase 1: Updating text index...")
        incremental_index()

        print("[SERVICE] Phase 2: Updating image index...")
        index_images()

        print("[SERVICE] All indices up to date.")
        if status_callback:
            status_callback("Status: Index up to date ✓")

    except Exception as e:
        print(f"[SERVICE] Indexing failed: {e}")
        if status_callback:
            status_callback(f"Status: Error — {e}")


# =============================================================================
# PART 2 — WATCHDOG EVENT HANDLER
# =============================================================================

class DocumentHandler(FileSystemEventHandler):

    def __init__(self, debounce_seconds=5, status_callback=None):
        self._debounce = debounce_seconds
        self._timer    = None
        self._lock     = threading.Lock()
        self._callback = status_callback

    def _is_relevant(self, path: str) -> bool:
        """Return True if this file should trigger a reindex."""
        filename = os.path.basename(path)
        ext      = os.path.splitext(filename)[1].lower()

        # text document — use should_skip logic
        if not should_skip(filename):
            return True

        # image file — check against image extensions
        if ext in IMAGE_EXTS:
            return True

        return False

    def on_created(self, event):
        if event.is_directory:
            return
        if not self._is_relevant(event.src_path):
            return
        print(f"  [WATCH] New file: {event.src_path}")
        self._schedule_reindex()

    def on_modified(self, event):
        if event.is_directory:
            return
        if not self._is_relevant(event.src_path):
            return
        print(f"  [WATCH] Modified: {event.src_path}")
        self._schedule_reindex()

    def on_deleted(self, event):
        if event.is_directory:
            return
        if not self._is_relevant(event.src_path):
            return
        print(f"  [WATCH] Deleted: {event.src_path}")
        self._schedule_reindex()

    def on_moved(self, event):
        if event.is_directory:
            return
        print(f"  [WATCH] Moved: {event.src_path} → {event.dest_path}")
        self._schedule_reindex()

    def _schedule_reindex(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce, self._run_reindex)
            self._timer.start()

    def _run_reindex(self):
        run_dual_index(self._callback)


# =============================================================================
# PART 3 — NETWORK DRIVE DETECTION
# =============================================================================

def is_network_path(path: str) -> bool:
    if path.startswith("\\\\"):
        return True
    drive = os.path.splitdrive(path)[0]
    if not drive:
        return False
    try:
        result = subprocess.run(
            f'net use {drive}',
            capture_output=True, text=True, shell=True
        )
        return result.returncode == 0
    except Exception:
        return False


# =============================================================================
# PART 4 — PERIODIC SAFETY SCAN
# =============================================================================

def _periodic_scan_loop(stop_event: threading.Event, status_callback=None):
    while not stop_event.is_set():
        stop_event.wait(CHECK_INTERVAL)
        if stop_event.is_set():
            break
        print("\n[PERIODIC] Safety scan running...")
        run_dual_index(status_callback)


# =============================================================================
# PART 5 — SERVICE CONTROL
# =============================================================================

def start_indexing_service(status_callback=None):
    handler = DocumentHandler(debounce_seconds=5, status_callback=status_callback)

    if is_network_path(DOC_FOLDER):
        observer = PollingObserver(timeout=30)
        print("[SERVICE] Network drive — PollingObserver active")
    else:
        observer = Observer()
        print("[SERVICE] Local drive — native Observer active")

    # Watch the document folder
    observer.schedule(handler, DOC_FOLDER, recursive=True)

    # If image folder is different from doc folder, watch it separately
    # In your case both are the same (D:\coding\college) so this is a no-op
    # but kept here for when they differ
    if IMAGE_FOLDER != DOC_FOLDER:
        observer.schedule(handler, IMAGE_FOLDER, recursive=True)
        print(f"[SERVICE] Also watching image folder: {IMAGE_FOLDER}")

    observer.start()
    print(f"[SERVICE] Watching: {DOC_FOLDER}")

    stop_event  = threading.Event()
    scan_thread = threading.Thread(
        target=_periodic_scan_loop,
        args=(stop_event, status_callback),
        daemon=True
    )
    scan_thread.start()

    print(f"[SERVICE] Periodic safety scan every {CHECK_INTERVAL}s active.")

    if status_callback:
        status_callback("Status: Watching for changes...")

    return observer, stop_event


def stop_indexing_service(observer, stop_event):
    print("\n[SERVICE] Shutting down...")
    stop_event.set()
    observer.stop()
    observer.join()
    print("[SERVICE] Stopped.")


# =============================================================================
# STANDALONE MODE
# =============================================================================

if __name__ == "__main__":
    print("[STARTUP] Running initial full index (Text + Images)...")
    run_dual_index()

    observer, stop_event = start_indexing_service()
    print(f"\n[STARTUP] Monitoring: {DOC_FOLDER}")
    print("[STARTUP] Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_indexing_service(observer, stop_event)