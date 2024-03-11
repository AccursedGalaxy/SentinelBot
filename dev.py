import os
import subprocess
import sys
import threading
import time

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from logger_config import setup_logging

watcher_logger = setup_logging("watcher", "blue")

dev_file = "main.py"


class MyHandler(PatternMatchingEventHandler):
    patterns = ["*.py"]

    def __init__(self):
        super().__init__()
        self.restart_script()  # Start main.py when the handler is initialized

    def process(self, event):
        watcher_logger.info(f"Event type: {event.event_type}  path: {event.src_path}")
        self.restart_script()

    def restart_script(self):
        subprocess.run(["pkill", "-f", dev_file])
        time.sleep(1)  # Give it a moment to terminate

        try:
            process = subprocess.Popen(
                [sys.executable, dev_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                env=os.environ.copy(),
            )
            watcher_logger.info(f"Restarted main.py with PID: {process.pid}")

            # Handle stdout and stderr in separate threads
            threading.Thread(target=self.log_output, args=(process.stdout,)).start()
            threading.Thread(target=self.log_output, args=(process.stderr,)).start()

        except Exception as e:
            watcher_logger.error(f"Failed to restart main.py: {e}")

    def log_output(self, stream):
        for line in iter(stream.readline, ""):
            watcher_logger.info(line.strip())

    def on_modified(self, event):
        self.process(event)


if __name__ == "__main__":
    path = sys.path[0]
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    watcher_logger.info("Starting watcher...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
        watcher_logger.info("Watcher script stopped.")
