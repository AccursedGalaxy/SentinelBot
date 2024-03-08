import subprocess
import sys
import time

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from logger_config import setup_logging

logger = setup_logging()


class MyHandler(PatternMatchingEventHandler):
    patterns = ["*.py"]

    def process(self, event):
        logger.info(f"Event type: {event.event_type}  path: {event.src_path}")
        subprocess.run(["pkill", "-f", "main.py"])
        subprocess.Popen([sys.executable, "main.py"])

    def on_modified(self, event):
        self.process(event)


if __name__ == "__main__":
    path = sys.path[0]
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
        logger.info("Watcher script stopped.")
