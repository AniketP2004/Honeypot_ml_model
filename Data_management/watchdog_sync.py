import sys
import time
import threading
import logging
import os
import config
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sync_lock = threading.Lock()

# Create log directory if none exist
os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)


# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# File handler
file_handler = logging.FileHandler(config.LOG_FILE)
file_handler.setLevel(logging.INFO)


# Stream handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

#format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

#Attach both handlers
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# retrain lock
retrain_lock = threading.Lock()
retrain_timer = None

def trigger_retrain():
    logger.info("Starting auto retraining.")
    try:
        result1 = subprocess.run(
            [sys.executable, r"C:\Users\project\Desktop\Honeypot_New_Repo\Data_preprocessing\data_cleaning.py"],
            capture_output=True, text = True
        )
        if result1.returncode != 0:
            logger.error(f"data_cleaning failed: {result1.stderr}")
            return
        logger.info("Data cleaning complete")

        result2= subprocess.run(
            [sys.executable, r"C:\Users\project\Desktop\Honeypot_New_Repo\Training\train.py"],
            capture_output= True, text= True
        )

        if result2.returncode != 0:
            logger.error(f"Training failed: {result2.stderr}")
            return
        logger.info("Retraining complete")

    except Exception as e:
        logger.error(f"Retrain pipleine crashed: {e}")

def schedule_retrain():
    global retrain_timer
    with retrain_lock:
        if retrain_timer:
            retrain_timer.cancel()
        retrain_timer = threading.Timer(30.0, trigger_retrain)
        retrain_timer.start()

def get_last_position():
    try:
        with open(config.TRACKER_FILE, 'r') as f:
            lines = f.read()

            if not lines.strip():
                return 0
            
            final = int(lines)
            return final
    except FileNotFoundError:
        return 0
    

def get_new_packets(start_pos):
    try:
        with open(config.SOURCE_LOG, 'r') as f:
            f.seek(start_pos)
            content = f.read()
            new_lines = f.tell()
            return (content, new_lines)
    except FileNotFoundError:
        return (None, start_pos)

def append_to_destination(new_line_count):
    
    try:
        os.makedirs(os.path.dirname(config.DESTINATION_LOG), exist_ok=True)
        with open (config.DESTINATION_LOG, 'a') as f:
            f.write(new_line_count)
        logger.info("Successfully appended new packets")
    except Exception as e:
        logger.error(f"Failed to append to destination: {e}")


def update_position(new_pos):
    try:
        with open (config.TRACKER_FILE, 'w') as f:
            f.write(str(new_pos))
        logger.info(f"Position updated to: {new_pos}")
    except Exception as e:
        logger.error(f"Failed to update position: {e}")


def sync_new_packets():
    with sync_lock:
        current_pos = get_last_position()
        content, new_pos = get_new_packets(current_pos)
        if content == None:
            return logger.error("No content found")
        elif content == "":
            return logger.info("No new packets")
        append_to_destination(content)
        update_position(new_pos)
        logger.info(f"Appended {new_pos - current_pos} bytes")
        schedule_retrain()

class LogFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        src = os.path.abspath(event.src_path)
        target = os.path.abspath(config.SOURCE_LOG)
        if src!= target:
            return
        logger.info(f"File modification detected: {event.src_path}")
        sync_new_packets()

def main():
    source_dir = os.path.dirname(config.SOURCE_LOG)
    if not os.path.exists(config.SOURCE_LOG):
        logger.error("Source file not found")
        sys.exit(1)

    event_handler = LogFileHandler()
    observer = Observer()
    observer.schedule(event_handler, source_dir, recursive=False)
    observer.start()

    logger.info("Monitoring started")
    logger.info(f"Source: {config.SOURCE_LOG}")
    logger.info(f"Destination: {config.DESTINATION_LOG}")
    logger.info(f"Current position: {get_last_position()}")

    try:
        while True:
            time.sleep(config.POLL_INTERVAL)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Monitoring stopped")
    
    observer.join()


if __name__ == "__main__":
    main()