import json
import os
from datetime import datetime
from typing import Dict

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "decisions.log")


def log_decision(entry: Dict) -> None:
    """Append a decision entry to the log file as a JSON line."""
    record = {"timestamp": datetime.now().isoformat(), **entry}
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_recent_log(n: int = 10) -> list:
    """Return the last n entries from the log file."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines[-n:]]
