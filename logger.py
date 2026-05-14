import csv
from datetime import datetime
from pathlib import Path


LOG_FILE = Path("data/logs.csv")
LOG_FILE.parent.mkdir(exist_ok=True)


def save_log(role, question, answer, latency):
    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)

        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "role",
                    "question",
                    "answer",
                    "latency",
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(),
                role,
                question,
                answer,
                latency,
            ]
        )
