import csv
import os
import time
from datetime import datetime


class ViolationTracker:

    def __init__(self, log_file="data/logs.csv", cooldown=5):

        self.log_file = log_file
        self.cooldown = cooldown
        self.last_logged_time = 0

        # Create data folder if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(log_file):

            with open(log_file, "w", newline="") as file:

                writer = csv.writer(file)

                writer.writerow([
                    "timestamp",
                    "violation_count"
                ])

    def log_violation(self, violation_count):

        current_time = time.time()

        # Don't log too frequently
        if current_time - self.last_logged_time < self.cooldown:
            return False

        if violation_count > 0:

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with open(
                self.log_file,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    timestamp,
                    violation_count
                ])

            self.last_logged_time = current_time

            return True

        return False