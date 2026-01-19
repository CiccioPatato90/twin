import os


class Logger:
    _best_value = float("inf")

    def __init__(self, filename="evaluations.txt"):
        self.filename = filename

        # Check if file exists; if not, create it and start at 0
        if not os.path.exists(self.filename):
            self._write_value(0)

    def _read_value(self):
        """Helper to read the integer from the file."""
        try:
            with open(self.filename, "r") as f:
                content = f.read().strip()
                # Return 0 if file is empty, otherwise parse int
                return int(content) if content else 0
        except ValueError:
            return 0  # Fallback if file contains junk

    def _write_value(self, value):
        """Helper to write the integer to the file."""
        with open(self.filename, "a") as f:
            f.write(str(value))
            f.write(str("\n"))

    def log_best(self, value, prefix):
        if value < Logger._best_value:
            Logger._best_value = value
        self._write_value(f"{prefix}: {Logger._best_value}")

    def sync_best(self, value):
        if value < Logger._best_value:
            Logger._best_value = value

    def get_evaluations(self):
        return self._read_value()


# --- Usage Example ---
# evals = FileEvaluations()
# print(evals.get_evaluations()) # e.g., 0
# evals.increment()
# print(evals.get_evaluations()) # e.g., 1
