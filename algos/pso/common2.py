import os


class Logger:
    _best_value = float("inf")
    _initialized = False

    def __init__(self, filename="evaluations.txt"):
        self.filename = filename

        # Only scan the file once per program execution to sync memory
        if not Logger._initialized:
            self._recover_best_from_file()
            Logger._initialized = True

        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                pass

    def _recover_best_from_file(self):
        """Scans the file for the lowest value ever recorded."""
        if not os.path.exists(self.filename):
            return

        found_val = float("inf")
        try:
            with open(self.filename, "r") as f:
                for line in f:
                    # Parse lines like "SWARM: 12.345"
                    if ":" in line:
                        parts = line.split(":")
                        try:
                            val = float(parts[-1].strip())
                            if val < found_val:
                                found_val = val
                        except ValueError:
                            continue

            # Update the static memory
            if found_val < Logger._best_value:
                Logger._best_value = found_val
        except Exception:
            pass

    def log_best(self, value, prefix):
        # Only log if we beat the historical best (Monotonic)
        if value < Logger._best_value:
            Logger._best_value = value
            self._write_value(f"{prefix}: {Logger._best_value}")

    def sync_best(self, value):
        if value < Logger._best_value:
            Logger._best_value = value

    def _write_value(self, text):
        with open(self.filename, "a") as f:
            f.write(str(text) + "\n")

    def get_evaluations(self):
        # (Optional) Implement line counting if needed
        return 0
