# utils/logger.py
import datetime
import os
from utils.log_plotter import plot_round_metrics_from_log


class Logger:
    def __init__(self, log_file=None, *, config=None, log_dir: str = "./log"):
        # 从 * 之后的参数必须用关键字传递
        if config is not None:
            os.makedirs(log_dir, exist_ok=True)

            prefix = config.log_prefix()
            fp = config.fingerprint()

            log_file = os.path.join(log_dir, f"{prefix}_{fp}.log")

        self._path = log_file
        self.log_file = open(log_file, "w") if log_file else None

    def info(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()
        if self._path:
            try:
                fig_path = plot_round_metrics_from_log(self._path, figs_root="./figs")
                if fig_path:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] Round metric figure saved: {fig_path}")
            except Exception as exc:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Failed to plot round metrics: {exc}")
