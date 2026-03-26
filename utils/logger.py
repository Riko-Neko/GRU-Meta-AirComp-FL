# utils/logger.py
import datetime
import hashlib
import os
from utils.log_plotter import plot_round_metrics_from_log


def _shorten_name_component(component: str, max_len: int) -> str:
    text = str(component)
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    if len(text) <= max_len:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    reserve = len("_H") + len(digest)
    if max_len <= reserve:
        return f"H{digest}"[:max_len]
    keep = max_len - reserve
    return f"{text[:keep]}_H{digest}"


class Logger:
    def __init__(self, log_file=None, *, config=None, log_dir: str = "./log"):
        # 从 * 之后的参数必须用关键字传递
        if config is not None:
            os.makedirs(log_dir, exist_ok=True)

            prefix = config.log_prefix()
            optimizer_tag = config.optimizer_tag()
            timestamp_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            ext = ".log"
            # Keep basename safely below common filesystem limits.
            max_basename_len = 240
            suffix = f"_{optimizer_tag}_{timestamp_tag}{ext}"
            safe_prefix = _shorten_name_component(prefix, max_basename_len - len(suffix))
            log_file = os.path.join(log_dir, f"{safe_prefix}{suffix}")

        self._path = log_file
        self._stem = os.path.splitext(os.path.basename(log_file))[0] if log_file else None
        self.log_file = open(log_file, "w") if log_file else None

    @property
    def path(self):
        return self._path

    @property
    def stem(self):
        return self._stem

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
